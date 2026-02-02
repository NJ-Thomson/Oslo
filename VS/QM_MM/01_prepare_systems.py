#!/usr/bin/env python3
"""
01_prepare_systems.py - Prepare System Snapshots for QM/MM

This script:
1. Extracts a snapshot from equilibrated MD trajectory
2. Identifies covalent bond atoms (Cys-SG, warhead-C)
3. Creates clean starting structures for umbrella sampling
4. Writes system metadata

Usage:
    python 01_prepare_systems.py --config config.yaml

Output:
    07_qmmm_results/{receptor}/{inhibitor}/{warhead}/
        ├── snapshot.gro        - Starting structure
        ├── topology/           - Copy of topology files
        └── system_info.json    - Atom indices and metadata
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

try:
    import MDAnalysis as mda
    HAS_MDA = True
except ImportError:
    HAS_MDA = False
    print("WARNING: MDAnalysis not installed. Using fallback methods.")


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_gmx() -> Optional[str]:
    """Find GROMACS executable."""
    for cmd in ['gmx', 'gmx_mpi']:
        try:
            result = subprocess.run([cmd, '--version'], capture_output=True)
            if result.returncode == 0:
                return cmd
        except FileNotFoundError:
            continue
    return None


def extract_snapshot_gmx(
    traj_file: Path,
    top_file: Path,
    output_gro: Path,
    time_ps: float = -1
) -> bool:
    """Extract a snapshot from trajectory using GROMACS."""
    gmx = find_gmx()
    if not gmx:
        print("ERROR: GROMACS not found")
        return False
    
    # Get structure file (gro or tpr)
    gro_file = traj_file.parent / 'npt.gro'
    tpr_file = traj_file.parent / 'npt.tpr'
    
    struct_file = tpr_file if tpr_file.exists() else gro_file
    
    cmd = [
        gmx, 'trjconv',
        '-s', str(struct_file),
        '-f', str(traj_file),
        '-o', str(output_gro),
        '-dump', str(time_ps),  # -1 = last frame
        '-pbc', 'mol',
    ]
    
    # Run with "System" selection
    try:
        result = subprocess.run(
            cmd,
            input='0\n',  # Select System
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def find_covalent_atoms_mda(
    gro_file: Path,
    top_file: Path,
    cys_resid: int,
    warhead: str,
    config: dict
) -> Dict:
    """Find covalent bond atoms using MDAnalysis."""
    if not HAS_MDA:
        return find_covalent_atoms_simple(gro_file, cys_resid, warhead, config)
    
    # Load structure
    u = mda.Universe(str(gro_file))
    
    # Find cysteine sulfur (may be in CYL/CA2 residue)
    cys_sel = u.select_atoms(f'resid {cys_resid} and name SG')
    if len(cys_sel) == 0:
        # Try selecting by residue name pattern
        cys_sel = u.select_atoms(f'resid {cys_resid} and (name SG or name S)')
    
    if len(cys_sel) == 0:
        raise ValueError(f"Could not find sulfur in residue {cys_resid}")
    
    sg_atom = cys_sel[0]
    sg_index = sg_atom.index  # 0-based
    
    # Find reactive carbon (bonded to SG)
    warhead_cfg = config['systems']['warheads'][warhead]
    reactive_c_name = warhead_cfg['reactive_carbon']
    
    # The reactive carbon should be in the same residue as modified cysteine
    c1_sel = u.select_atoms(f'resid {cys_resid} and name {reactive_c_name}')
    
    if len(c1_sel) == 0:
        raise ValueError(f"Could not find {reactive_c_name} in residue {cys_resid}")
    
    c1_atom = c1_sel[0]
    c1_index = c1_atom.index  # 0-based
    
    # Get all atoms in the covalent residue
    cyl_atoms = u.select_atoms(f'resid {cys_resid}')
    cyl_indices = [a.index for a in cyl_atoms]
    
    # Get residue info
    resname = sg_atom.resname
    
    return {
        'cys_resid': cys_resid,
        'cys_resname': resname,
        'sg_index_0based': sg_index,
        'sg_index_1based': sg_index + 1,
        'c1_index_0based': c1_index,
        'c1_index_1based': c1_index + 1,
        'cyl_atom_indices_0based': cyl_indices,
        'cyl_atom_indices_1based': [i + 1 for i in cyl_indices],
        'n_atoms_total': len(u.atoms),
        'reactive_carbon_name': reactive_c_name,
    }


def find_covalent_atoms_simple(
    gro_file: Path,
    cys_resid: int,
    warhead: str,
    config: dict
) -> Dict:
    """Find covalent bond atoms by parsing GRO file directly."""
    warhead_cfg = config['systems']['warheads'][warhead]
    reactive_c_name = warhead_cfg['reactive_carbon']
    
    sg_index = None
    c1_index = None
    cyl_indices = []
    resname = None
    n_atoms = 0
    
    with open(gro_file) as f:
        lines = f.readlines()
        n_atoms = int(lines[1].strip())
        
        for i, line in enumerate(lines[2:-1], start=0):  # Skip header and box
            if len(line) < 20:
                continue
            
            # GRO format: resid(5), resname(5), atomname(5), atomnr(5), x, y, z
            res_num = int(line[0:5])
            res_name = line[5:10].strip()
            atom_name = line[10:15].strip()
            
            if res_num == cys_resid:
                cyl_indices.append(i)
                resname = res_name
                
                if atom_name == 'SG' or atom_name == 'S':
                    sg_index = i
                if atom_name == reactive_c_name:
                    c1_index = i
    
    if sg_index is None or c1_index is None:
        raise ValueError(f"Could not find SG or {reactive_c_name} in residue {cys_resid}")
    
    return {
        'cys_resid': cys_resid,
        'cys_resname': resname,
        'sg_index_0based': sg_index,
        'sg_index_1based': sg_index + 1,
        'c1_index_0based': c1_index,
        'c1_index_1based': c1_index + 1,
        'cyl_atom_indices_0based': cyl_indices,
        'cyl_atom_indices_1based': [i + 1 for i in cyl_indices],
        'n_atoms_total': n_atoms,
        'reactive_carbon_name': reactive_c_name,
    }


def prepare_system(
    input_dir: Path,
    output_dir: Path,
    receptor: str,
    inhibitor: str,
    warhead: str,
    config: dict
) -> bool:
    """Prepare a single system for QM/MM."""
    
    system_input = input_dir / receptor / inhibitor / warhead
    system_output = output_dir / receptor / inhibitor / warhead
    
    print(f"  Processing: {receptor}/{inhibitor}/{warhead}")
    
    # Check input files
    gro_file = system_input / 'npt.gro'
    top_file = system_input / 'topol.top'
    traj_file = system_input / 'npt.xtc'
    
    if not gro_file.exists():
        print(f"    ERROR: {gro_file} not found")
        return False
    
    # Create output directory
    system_output.mkdir(parents=True, exist_ok=True)
    
    # Extract snapshot (last frame from NPT equilibration)
    snapshot_gro = system_output / 'snapshot.gro'
    
    if traj_file.exists():
        print("    Extracting snapshot from trajectory...")
        success = extract_snapshot_gmx(traj_file, top_file, snapshot_gro, time_ps=-1)
        if not success:
            print("    Falling back to NPT .gro file")
            shutil.copy(gro_file, snapshot_gro)
    else:
        print("    Using NPT equilibrated structure")
        shutil.copy(gro_file, snapshot_gro)
    
    # Copy topology
    topology_dir = system_output / 'topology'
    topology_dir.mkdir(exist_ok=True)
    
    shutil.copy(top_file, topology_dir / 'topol.top')
    
    # Copy force field if present
    ff_dir = system_input / f'amber99sb-ildn-*.ff'
    import glob
    for ff in glob.glob(str(system_input / 'amber99sb-ildn-*.ff')):
        ff_path = Path(ff)
        if ff_path.is_dir():
            shutil.copytree(ff_path, topology_dir / ff_path.name, dirs_exist_ok=True)
    
    # Copy include files
    for itp in system_input.glob('*.itp'):
        shutil.copy(itp, topology_dir / itp.name)
    
    # Find covalent bond atoms
    cys_resid = config['systems']['receptors'][receptor]['cys_resid']
    
    print("    Identifying covalent bond atoms...")
    try:
        atom_info = find_covalent_atoms_mda(
            snapshot_gro, top_file, cys_resid, warhead, config
        )
    except Exception as e:
        print(f"    ERROR finding atoms: {e}")
        return False
    
    # Save system info
    system_info = {
        'receptor': receptor,
        'inhibitor': inhibitor,
        'warhead': warhead,
        'warhead_name': config['systems']['warheads'][warhead]['name'],
        'covalent_atoms': atom_info,
        'input_path': str(system_input),
        'files': {
            'snapshot': str(snapshot_gro),
            'topology': str(topology_dir / 'topol.top'),
        }
    }
    
    with open(system_output / 'system_info.json', 'w') as f:
        json.dump(system_info, f, indent=2)
    
    print(f"    ✓ SG atom: {atom_info['sg_index_1based']}")
    print(f"    ✓ C1 atom: {atom_info['c1_index_1based']}")
    print(f"    ✓ CYL residue: {atom_info['cys_resname']} ({len(atom_info['cyl_atom_indices_0based'])} atoms)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare systems for QM/MM")
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'config.yaml')
    parser.add_argument('--receptors', nargs='+', default=None)
    parser.add_argument('--inhibitors', nargs='+', default=None)
    parser.add_argument('--warheads', nargs='+', default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    base_dir = args.config.parent
    
    input_dir = Path(config['paths']['input_dir'])
    if not input_dir.is_absolute():
        input_dir = (base_dir / input_dir).resolve()
    
    output_dir = Path(config['paths']['output_dir'])
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()
    
    print("=" * 60)
    print("Step 1: Prepare Systems for QM/MM")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Get system lists
    receptors = args.receptors or list(config['systems']['receptors'].keys())
    inhibitors = args.inhibitors or config['systems']['inhibitors']
    warheads = args.warheads or list(config['systems']['warheads'].keys())
    
    success_count = 0
    total_count = 0
    
    for receptor in receptors:
        for inhibitor in inhibitors:
            for warhead in warheads:
                total_count += 1
                if prepare_system(
                    input_dir, output_dir, receptor, inhibitor, warhead, config
                ):
                    success_count += 1
    
    print()
    print(f"Prepared {success_count}/{total_count} systems successfully")
    
    return success_count == total_count


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
