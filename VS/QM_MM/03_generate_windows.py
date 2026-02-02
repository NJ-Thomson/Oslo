#!/usr/bin/env python3
"""
03_generate_windows.py - Generate Umbrella Sampling Windows

Creates starting structures for each umbrella window along the S-C reaction
coordinate by:
1. Generating target distances from r_min to r_max
2. Creating steered MD input to pull along coordinate (or use direct placement)
3. Setting up window directories

Usage:
    python 03_generate_windows.py --config config.yaml

Output:
    07_qmmm_results/{system}/windows/window_XXX/
        ├── conf.gro          - Starting structure for window
        └── window_info.json  - Window metadata
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    import MDAnalysis as mda
    HAS_MDA = True
except ImportError:
    HAS_MDA = False


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_gmx() -> Optional[str]:
    """Find GROMACS executable."""
    for cmd in ['gmx', 'gmx_mpi', 'gmx_mimic']:
        try:
            result = subprocess.run([cmd, '--version'], capture_output=True)
            if result.returncode == 0:
                return cmd
        except FileNotFoundError:
            continue
    return None


def generate_window_distances(config: dict) -> List[float]:
    """Generate target distances for umbrella windows."""
    umb_cfg = config['umbrella_sampling']['windows']
    
    r_min = umb_cfg['r_min_nm']
    r_max = umb_cfg['r_max_nm']
    spacing = umb_cfg['spacing_nm']
    
    # Generate evenly spaced windows
    n_windows = int(np.round((r_max - r_min) / spacing)) + 1
    distances = np.linspace(r_min, r_max, n_windows)
    
    return distances.tolist()


def measure_sc_distance(gro_file: Path, sg_index: int, c1_index: int) -> float:
    """Measure current S-C distance in nm."""
    if HAS_MDA:
        u = mda.Universe(str(gro_file))
        sg_pos = u.atoms[sg_index].position
        c1_pos = u.atoms[c1_index].position
        dist_A = np.linalg.norm(sg_pos - c1_pos)
        return dist_A / 10.0  # Convert to nm
    else:
        # Parse GRO file manually
        with open(gro_file) as f:
            lines = f.readlines()
        
        def get_coords(atom_idx):
            line = lines[atom_idx + 2]  # +2 for header lines
            x = float(line[20:28])
            y = float(line[28:36])
            z = float(line[36:44])
            return np.array([x, y, z])
        
        sg_pos = get_coords(sg_index)
        c1_pos = get_coords(c1_index)
        return np.linalg.norm(sg_pos - c1_pos)


def create_window_structure_smd(
    input_gro: Path,
    output_gro: Path,
    sg_index: int,
    c1_index: int,
    target_dist_nm: float,
    gmx: str,
    work_dir: Path
) -> bool:
    """Create window structure using steered MD."""
    # This would run a short steered MD to move to target distance
    # For now, we'll use direct coordinate manipulation as a simpler approach
    return create_window_structure_direct(
        input_gro, output_gro, sg_index, c1_index, target_dist_nm
    )


def create_window_structure_direct(
    input_gro: Path,
    output_gro: Path,
    sg_index: int,
    c1_index: int,
    target_dist_nm: float
) -> bool:
    """Create window structure by direct coordinate manipulation."""
    if not HAS_MDA:
        # Fallback: just copy the structure (will be adjusted during equilibration)
        shutil.copy(input_gro, output_gro)
        return True
    
    u = mda.Universe(str(input_gro))
    
    # Get current positions
    sg_atom = u.atoms[sg_index]
    c1_atom = u.atoms[c1_index]
    
    sg_pos = sg_atom.position
    c1_pos = c1_atom.position
    
    # Current distance and direction
    vec = c1_pos - sg_pos
    current_dist = np.linalg.norm(vec)
    direction = vec / current_dist
    
    # Target distance in Angstroms
    target_dist_A = target_dist_nm * 10.0
    
    # Calculate displacement needed
    displacement = target_dist_A - current_dist
    
    # Move C1 and everything bonded to it (the ligand part)
    # For simplicity, we move just C1 - the system will relax during equilibration
    # A more sophisticated approach would move the entire warhead/ligand
    
    # Get all atoms in the same residue as C1 that are "beyond" C1 from SG
    cyl_resid = c1_atom.resid
    ligand_atoms = u.select_atoms(f'resid {cyl_resid}')
    
    # Find atoms to move (those closer to C1 than to SG in the bonding sense)
    # Simple heuristic: move atoms that are farther from SG than C1 is
    atoms_to_move = []
    for atom in ligand_atoms:
        dist_to_sg = np.linalg.norm(atom.position - sg_pos)
        if dist_to_sg >= current_dist * 0.9:  # Move atoms beyond ~90% of current S-C distance
            atoms_to_move.append(atom.index)
    
    # Apply displacement
    for idx in atoms_to_move:
        u.atoms[idx].position += direction * displacement
    
    # Write output
    u.atoms.write(str(output_gro))
    
    return True


def create_pull_index(
    gro_file: Path,
    output_ndx: Path,
    sg_index: int,
    c1_index: int
) -> bool:
    """Create index file with pull groups."""
    with open(output_ndx, 'w') as f:
        f.write("[ QM_SG ]\n")
        f.write(f"{sg_index + 1}\n\n")  # 1-based for GROMACS
        f.write("[ QM_C1 ]\n")
        f.write(f"{c1_index + 1}\n\n")
    return True


def setup_window(
    system_dir: Path,
    window_idx: int,
    target_dist_nm: float,
    config: dict
) -> bool:
    """Set up a single umbrella window."""
    
    # Load system info
    with open(system_dir / 'system_info.json') as f:
        system_info = json.load(f)
    
    sg_index = system_info['covalent_atoms']['sg_index_0based']
    c1_index = system_info['covalent_atoms']['c1_index_0based']
    
    # Create window directory
    window_dir = system_dir / 'windows' / f'window_{window_idx:03d}'
    window_dir.mkdir(parents=True, exist_ok=True)
    
    # Input structure (from equilibrated snapshot)
    input_gro = system_dir / 'snapshot.gro'
    output_gro = window_dir / 'conf.gro'
    
    # Create starting structure for this window
    success = create_window_structure_direct(
        input_gro, output_gro, sg_index, c1_index, target_dist_nm
    )
    
    if not success:
        return False
    
    # Measure actual distance achieved
    actual_dist = measure_sc_distance(output_gro, sg_index, c1_index)
    
    # Create pull index file
    create_pull_index(output_gro, window_dir / 'pull.ndx', sg_index, c1_index)
    
    # Save window info
    window_info = {
        'window_index': window_idx,
        'target_distance_nm': target_dist_nm,
        'target_distance_A': target_dist_nm * 10,
        'initial_distance_nm': actual_dist,
        'initial_distance_A': actual_dist * 10,
        'sg_index_1based': sg_index + 1,
        'c1_index_1based': c1_index + 1,
    }
    
    with open(window_dir / 'window_info.json', 'w') as f:
        json.dump(window_info, f, indent=2)
    
    return True


def process_system(system_dir: Path, config: dict) -> bool:
    """Generate all umbrella windows for a system."""
    
    # Load system info
    info_file = system_dir / 'system_info.json'
    if not info_file.exists():
        print(f"    ERROR: system_info.json not found")
        return False
    
    with open(info_file) as f:
        system_info = json.load(f)
    
    print(f"  Processing: {system_info['receptor']}/{system_info['inhibitor']}/{system_info['warhead']}")
    
    # Generate window distances
    distances = generate_window_distances(config)
    n_windows = len(distances)
    
    print(f"    Generating {n_windows} windows from {distances[0]:.3f} to {distances[-1]:.3f} nm")
    
    # Get current S-C distance
    gro_file = system_dir / 'snapshot.gro'
    sg_idx = system_info['covalent_atoms']['sg_index_0based']
    c1_idx = system_info['covalent_atoms']['c1_index_0based']
    current_dist = measure_sc_distance(gro_file, sg_idx, c1_idx)
    
    print(f"    Current S-C distance: {current_dist:.3f} nm ({current_dist*10:.2f} Å)")
    
    # Create windows directory
    windows_dir = system_dir / 'windows'
    if windows_dir.exists():
        shutil.rmtree(windows_dir)
    windows_dir.mkdir()
    
    # Generate each window
    success_count = 0
    for i, target_dist in enumerate(distances):
        if setup_window(system_dir, i, target_dist, config):
            success_count += 1
    
    print(f"    ✓ Created {success_count}/{n_windows} windows")
    
    # Save window manifest
    manifest = {
        'n_windows': n_windows,
        'distances_nm': distances,
        'current_sc_distance_nm': current_dist,
    }
    with open(system_dir / 'windows_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return success_count == n_windows


def main():
    parser = argparse.ArgumentParser(description="Generate umbrella sampling windows")
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'config.yaml')
    parser.add_argument('--r-min', type=float, default=None, help='Minimum distance (nm)')
    parser.add_argument('--r-max', type=float, default=None, help='Maximum distance (nm)')
    parser.add_argument('--spacing', type=float, default=None, help='Window spacing (nm)')
    parser.add_argument('--receptors', nargs='+', default=None)
    parser.add_argument('--inhibitors', nargs='+', default=None)
    parser.add_argument('--warheads', nargs='+', default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    base_dir = args.config.parent
    
    # Override config with command line args
    if args.r_min is not None:
        config['umbrella_sampling']['windows']['r_min_nm'] = args.r_min
    if args.r_max is not None:
        config['umbrella_sampling']['windows']['r_max_nm'] = args.r_max
    if args.spacing is not None:
        config['umbrella_sampling']['windows']['spacing_nm'] = args.spacing
    
    output_dir = Path(config['paths']['output_dir'])
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()
    
    print("=" * 60)
    print("Step 3: Generate Umbrella Sampling Windows")
    print("=" * 60)
    
    umb_cfg = config['umbrella_sampling']['windows']
    print(f"Distance range: {umb_cfg['r_min_nm']:.2f} - {umb_cfg['r_max_nm']:.2f} nm")
    print(f"Spacing: {umb_cfg['spacing_nm']:.3f} nm")
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
                system_dir = output_dir / receptor / inhibitor / warhead
                if system_dir.exists() and (system_dir / 'system_info.json').exists():
                    total_count += 1
                    if process_system(system_dir, config):
                        success_count += 1
    
    print()
    print(f"Generated windows for {success_count}/{total_count} systems")
    
    return success_count == total_count


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
