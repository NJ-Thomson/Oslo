#!/usr/bin/env python3
"""
02_define_qm_region.py - Define QM Region for QM/MM Calculations

Defines which atoms are treated quantum mechanically:
- Core: Cysteine backbone + sidechain + warhead reactive atoms
- Extended: First shell of ligand + H-bond partners (optional)

The QM/MM boundary uses link atoms (hydrogen caps) at C-C bonds.

Usage:
    python 02_define_qm_region.py --config config.yaml [--qm-size minimal|extended]

Output:
    07_qmmm_results/{system}/qm_region.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml

try:
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import distance_array
    import numpy as np
    HAS_MDA = True
except ImportError:
    HAS_MDA = False
    mda = None  # Placeholder
    import numpy as np
    print("WARNING: MDAnalysis not installed. Limited functionality.")


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# Cysteine backbone and sidechain atoms (standard)
CYSTEINE_ATOMS = {
    'backbone': ['N', 'H', 'CA', 'HA', 'C', 'O'],
    'sidechain': ['CB', 'HB2', 'HB3', 'SG'],  # Note: HB1/HB2 or HB2/HB3 naming varies
}

# Warhead atoms to include in QM (minimal set near reactive center)
WARHEAD_ATOMS_MINIMAL = {
    'acry': ['C1', 'C2', 'H3', 'H4', 'N3', 'H6', 'C4', 'O5'],  # Acrylamide core
    'chlo': ['C1', 'H3', 'H4', 'N3', 'H6', 'C4', 'O5'],  # Chloroacetamide (Cl departs)
}


def get_bonded_atoms_mda(universe, atom_indices: List[int], n_bonds: int = 1) -> Set[int]:
    """Get atoms within n bonds of given atoms using MDAnalysis."""
    if not hasattr(universe, 'bonds') or len(universe.bonds) == 0:
        # No bond information, use distance-based approximation
        return get_nearby_atoms_distance(universe, atom_indices, cutoff=1.6 * n_bonds)
    
    result = set(atom_indices)
    frontier = set(atom_indices)
    
    for _ in range(n_bonds):
        new_frontier = set()
        for idx in frontier:
            atom = universe.atoms[idx]
            for bond in atom.bonds:
                for bonded_atom in bond.atoms:
                    if bonded_atom.index not in result:
                        new_frontier.add(bonded_atom.index)
        result.update(new_frontier)
        frontier = new_frontier
    
    return result


def get_nearby_atoms_distance(universe, atom_indices: List[int], cutoff: float = 3.5) -> Set[int]:
    """Get atoms within distance cutoff of given atoms."""
    ref_atoms = universe.atoms[atom_indices]
    all_atoms = universe.atoms
    
    # Calculate distances
    distances = distance_array(ref_atoms.positions, all_atoms.positions)
    
    # Find atoms within cutoff
    nearby = set()
    for i, row in enumerate(distances):
        for j, d in enumerate(row):
            if d < cutoff and j not in atom_indices:
                nearby.add(j)
    
    return nearby


def find_hbond_partners(
    universe, 
    qm_atom_indices: List[int],
    cutoff: float = 3.5
) -> List[Dict]:
    """Find residues H-bonding to QM region."""
    hbond_partners = []
    
    qm_atoms = universe.atoms[qm_atom_indices]
    
    # Find potential H-bond donors/acceptors in QM region
    qm_polar = qm_atoms.select_atoms('name N* O* S*')
    
    # Find nearby polar atoms not in QM region
    for qm_atom in qm_polar:
        nearby = universe.select_atoms(
            f'(name N* O*) and (not index {" ".join(str(i) for i in qm_atom_indices)}) '
            f'and around {cutoff} index {qm_atom.index}'
        )
        
        for partner in nearby:
            # Record the residue
            hbond_partners.append({
                'resid': partner.resid,
                'resname': partner.resname,
                'atom_name': partner.name,
                'atom_index': partner.index,
                'distance': np.linalg.norm(qm_atom.position - partner.position),
            })
    
    # Deduplicate by residue
    seen_resids = set()
    unique_partners = []
    for p in sorted(hbond_partners, key=lambda x: x['distance']):
        if p['resid'] not in seen_resids:
            seen_resids.add(p['resid'])
            unique_partners.append(p)
    
    return unique_partners


def define_qm_region_mda(
    gro_file: Path,
    system_info: Dict,
    config: dict,
    qm_size: str = 'extended'
) -> Dict:
    """Define QM region using MDAnalysis."""
    if not HAS_MDA:
        return define_qm_region_simple(gro_file, system_info, config)
    
    u = mda.Universe(str(gro_file))
    
    cys_resid = system_info['covalent_atoms']['cys_resid']
    warhead = system_info['warhead']
    
    qm_indices = []
    qm_info = {
        'sections': {},
        'link_atoms': [],
    }
    
    # 1. Cysteine backbone
    backbone_sel = f'resid {cys_resid} and name ' + ' '.join(CYSTEINE_ATOMS['backbone'])
    backbone = u.select_atoms(backbone_sel)
    backbone_indices = [a.index for a in backbone]
    qm_indices.extend(backbone_indices)
    qm_info['sections']['cysteine_backbone'] = backbone_indices
    
    # 2. Cysteine sidechain  
    sidechain_names = CYSTEINE_ATOMS['sidechain'] + ['HB1']  # Include alternate naming
    sidechain_sel = f'resid {cys_resid} and name ' + ' '.join(sidechain_names)
    sidechain = u.select_atoms(sidechain_sel)
    sidechain_indices = [a.index for a in sidechain]
    qm_indices.extend(sidechain_indices)
    qm_info['sections']['cysteine_sidechain'] = sidechain_indices
    
    # 3. Warhead reactive atoms
    warhead_atom_names = WARHEAD_ATOMS_MINIMAL.get(warhead, [])
    if warhead_atom_names:
        warhead_sel = f'resid {cys_resid} and name ' + ' '.join(warhead_atom_names)
        warhead_atoms = u.select_atoms(warhead_sel)
        warhead_indices = [a.index for a in warhead_atoms]
    else:
        # Fallback: include all non-standard atoms in the CYL residue
        c1_idx = system_info['covalent_atoms']['c1_index_0based']
        warhead_indices = list(get_bonded_atoms_mda(u, [c1_idx], n_bonds=4))
        warhead_indices = [i for i in warhead_indices if i not in qm_indices]
    
    qm_indices.extend(warhead_indices)
    qm_info['sections']['warhead_core'] = warhead_indices
    
    # 4. Extended region (optional)
    if qm_size == 'extended':
        qm_cfg = config.get('qm_region', {}).get('extended', {})
        
        # First shell of ligand (atoms within 2 bonds of warhead core)
        if qm_cfg.get('include_first_shell', True):
            c1_idx = system_info['covalent_atoms']['c1_index_0based']
            first_shell = get_bonded_atoms_mda(u, warhead_indices, n_bonds=2)
            first_shell = [i for i in first_shell if i not in qm_indices]
            qm_indices.extend(first_shell)
            qm_info['sections']['ligand_first_shell'] = list(first_shell)
        
        # H-bond partners
        if qm_cfg.get('include_hbond_partners', True):
            hbond_cutoff = qm_cfg.get('hbond_cutoff_A', 3.5)
            hbond_partners = find_hbond_partners(u, qm_indices, cutoff=hbond_cutoff)
            
            # Add sidechain atoms of H-bonding residues (but not full residues)
            hbond_indices = []
            for partner in hbond_partners[:3]:  # Limit to 3 closest
                partner_atoms = u.select_atoms(
                    f'resid {partner["resid"]} and (name N* O* H* or name C[BGDEZ]*)'
                )
                hbond_indices.extend([a.index for a in partner_atoms])
            
            hbond_indices = [i for i in hbond_indices if i not in qm_indices]
            qm_indices.extend(hbond_indices)
            qm_info['sections']['hbond_partners'] = hbond_indices
            qm_info['hbond_residues'] = [p['resid'] for p in hbond_partners[:3]]
    
    # Remove duplicates and sort
    qm_indices = sorted(set(qm_indices))
    
    # 5. Identify QM/MM boundary atoms (for link atoms)
    link_atoms = identify_link_atoms(u, qm_indices)
    qm_info['link_atoms'] = link_atoms
    
    # 6. Calculate total charge of QM region
    qm_charge = sum(u.atoms[i].charge for i in qm_indices if hasattr(u.atoms[i], 'charge'))
    qm_charge_rounded = round(qm_charge)
    
    return {
        'qm_indices_0based': qm_indices,
        'qm_indices_1based': [i + 1 for i in qm_indices],
        'n_qm_atoms': len(qm_indices),
        'qm_charge': qm_charge_rounded,
        'qm_charge_exact': qm_charge,
        'info': qm_info,
        'qm_size': qm_size,
    }


def identify_link_atoms(universe, qm_indices: List[int]) -> List[Dict]:
    """Identify QM/MM boundary bonds where link atoms are needed."""
    link_atoms = []
    qm_set = set(qm_indices)
    
    # For each QM atom, check if it's bonded to MM atoms
    for qm_idx in qm_indices:
        qm_atom = universe.atoms[qm_idx]
        
        if not hasattr(qm_atom, 'bonds'):
            continue
        
        for bond in qm_atom.bonds:
            for bonded_atom in bond.atoms:
                if bonded_atom.index not in qm_set:
                    # This is a QM/MM boundary
                    # Check if it's a C-C bond (typical for link atoms)
                    if 'C' in qm_atom.name and 'C' in bonded_atom.name:
                        link_atoms.append({
                            'qm_atom_index': qm_idx,
                            'qm_atom_name': qm_atom.name,
                            'mm_atom_index': bonded_atom.index,
                            'mm_atom_name': bonded_atom.name,
                            'bond_type': 'C-C',
                        })
    
    return link_atoms


def define_qm_region_simple(gro_file: Path, system_info: Dict, config: dict) -> Dict:
    """Simple QM region definition without MDAnalysis."""
    # Use the CYL residue atoms from system_info
    cyl_indices = system_info['covalent_atoms']['cyl_atom_indices_0based']
    
    return {
        'qm_indices_0based': cyl_indices,
        'qm_indices_1based': [i + 1 for i in cyl_indices],
        'n_qm_atoms': len(cyl_indices),
        'qm_charge': 0,  # Assume neutral
        'info': {'sections': {'cyl_residue': cyl_indices}},
        'qm_size': 'minimal',
    }


def process_system(system_dir: Path, config: dict, qm_size: str) -> bool:
    """Define QM region for a single system."""
    
    # Load system info
    info_file = system_dir / 'system_info.json'
    if not info_file.exists():
        print(f"    ERROR: system_info.json not found")
        return False
    
    with open(info_file) as f:
        system_info = json.load(f)
    
    # Get snapshot
    gro_file = system_dir / 'snapshot.gro'
    if not gro_file.exists():
        print(f"    ERROR: snapshot.gro not found")
        return False
    
    print(f"  Processing: {system_info['receptor']}/{system_info['inhibitor']}/{system_info['warhead']}")
    
    # Define QM region
    try:
        qm_region = define_qm_region_mda(gro_file, system_info, config, qm_size)
    except Exception as e:
        print(f"    ERROR: {e}")
        return False
    
    # Save QM region definition
    with open(system_dir / 'qm_region.json', 'w') as f:
        json.dump(qm_region, f, indent=2)
    
    print(f"    ✓ QM region: {qm_region['n_qm_atoms']} atoms")
    print(f"    ✓ QM charge: {qm_region['qm_charge']}")
    print(f"    ✓ Link atoms: {len(qm_region['info'].get('link_atoms', []))}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Define QM region for QM/MM")
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'config.yaml')
    parser.add_argument('--qm-size', choices=['minimal', 'extended'], default='extended',
                       help='QM region size')
    parser.add_argument('--receptors', nargs='+', default=None)
    parser.add_argument('--inhibitors', nargs='+', default=None)
    parser.add_argument('--warheads', nargs='+', default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    base_dir = args.config.parent
    
    output_dir = Path(config['paths']['output_dir'])
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()
    
    print("=" * 60)
    print("Step 2: Define QM Region")
    print("=" * 60)
    print(f"QM size: {args.qm_size}")
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
                if system_dir.exists():
                    total_count += 1
                    if process_system(system_dir, config, args.qm_size):
                        success_count += 1
    
    print()
    print(f"Defined QM region for {success_count}/{total_count} systems")
    
    return success_count == total_count


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
