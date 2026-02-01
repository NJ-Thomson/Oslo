#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
b05_create_cyl_residue.py

Create a proper CYL (Cysteine-Ligand) residue for GROMACS MD and RBFE.

This script creates a custom amino acid residue that includes:
- Proper peptide backbone (N, H, CA, HA, C, O)
- Cysteine side chain (CB, HB, SG)
- Covalently attached ligand with GAFF2 atom types

Charge handling (important for FEP):
- By default, uses AM1-BCC charges from antechamber for the ENTIRE residue
  (backbone + ligand) to maintain consistent charge derivation
- Use --use-amber-backbone-charges to replace backbone charges with AMBER
  (not recommended for FEP as it mixes charge models)

For RBFE transformations (CYL1 -> CYL2):
- The backbone atoms are common between states
- Only ligand atoms transform
- Use pmx or similar tools for the transformation setup

Usage:
    python b05_create_cyl_residue.py \\
        --adduct-mol2 params/adduct_gaff2.mol2 \\
        --residue-name CYL \\
        --output-dir Outputs/Covalent/md_prep

    # This will automatically:
    # 1. Find GROMACS installation via 'which gmx'
    # 2. Copy amber99sb-ildn.ff to output-dir/amber99sb-ildn-cyl.ff
    # 3. Add CYL residue to aminoacids.rtp
    # 4. Add GAFF2 atomtypes to ffnonbonded.itp

Outputs:
    - amber99sb-ildn-{resname}.ff/ - Modified force field with custom residue
    - {resname}_residue/           - Standalone residue files for reference
"""

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def find_gromacs_data_dir() -> Optional[Path]:
    """
    Find the GROMACS data directory containing force fields.

    Searches in order:
    1. GMXDATA environment variable
    2. 'which gmx' -> derive from executable path
    3. Common conda/system paths

    Returns:
        Path to GROMACS top directory (containing *.ff folders), or None
    """
    # Try GMXDATA environment variable
    gmxdata = os.environ.get('GMXDATA')
    if gmxdata:
        top_dir = Path(gmxdata) / 'top'
        if top_dir.exists():
            return top_dir

    # Try to find gmx executable
    try:
        result = subprocess.run(['which', 'gmx'], capture_output=True, text=True, check=True)
        gmx_path = Path(result.stdout.strip())

        # GROMACS can be in:
        # - conda: .../envs/name/bin.AVX2_256/gmx -> .../envs/name/share/gromacs/top
        # - conda: .../envs/name/bin/gmx -> .../envs/name/share/gromacs/top
        # - system: /usr/bin/gmx -> /usr/share/gromacs/top

        # Get prefix by going up from bin directory
        # Handle conda bin.AVX2_256 style paths
        bin_dir = gmx_path.parent
        if bin_dir.name.startswith('bin'):
            prefix = bin_dir.parent
        else:
            prefix = bin_dir.parent.parent

        top_dir = prefix / 'share' / 'gromacs' / 'top'
        if top_dir.exists():
            return top_dir

    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try common paths
    common_paths = [
        Path('/usr/share/gromacs/top'),
        Path('/usr/local/gromacs/share/gromacs/top'),
        Path.home() / 'miniconda3' / 'envs' / 'benchmark' / 'share' / 'gromacs' / 'top',
    ]

    for path in common_paths:
        if path.exists():
            return path

    return None


def copy_forcefield(gmx_top_dir: Path, ff_name: str, output_dir: Path, new_ff_name: str = None) -> Path:
    """
    Copy a GROMACS force field directory to a new location.

    Args:
        gmx_top_dir: GROMACS top directory containing force fields
        ff_name: Name of force field (e.g., 'amber99sb-ildn')
        output_dir: Directory to copy force field to
        new_ff_name: New name for the force field (default: same as ff_name)

    Returns:
        Path to the copied force field directory
    """
    src_ff = gmx_top_dir / f"{ff_name}.ff"
    if not src_ff.exists():
        raise FileNotFoundError(f"Force field not found: {src_ff}")

    if new_ff_name is None:
        new_ff_name = ff_name

    dst_ff = output_dir / f"{new_ff_name}.ff"

    if dst_ff.exists():
        print(f"  Force field already exists: {dst_ff}")
        print(f"  Removing and re-copying...")
        shutil.rmtree(dst_ff)

    shutil.copytree(src_ff, dst_ff)

    # Also copy residuetypes.dat to the output directory (GMXLIB path)
    # This is needed for pdb2gmx to recognize custom residue types
    src_restypes = gmx_top_dir / 'residuetypes.dat'
    dst_restypes = output_dir / 'residuetypes.dat'
    if src_restypes.exists() and not dst_restypes.exists():
        shutil.copy(src_restypes, dst_restypes)
        print(f"  Copied residuetypes.dat to {output_dir}")

    return dst_ff


def add_residue_to_residuetypes(output_dir: Path, residue_name: str, residue_type: str = "Protein"):
    """
    Add a custom residue to residuetypes.dat so pdb2gmx recognizes it.

    Args:
        output_dir: Directory containing residuetypes.dat (GMXLIB path)
        residue_name: Name of the residue (e.g., CYL, CX2)
        residue_type: Type of residue (default: Protein)
    """
    restypes_file = output_dir / 'residuetypes.dat'
    if not restypes_file.exists():
        print(f"  WARNING: residuetypes.dat not found in {output_dir}")
        return

    with open(restypes_file) as f:
        content = f.read()

    # Check if already present
    if residue_name in content:
        print(f"  {residue_name} already in residuetypes.dat")
        return

    # Add the residue
    with open(restypes_file, 'a') as f:
        f.write(f"{residue_name}\t{residue_type}\n")
    print(f"  Added {residue_name} as {residue_type} to residuetypes.dat")


def add_hdb_entry_to_forcefield(ff_dir: Path, residue_name: str) -> bool:
    """
    Add an HDB entry for the custom residue to tell pdb2gmx how to build
    missing backbone hydrogen H.

    The HDB format is:
        RESIDUE_NAME  NUM_ADD_GROUPS
        num_H  type  name  atom  [bonded atoms for geometry...]

    For CYL residues, we only define H since ligand hydrogens are in the PDB.

    Args:
        ff_dir: Path to force field directory
        residue_name: Name of the custom residue (e.g., CYL, CC2)

    Returns:
        True if added successfully, False if already exists
    """
    hdb_file = ff_dir / 'aminoacids.hdb'
    if not hdb_file.exists():
        print(f"  WARNING: aminoacids.hdb not found in {ff_dir}")
        return False

    with open(hdb_file) as f:
        content = f.read()

    # Check if residue already exists
    if f'\n{residue_name}\t' in content or content.startswith(f'{residue_name}\t'):
        print(f"  Residue {residue_name} already exists in {hdb_file.name}")
        return False

    # Add HDB entry for the custom residue
    # Only define backbone H - ligand hydrogens are in the input PDB
    # Format: num_H type H_name bonded_atom geometry_atoms...
    # Type 1 = single H with angle
    hdb_entry = f"""
{residue_name}\t1
1\t1\tH\tN\t-C\tCA
"""

    # Append the new residue
    with open(hdb_file, 'a') as f:
        f.write(hdb_entry)

    return True


def add_rtp_entry_to_forcefield(ff_dir: Path, rtp_content: str, residue_name: str) -> bool:
    """
    Add an RTP entry to the force field's aminoacids.rtp file.

    Args:
        ff_dir: Path to force field directory
        rtp_content: RTP entry content to add
        residue_name: Residue name (to check for existing entry)

    Returns:
        True if added successfully, False if already exists
    """
    rtp_file = ff_dir / 'aminoacids.rtp'
    if not rtp_file.exists():
        raise FileNotFoundError(f"aminoacids.rtp not found in {ff_dir}")

    with open(rtp_file) as f:
        content = f.read()

    # Check if residue already exists
    if f'[ {residue_name} ]' in content:
        print(f"  Residue {residue_name} already exists in {rtp_file}")
        return False

    # Append the new residue
    with open(rtp_file, 'a') as f:
        f.write('\n\n')
        f.write(rtp_content)
        f.write('\n')

    return True


def add_gaff2_atomtypes_to_forcefield(ff_dir: Path, gaff2_types: List[dict]) -> Path:
    """
    Add GAFF2 atomtypes to the force field.

    Updates:
    1. atomtypes.atp - atom type names and masses (required by pdb2gmx)
    2. gaff2_atomtypes.itp - full LJ parameters
    3. ffnonbonded.itp - include directive for gaff2_atomtypes.itp

    Args:
        ff_dir: Path to force field directory
        gaff2_types: List of GAFF2 atomtype dicts

    Returns:
        Path to the created atomtypes file
    """
    # 1. Add to atomtypes.atp (required for pdb2gmx to recognize atom types)
    atp_file = ff_dir / 'atomtypes.atp'
    if atp_file.exists():
        with open(atp_file) as f:
            content = f.read()

        # Check which types need to be added
        existing_types = set()
        for line in content.split('\n'):
            if line.strip() and not line.startswith(';'):
                parts = line.split()
                if parts:
                    existing_types.add(parts[0])

        # Append new types
        new_types = []
        for at in gaff2_types:
            if at['name'] not in existing_types:
                comment = get_gaff2_description(at['name'])
                new_types.append(f"{at['name']:<6s}  {at['mass']:>10.5f}\t; {comment}")

        if new_types:
            with open(atp_file, 'a') as f:
                f.write("; GAFF2 atomtypes for covalent ligands\n")
                for line in new_types:
                    f.write(line + '\n')
            print(f"  Added {len(new_types)} GAFF2 types to atomtypes.atp")

    # 2. Create GAFF2 atomtypes file with full LJ parameters
    gaff2_file = ff_dir / 'gaff2_atomtypes.itp'

    with open(gaff2_file, 'w') as f:
        f.write("; GAFF2 atomtypes for covalent ligand residues (CYL)\n")
        f.write("; Generated by b05_create_cyl_residue.py\n")
        f.write(";\n")
        f.write("[ atomtypes ]\n")
        f.write("; name    at.num    mass      charge    ptype    sigma        epsilon\n")

        for at in gaff2_types:
            # Determine atomic number from type
            at_num = get_atomic_number(at['name'])
            f.write(f"  {at['name']:<6s}  {at_num:>3d}  {at['mass']:>10.4f}  {at['charge']:>8.4f}  "
                   f"{at['ptype']}  {at['sigma']:.6e}  {at['epsilon']:.6e}\n")

    # 3. Check if already included in ffnonbonded.itp
    ffnb_file = ff_dir / 'ffnonbonded.itp'
    if ffnb_file.exists():
        with open(ffnb_file) as f:
            content = f.read()

        include_line = '#include "gaff2_atomtypes.itp"'
        if include_line not in content:
            # Add include at the beginning (after any initial comments)
            lines = content.split('\n')
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith(';'):
                    insert_idx = i
                    break

            lines.insert(insert_idx, f'\n; GAFF2 atomtypes for covalent ligands\n{include_line}\n')

            with open(ffnb_file, 'w') as f:
                f.write('\n'.join(lines))

    return gaff2_file


def get_gaff2_description(atype: str) -> str:
    """Get description for GAFF2 atom type."""
    descriptions = {
        'c': 'sp2 C carbonyl',
        'c1': 'sp C',
        'c2': 'sp2 C alkene',
        'c3': 'sp3 C',
        'c6': 'sp3 C in 6-membered ring',
        'ca': 'sp2 C aromatic',
        'cc': 'sp2 C in conj. ring (C-C)',
        'cd': 'sp2 C in conj. ring (C-C)',
        'ce': 'sp2 C in conj. chain (inner)',
        'cf': 'sp2 C in conj. chain (inner)',
        'n': 'sp2 N amide',
        'n1': 'sp N',
        'n2': 'sp2 N with 2 subs',
        'n3': 'sp3 N with 3 subs',
        'n4': 'sp3 N with 4 subs',
        'na': 'sp2 N with 3 subs',
        'nb': 'sp2 N in aromatic ring',
        'nc': 'sp2 N in conj. ring (inner)',
        'nd': 'sp2 N in conj. ring (inner)',
        'ne': 'sp2 N in conj. chain (inner)',
        'nf': 'sp2 N in conj. chain (inner)',
        'nh': 'sp2 N amine',
        'ns': 'sp2 N amide with 1 sub',
        'nu': 'sp2 N amide with 2 subs',
        'o': 'sp2 O carbonyl',
        'oh': 'sp3 O hydroxyl',
        'os': 'sp3 O ether',
        's': 'sp2 S',
        's2': 'sp2 S',
        'ss': 'sp3 S thioether',
        'sh': 'sp3 S thiol',
        'h1': 'H on C with 1 EW group',
        'h2': 'H on C with 2 EW groups',
        'h3': 'H on C with 3 EW groups',
        'h4': 'H on aromatic C with 1 EW',
        'h5': 'H on aromatic C with 2 EW',
        'ha': 'H on aromatic C',
        'hc': 'H on aliphatic C',
        'hn': 'H on N',
        'ho': 'H on O',
        'hs': 'H on S',
    }
    return descriptions.get(atype.lower(), f'GAFF2 {atype}')


def get_atomic_number(gaff_type: str) -> int:
    """Get atomic number from GAFF2 atom type."""
    type_lower = gaff_type.lower()
    if type_lower.startswith('c'):
        return 6
    elif type_lower.startswith('n'):
        return 7
    elif type_lower.startswith('o'):
        return 8
    elif type_lower.startswith('s'):
        return 16
    elif type_lower.startswith('h'):
        return 1
    elif type_lower.startswith('f'):
        return 9
    elif type_lower.startswith('cl'):
        return 17
    elif type_lower.startswith('br'):
        return 35
    elif type_lower.startswith('i'):
        return 53
    elif type_lower.startswith('p'):
        return 15
    else:
        return 6  # Default to carbon


# Standard AMBER99SB-ILDN backbone atom types and charges for amino acids
AMBER_BACKBONE = {
    # Atom: (type, charge) - charges from standard CYS/CYX
    'N':  ('N',  -0.4157),
    'H':  ('H',   0.2719),
    'CA': ('CT',  0.0213),  # Note: CYX has slightly different
    'HA': ('H1',  0.1124),
    'C':  ('C',   0.5973),
    'O':  ('O',  -0.5679),
}

# CYS sidechain (for CYX - covalent cysteine)
AMBER_CYX_SIDECHAIN = {
    'CB':  ('CT', -0.0351),
    'HB2': ('H1',  0.0508),  # HB1 in some conventions
    'HB3': ('H1',  0.0508),  # HB2 in some conventions
    'SG':  ('S',  -0.1438),  # Thioether sulfur
}

# GAFF2 to AMBER type mapping for common types
GAFF2_TO_AMBER = {
    'c3': 'CT',  # sp3 carbon
    'c':  'C',   # carbonyl carbon
    'n':  'N',   # sp2 nitrogen in amide
    'n8': 'N3',  # sp3 nitrogen (will be replaced)
    'hn': 'H',   # H on nitrogen
    'h1': 'H1',  # H on sp3 carbon with 1 EW
    'hc': 'HC',  # H on sp3 carbon
    'ss': 'S',   # sulfur in thioether (we'll use GAFF for ligand S)
}


def parse_mol2(mol2_path: Path) -> Tuple[List[dict], List[Tuple[int, int]]]:
    """
    Parse mol2 file to extract atoms and bonds.

    Returns:
        atoms: List of dicts with keys: idx, name, x, y, z, type, resname, charge
        bonds: List of (atom1_idx, atom2_idx) tuples
    """
    atoms = []
    bonds = []

    with open(mol2_path) as f:
        content = f.read()

    # Parse ATOM section
    atom_match = re.search(r'@<TRIPOS>ATOM\n(.*?)@<TRIPOS>', content, re.DOTALL)
    if atom_match:
        for line in atom_match.group(1).strip().split('\n'):
            parts = line.split()
            if len(parts) >= 9:
                atoms.append({
                    'idx': int(parts[0]),
                    'name': parts[1],
                    'x': float(parts[2]),
                    'y': float(parts[3]),
                    'z': float(parts[4]),
                    'type': parts[5],
                    'resname': parts[7],
                    'charge': float(parts[8])
                })

    # Parse BOND section
    bond_match = re.search(r'@<TRIPOS>BOND\n(.*?)(?:@<TRIPOS>|$)', content, re.DOTALL)
    if bond_match:
        for line in bond_match.group(1).strip().split('\n'):
            parts = line.split()
            if len(parts) >= 3:
                bonds.append((int(parts[1]), int(parts[2])))

    return atoms, bonds


def identify_cys_and_ligand_atoms(atoms: List[dict], bonds: List[Tuple[int, int]]) -> dict:
    """
    Identify which atoms belong to CYS backbone/caps vs ligand.

    The mol2 from antechamber has:
    - CYS backbone: N, CA, CB, SG (and their H atoms)
    - N-cap: Extra atoms on N-terminus (CH3-NH2 cap)
    - C-cap: Extra atoms on C-terminus (CH3 cap)
    - Ligand: Everything attached to SG that isn't CB

    Returns dict with atom indices for each category.
    """
    # Build adjacency list
    adj = {a['idx']: [] for a in atoms}
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)

    # Find key atoms by name
    atom_by_name = {a['name']: a['idx'] for a in atoms}
    idx_to_atom = {a['idx']: a for a in atoms}

    sg_idx = atom_by_name.get('SG')
    cb_idx = atom_by_name.get('CB')
    ca_idx = atom_by_name.get('CA')
    n_idx = atom_by_name.get('N')

    if not all([sg_idx, cb_idx, ca_idx, n_idx]):
        raise ValueError("Could not find CYS backbone atoms (N, CA, CB, SG)")

    # Identify CYS backbone atoms (standard backbone + CB, SG)
    backbone_names = {'N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'SG'}
    backbone_indices = {atom_by_name[n] for n in backbone_names if n in atom_by_name}

    # Identify cap atoms (attached to N or to a carbon attached to CA that isn't CB)
    # N-cap: atoms attached to N that aren't CA or standard H
    n_cap_indices = set()
    for neighbor in adj[n_idx]:
        atom = idx_to_atom[neighbor]
        if atom['name'] not in backbone_names:
            n_cap_indices.add(neighbor)
            # Also add any atoms attached to this cap atom
            for nn in adj[neighbor]:
                if nn not in backbone_indices and nn != n_idx:
                    n_cap_indices.add(nn)

    # C-cap: CC and its hydrogens (attached to CA, not CB)
    c_cap_indices = set()
    cc_idx = atom_by_name.get('CC')
    if cc_idx:
        c_cap_indices.add(cc_idx)
        for neighbor in adj[cc_idx]:
            atom = idx_to_atom[neighbor]
            if atom['name'].startswith('HC'):
                c_cap_indices.add(neighbor)

    # Ligand: everything attached to SG that isn't CB, and their descendants
    ligand_indices = set()

    def find_ligand_atoms(start_idx, visited):
        """BFS to find all ligand atoms starting from SG neighbor."""
        queue = [start_idx]
        while queue:
            idx = queue.pop(0)
            if idx in visited:
                continue
            visited.add(idx)
            ligand_indices.add(idx)
            for neighbor in adj[idx]:
                if neighbor not in visited and neighbor != sg_idx:
                    queue.append(neighbor)

    # Start from SG neighbors that aren't CB
    for neighbor in adj[sg_idx]:
        if neighbor != cb_idx:
            find_ligand_atoms(neighbor, set())

    return {
        'backbone': backbone_indices,
        'n_cap': n_cap_indices,
        'c_cap': c_cap_indices,
        'ligand': ligand_indices,
        'sg_idx': sg_idx,
        'cb_idx': cb_idx,
        'ca_idx': ca_idx,
        'n_idx': n_idx,
    }


def calculate_cap_charges(atoms: List[dict], categories: dict) -> dict:
    """
    Calculate total charges in each category for charge redistribution.
    """
    idx_to_atom = {a['idx']: a for a in atoms}

    charges = {}
    for cat, indices in categories.items():
        if isinstance(indices, set):
            charges[cat] = sum(idx_to_atom[i]['charge'] for i in indices)

    return charges


def create_cyl_rtp_entry(
    atoms: List[dict],
    bonds: List[Tuple[int, int]],
    categories: dict,
    residue_name: str = "CYL",
    use_amber_backbone_charges: bool = False
) -> Tuple[str, dict]:
    """
    Create RTP entry for CYL residue with proper peptide backbone.

    Strategy:
    - Remove cap atoms (ACE N-cap, NME C-cap)
    - Keep AM1-BCC charges from antechamber for consistency (default)
    - Or use AMBER charges for backbone if use_amber_backbone_charges=True
    - Apply minimal charge correction to ensure integer total

    For FEP calculations, use_amber_backbone_charges=False (default) is recommended
    to maintain consistent charge derivation throughout the residue.
    """
    idx_to_atom = {a['idx']: a for a in atoms}
    atom_by_name = {a['name']: a for a in atoms}

    # Calculate total charge from adduct (should be 0 for neutral)
    total_adduct_charge = sum(a['charge'] for a in atoms)

    # Charges being removed (caps)
    n_cap_charge = sum(idx_to_atom[i]['charge'] for i in categories['n_cap'])
    c_cap_charge = sum(idx_to_atom[i]['charge'] for i in categories['c_cap'])

    # Target charge should be integer
    target_charge = round(total_adduct_charge)

    # Build atom list for RTP
    rtp_atoms = []
    atom_renumber = {}  # old_idx -> new_idx
    new_idx = 0

    # Standard backbone atom order (including C and O)
    backbone_order = ['N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'SG', 'C', 'O']

    for name in backbone_order:
        if name not in atom_by_name:
            # C and O might not exist in capped structure - will handle below
            continue

        old_atom = atom_by_name[name]
        old_idx = old_atom['idx']

        # Skip if this atom is in a cap (shouldn't happen for backbone atoms)
        if old_idx in categories['n_cap'] or old_idx in categories['c_cap']:
            continue

        new_idx += 1
        atom_renumber[old_idx] = new_idx

        # ALWAYS use AMBER atom TYPES for backbone atoms
        # This ensures peptide bonds are compatible with adjacent residues
        # (which use AMBER types). The mol2 has GAFF2 types because it's
        # parameterized as a capped structure, not a peptide backbone.
        if name in AMBER_BACKBONE:
            atype = AMBER_BACKBONE[name][0]
        elif name in AMBER_CYX_SIDECHAIN:
            atype = AMBER_CYX_SIDECHAIN[name][0]
        else:
            atype = old_atom['type']

        # CHARGES can be either AMBER or AM1-BCC based on flag
        if use_amber_backbone_charges:
            if name in AMBER_BACKBONE:
                charge = AMBER_BACKBONE[name][1]
            elif name in AMBER_CYX_SIDECHAIN:
                charge = AMBER_CYX_SIDECHAIN[name][1]
            else:
                charge = old_atom['charge']
        else:
            # Keep AM1-BCC charges from antechamber (recommended for FEP)
            charge = old_atom['charge']

        rtp_atoms.append({
            'name': name,
            'type': atype,
            'charge': charge,
            'cgnr': new_idx
        })

    # Check if C and O were found - if not, we need to add them
    # This happens when the mol2 doesn't have proper backbone C/O naming
    backbone_names_added = {a['name'] for a in rtp_atoms}

    if 'C' not in backbone_names_added:
        # Backbone C not found - must add with AMBER charges (no AM1-BCC available)
        new_idx += 1
        if use_amber_backbone_charges:
            c_charge = AMBER_BACKBONE['C'][1]
        else:
            # For AM1-BCC mode, use AMBER charge for missing C (unavoidable)
            c_charge = AMBER_BACKBONE['C'][1]
            print(f"  WARNING: Backbone C not found in mol2, using AMBER charge ({c_charge:.4f})")
        rtp_atoms.append({
            'name': 'C',
            'type': AMBER_BACKBONE['C'][0],
            'charge': c_charge,
            'cgnr': new_idx
        })

    if 'O' not in backbone_names_added:
        new_idx += 1
        if use_amber_backbone_charges:
            o_charge = AMBER_BACKBONE['O'][1]
        else:
            o_charge = AMBER_BACKBONE['O'][1]
            print(f"  WARNING: Backbone O not found in mol2, using AMBER charge ({o_charge:.4f})")
        rtp_atoms.append({
            'name': 'O',
            'type': AMBER_BACKBONE['O'][0],
            'charge': o_charge,
            'cgnr': new_idx
        })

    # Add ligand atoms (keeping GAFF2 types and AM1-BCC charges)
    ligand_charge_sum = 0
    ligand_atoms_list = []

    # Track used names to avoid conflicts with backbone
    used_names = {a['name'] for a in rtp_atoms}

    for a in atoms:
        if a['idx'] in categories['ligand']:
            new_idx += 1
            atom_renumber[a['idx']] = new_idx
            ligand_charge_sum += a['charge']

            # Rename if conflicts with backbone
            name = a['name']
            if name in used_names:
                # Add L prefix for Ligand to distinguish
                if name == 'H':
                    name = 'HL'  # Ligand hydrogen
                elif name == 'N':
                    name = 'NL'
                elif name == 'C':
                    name = 'CL'
                elif name == 'O':
                    name = 'OL'
                else:
                    name = f"L{name}"

            used_names.add(name)
            ligand_atoms_list.append({
                'name': name,
                'orig_name': a['name'],  # Keep original for bonds
                'type': a['type'],
                'charge': a['charge'],
                'cgnr': new_idx
            })

    # Calculate total charge and correction needed
    backbone_charge = sum(a['charge'] for a in rtp_atoms)
    current_total = backbone_charge + ligand_charge_sum
    charge_deficit = target_charge - current_total

    # For FEP: distribute charge correction to BACKBONE atoms only
    # This preserves exact AM1-BCC charges on ligand (which differs between states)
    # while backbone (common between states) absorbs the correction
    # This ensures the charge correction doesn't affect ΔG calculations
    if abs(charge_deficit) > 1e-10 and rtp_atoms:
        charge_correction = charge_deficit / len(rtp_atoms)
        for a in rtp_atoms:
            a['charge'] += charge_correction
        print(f"  Charge deficit from cap removal: {charge_deficit:.6f}")
        print(f"  Distributed {charge_correction:.6f} to each of {len(rtp_atoms)} backbone atoms")
        # Recalculate backbone charge after correction
        backbone_charge = sum(a['charge'] for a in rtp_atoms)

    # Final adjustment: ensure total is EXACTLY the target integer AFTER rounding to write precision
    # When charges are written to RTP with limited decimal places (5), rounding errors accumulate.
    # We need to calculate what each charge will be AFTER rounding and adjust one atom's charge
    # so that the rounded sum equals exactly the target integer.
    # For FEP: apply this adjustment to a BACKBONE atom to preserve exact ligand charges
    WRITE_PRECISION = 5  # Number of decimal places used when writing charges to RTP

    all_atoms = rtp_atoms + ligand_atoms_list
    if all_atoms:
        # Calculate rounded sum
        rounded_total = sum(round(a['charge'], WRITE_PRECISION) for a in all_atoms)
        residual = target_charge - rounded_total

        if abs(residual) > 1e-10:
            # Apply residual to a BACKBONE atom (preserves exact ligand charges for FEP)
            # Choose the backbone atom with largest absolute charge (CA typically)
            backbone_atom = max(rtp_atoms, key=lambda a: abs(a['charge']))
            current_rounded = round(backbone_atom['charge'], WRITE_PRECISION)
            target_rounded = current_rounded + residual
            backbone_atom['charge'] = target_rounded

            print(f"  Rounding correction: adjusted {backbone_atom['name']} by {residual:.6e} for exact integer")

    # Build name mapping from original to renamed
    name_map = {}
    for a in ligand_atoms_list:
        name_map[a.get('orig_name', a['name'])] = a['name']

    rtp_atoms.extend(ligand_atoms_list)

    # Build bonds section
    # Standard backbone bonds
    rtp_bonds = [
        ('-C', 'N'),   # Peptide bond from previous residue
        ('N', 'H'),
        ('N', 'CA'),
        ('CA', 'HA'),
        ('CA', 'CB'),
        ('CA', 'C'),
        ('CB', 'HB2'),
        ('CB', 'HB3'),
        ('CB', 'SG'),
        ('C', 'O'),
        ('C', '+N'),   # Peptide bond to next residue
    ]

    # Add ligand bonds
    for i, j in bonds:
        # Skip bonds involving caps
        if i in categories['n_cap'] or j in categories['n_cap']:
            continue
        if i in categories['c_cap'] or j in categories['c_cap']:
            continue

        # Skip backbone-backbone bonds (already defined)
        i_is_backbone = i in categories['backbone']
        j_is_backbone = j in categories['backbone']
        if i_is_backbone and j_is_backbone:
            continue

        # Get atom names (use renamed names for ligand atoms)
        name_i = idx_to_atom[i]['name']
        name_j = idx_to_atom[j]['name']

        # Apply renaming for ligand atoms
        if i in categories['ligand']:
            name_i = name_map.get(name_i, name_i)
        if j in categories['ligand']:
            name_j = name_map.get(name_j, name_j)

        rtp_bonds.append((name_i, name_j))

    # Generate RTP text
    lines = []
    lines.append(f"[ {residue_name} ]")
    lines.append(f" ; Cysteine covalently attached to ligand")
    lines.append(f" ; Total charge: {target_charge} (redistributed from {total_adduct_charge:.4f})")
    lines.append(f" ; Generated by b05_create_cyl_residue.py")
    lines.append(" [ atoms ]")

    for a in rtp_atoms:
        lines.append(f"  {a['name']:<6s} {a['type']:<6s} {a['charge']:>10.5f}  {a['cgnr']}")

    lines.append(" [ bonds ]")
    for b in rtp_bonds:
        lines.append(f"  {b[0]:<6s} {b[1]:<6s}")

    # Add impropers for planarity (standard for backbone)
    lines.append(" [ impropers ]")
    lines.append("  -C    CA    N     H")  # Peptide bond planarity
    lines.append("  CA    +N    C     O")  # Carbonyl planarity

    return '\n'.join(lines), name_map


def extract_gaff2_atomtypes(
    atoms: List[dict],
    categories: dict,
    acpype_atomtypes: Optional[Dict[str, dict]] = None
) -> List[dict]:
    """
    Extract GAFF2 atom types used by ligand atoms.

    Args:
        atoms: List of atom dictionaries from mol2
        categories: Dict with 'ligand' atom indices
        acpype_atomtypes: Optional dict from parse_acpype_atomtypes() with
                          accurate LJ parameters from acpype output

    If acpype_atomtypes is provided, uses those values for mass/sigma/epsilon.
    Otherwise falls back to hardcoded dictionaries (may be incomplete).
    """
    idx_to_atom = {a['idx']: a for a in atoms}

    types_seen = set()
    atomtypes = []

    for idx in categories['ligand']:
        atom = idx_to_atom[idx]
        atype = atom['type']
        if atype not in types_seen:
            types_seen.add(atype)

            # Use acpype-parsed values if available, else hardcoded fallback
            if acpype_atomtypes and atype in acpype_atomtypes:
                params = acpype_atomtypes[atype]
                atomtypes.append({
                    'name': atype,
                    'mass': params['mass'],
                    'charge': 0.0,
                    'ptype': 'A',
                    'sigma': params['sigma'],
                    'epsilon': params['epsilon']
                })
            else:
                # Fallback to hardcoded values (may be incomplete)
                atomtypes.append({
                    'name': atype,
                    'mass': get_gaff2_mass(atype),
                    'charge': 0.0,
                    'ptype': 'A',
                    'sigma': get_gaff2_sigma(atype),
                    'epsilon': get_gaff2_epsilon(atype)
                })
                if acpype_atomtypes:
                    print(f"  WARNING: atomtype '{atype}' not in acpype output, using hardcoded fallback")

    return atomtypes


def get_gaff2_mass(atype: str) -> float:
    """Get mass for GAFF2 atom type."""
    masses = {
        'c': 12.01, 'c1': 12.01, 'c2': 12.01, 'c3': 12.01, 'c6': 12.01,
        'ca': 12.01, 'cc': 12.01, 'cd': 12.01, 'ce': 12.01, 'cf': 12.01,
        'n': 14.01, 'n1': 14.01, 'n2': 14.01, 'n3': 14.01, 'n4': 14.01,
        'na': 14.01, 'nb': 14.01, 'nc': 14.01, 'nd': 14.01, 'ne': 14.01,
        'nf': 14.01, 'nh': 14.01, 'ns': 14.01, 'nu': 14.01, 'n8': 14.01,
        'o': 16.00, 'oh': 16.00, 'os': 16.00,
        's': 32.06, 's2': 32.06, 'ss': 32.06, 'sh': 32.06,
        'h1': 1.008, 'h2': 1.008, 'h3': 1.008, 'h4': 1.008, 'h5': 1.008,
        'ha': 1.008, 'hc': 1.008, 'hn': 1.008, 'ho': 1.008, 'hs': 1.008,
    }
    return masses.get(atype.lower(), 12.01)


def get_gaff2_sigma(atype: str) -> float:
    """Get sigma (nm) for GAFF2 atom type."""
    # Approximate values - should be refined from actual GAFF2 params
    sigmas = {
        'c': 0.33152, 'c2': 0.33152, 'c3': 0.33977, 'c6': 0.33977,
        'ca': 0.33152, 'cc': 0.33152, 'cd': 0.33152, 'ce': 0.33152,
        'n': 0.32500, 'na': 0.32058, 'nb': 0.32500, 'nc': 0.32500,
        'nd': 0.32500, 'ne': 0.32500, 'ns': 0.32500, 'nu': 0.32790,
        'o': 0.29599, 'oh': 0.30610, 'os': 0.30000,
        's': 0.35636, 'ss': 0.35324,
        'h1': 0.24200, 'h2': 0.22930, 'h4': 0.25363, 'h5': 0.24214,
        'ha': 0.25996, 'hc': 0.26495, 'hn': 0.10690,
    }
    return sigmas.get(atype.lower(), 0.33)


def get_gaff2_epsilon(atype: str) -> float:
    """Get epsilon (kJ/mol) for GAFF2 atom type."""
    epsilons = {
        'c': 0.35982, 'c2': 0.35982, 'c3': 0.45773, 'c6': 0.45773,
        'ca': 0.35982, 'cc': 0.35982, 'cd': 0.35982, 'ce': 0.35982,
        'n': 0.71128, 'na': 0.85439, 'nb': 0.71128, 'nc': 0.71128,
        'nd': 0.71128, 'ne': 0.71128, 'ns': 0.71128, 'nu': 0.71128,
        'o': 0.87864, 'oh': 0.88031, 'os': 0.71176,
        's': 1.04600, 'ss': 1.04600,
        'h1': 0.08703, 'h2': 0.08284, 'h4': 0.06276, 'h5': 0.05021,
        'ha': 0.06276, 'hc': 0.06569, 'hn': 0.06569,
    }
    return epsilons.get(atype.lower(), 0.35)


def parse_acpype_atomtypes(itp_file: Path) -> Dict[str, dict]:
    """
    Parse atomtypes from acpype GMX.itp file.

    Returns dict of {typename: {'mass': float, 'sigma': float, 'epsilon': float}}

    This provides accurate LJ parameters for ALL GAFF2 atom types used,
    rather than relying on incomplete hardcoded dictionaries.
    """
    atomtypes = {}

    with open(itp_file) as f:
        content = f.read()

    # Find [ atomtypes ] section
    at_match = re.search(r'\[ atomtypes \]\s*\n(.*?)(?:\n\[|\Z)', content, re.DOTALL)
    if at_match:
        for line in at_match.group(1).strip().split('\n'):
            if line.strip() and not line.startswith(';'):
                parts = line.split()
                # acpype format: name bond_type mass charge ptype sigma epsilon ; comment
                if len(parts) >= 7:
                    name = parts[0]
                    try:
                        mass = float(parts[2])
                        sigma = float(parts[5])
                        epsilon = float(parts[6])

                        # acpype often sets mass=0, infer from atom type name
                        if mass < 0.5:
                            mass = infer_mass_from_typename(name)

                        atomtypes[name] = {
                            'mass': mass,
                            'sigma': sigma,
                            'epsilon': epsilon
                        }
                    except (ValueError, IndexError):
                        continue

    return atomtypes


def infer_mass_from_typename(typename: str) -> float:
    """Infer atomic mass from GAFF2 atom type name (first letter = element)."""
    gaff2_masses = {
        'h': 1.008,   # hydrogen
        'c': 12.01,   # carbon
        'n': 14.01,   # nitrogen
        'o': 16.00,   # oxygen
        'f': 19.00,   # fluorine
        's': 32.06,   # sulfur
        'p': 30.97,   # phosphorus
    }
    name_lower = typename.lower()
    # Handle two-letter elements
    if name_lower.startswith('cl'):
        return 35.45
    elif name_lower.startswith('br'):
        return 79.90
    # Single letter
    first_char = name_lower[0] if name_lower else ''
    return gaff2_masses.get(first_char, 12.01)


def create_cyl_pdb(
    atoms: List[dict],
    categories: dict,
    name_map: dict,
    residue_name: str = "CYL",
    chain: str = "A",
    resid: int = 1
) -> str:
    """
    Create PDB file for CYL residue (backbone + ligand, no caps).

    Args:
        name_map: Dict mapping original ligand atom names to renamed versions
    """
    idx_to_atom = {a['idx']: a for a in atoms}
    lines = []

    # Output backbone atoms first
    backbone_order = ['N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'SG']
    atom_num = 0

    for name in backbone_order:
        for a in atoms:
            if a['name'] == name and a['idx'] in categories['backbone']:
                atom_num += 1
                x, y, z = a['x'], a['y'], a['z']
                lines.append(
                    f"ATOM  {atom_num:5d} {name:4s} {residue_name:3s} {chain}{resid:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                )
                break

    # C and O from backbone need to be generated (they don't exist in capped structure)
    # For now, place them at approximate positions relative to CA
    ca_atom = next(a for a in atoms if a['name'] == 'CA')
    # Approximate C position (1.52 Å from CA in typical direction)
    c_x = ca_atom['x'] + 1.0
    c_y = ca_atom['y'] + 1.2
    c_z = ca_atom['z'] - 0.5
    atom_num += 1
    lines.append(
        f"ATOM  {atom_num:5d} {'C':4s} {residue_name:3s} {chain}{resid:4d}    "
        f"{c_x:8.3f}{c_y:8.3f}{c_z:8.3f}  1.00  0.00"
    )

    # O position (1.23 Å from C)
    o_x = c_x + 0.6
    o_y = c_y + 1.0
    o_z = c_z
    atom_num += 1
    lines.append(
        f"ATOM  {atom_num:5d} {'O':4s} {residue_name:3s} {chain}{resid:4d}    "
        f"{o_x:8.3f}{o_y:8.3f}{o_z:8.3f}  1.00  0.00"
    )

    # Output ligand atoms with renamed names
    for a in atoms:
        if a['idx'] in categories['ligand']:
            atom_num += 1
            x, y, z = a['x'], a['y'], a['z']
            # Use renamed name if available
            name = name_map.get(a['name'], a['name'])
            lines.append(
                f"HETATM{atom_num:5d} {name:4s} {residue_name:3s} {chain}{resid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
            )

    lines.append("END")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(
        description='Create CYL residue for GROMACS covalent ligand MD and RBFE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python b05_create_cyl_residue.py \\
        --adduct-mol2 Outputs/Covalent/params/adduct_gaff2.mol2 \\
        --residue-name CYL \\
        --output-dir Outputs/Covalent/md_prep

This will:
    1. Find GROMACS installation automatically (via 'which gmx')
    2. Copy amber99sb-ildn.ff to output-dir/amber99sb-ildn-cyl.ff
    3. Add CYL residue to aminoacids.rtp
    4. Add GAFF2 atomtypes to the force field

For RBFE (CYL1 -> CYL2):
    1. Run this script for each ligand with --residue-name CYL1, CYL2
    2. Use pmx to setup the transformation
    3. The backbone atoms are conserved, only ligand atoms transform
"""
    )

    ap.add_argument('--adduct-mol2', required=True,
                    help='Parameterized adduct mol2 from antechamber/acpype')
    ap.add_argument('--residue-name', default='CYL',
                    help='Residue name for the CYL residue (default: CYL)')
    ap.add_argument('--output-dir', default='md_prep',
                    help='Output directory (force field will be copied here)')
    ap.add_argument('--base-ff', default='amber99sb-ildn',
                    help='Base force field to copy from GROMACS (default: amber99sb-ildn)')
    ap.add_argument('--gmx-top-dir', default=None,
                    help='GROMACS top directory (auto-detected if not specified)')
    ap.add_argument('--skip-ff-copy', action='store_true',
                    help='Skip force field copy (use existing ff in output-dir)')
    ap.add_argument('--use-amber-backbone-charges', action='store_true',
                    help='Use AMBER charges for backbone instead of AM1-BCC (not recommended for FEP)')
    ap.add_argument('--acpype-itp', default=None,
                    help='Path to acpype GMX.itp file to get accurate atomtype LJ parameters')

    args = ap.parse_args()

    mol2_path = Path(args.adduct_mol2)
    if not mol2_path.exists():
        print(f"ERROR: mol2 file not found: {mol2_path}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Creating CYL Residue for Covalent Ligand")
    print("=" * 60)
    print(f"Input mol2:     {mol2_path}")
    print(f"Residue name:   {args.residue_name}")
    print(f"Output dir:     {output_dir}")
    print()

    # Find GROMACS and copy force field
    print("Step 0: Setting up force field...")
    if args.gmx_top_dir:
        gmx_top_dir = Path(args.gmx_top_dir)
    else:
        gmx_top_dir = find_gromacs_data_dir()

    if gmx_top_dir is None:
        print("ERROR: Could not find GROMACS data directory.")
        print("Please specify --gmx-top-dir or ensure 'gmx' is in PATH.")
        return 1

    print(f"  GROMACS top dir: {gmx_top_dir}")

    # Copy force field (use lowercase residue name in ff directory name)
    ff_name = f"{args.base_ff}-{args.residue_name.lower()}"
    ff_dir = output_dir / f"{ff_name}.ff"

    if not args.skip_ff_copy:
        print(f"  Copying {args.base_ff}.ff -> {ff_dir}")
        ff_dir = copy_forcefield(gmx_top_dir, args.base_ff, output_dir, ff_name)
        print(f"  Force field copied to: {ff_dir}")
    else:
        if not ff_dir.exists():
            print(f"ERROR: Force field not found: {ff_dir}")
            print("Run without --skip-ff-copy to create it.")
            return 1
        print(f"  Using existing force field: {ff_dir}")

    # Create output subdirectory for standalone files
    res_dir = output_dir / f"{args.residue_name.lower()}_residue"
    res_dir.mkdir(parents=True, exist_ok=True)
    print()

    # Parse mol2
    print("Step 1: Parsing mol2 file...")
    atoms, bonds = parse_mol2(mol2_path)
    print(f"  Total atoms: {len(atoms)}")
    print(f"  Total bonds: {len(bonds)}")

    # Identify atom categories
    print("\nStep 2: Identifying atom categories...")
    categories = identify_cys_and_ligand_atoms(atoms, bonds)
    print(f"  Backbone atoms: {len(categories['backbone'])}")
    print(f"  N-cap atoms:    {len(categories['n_cap'])}")
    print(f"  C-cap atoms:    {len(categories['c_cap'])}")
    print(f"  Ligand atoms:   {len(categories['ligand'])}")

    # Calculate charges
    charges = calculate_cap_charges(atoms, categories)
    print(f"\n  Charge distribution:")
    print(f"    Backbone: {charges['backbone']:.4f}")
    print(f"    N-cap:    {charges['n_cap']:.4f}")
    print(f"    C-cap:    {charges['c_cap']:.4f}")
    print(f"    Ligand:   {charges['ligand']:.4f}")
    print(f"    Total:    {sum(charges.values()):.4f}")

    # Create RTP entry
    print("\nStep 3: Creating RTP entry...")
    if args.use_amber_backbone_charges:
        print("  Using AMBER charges for backbone (not recommended for FEP)")
    else:
        print("  Using AM1-BCC charges throughout (recommended for FEP)")
    rtp_content, name_map = create_cyl_rtp_entry(
        atoms, bonds, categories, args.residue_name,
        use_amber_backbone_charges=args.use_amber_backbone_charges
    )

    # Save standalone RTP file
    rtp_file = res_dir / f"{args.residue_name.lower()}.rtp"
    with open(rtp_file, 'w') as f:
        f.write(rtp_content)
    print(f"  Standalone RTP: {rtp_file}")

    # Add to force field
    print(f"  Adding {args.residue_name} to {ff_dir}/aminoacids.rtp...")
    added = add_rtp_entry_to_forcefield(ff_dir, rtp_content, args.residue_name)
    if added:
        print(f"  Successfully added {args.residue_name} to aminoacids.rtp")
    else:
        print(f"  {args.residue_name} already exists in aminoacids.rtp")

    # Add HDB entry so pdb2gmx can build backbone H
    print(f"  Adding {args.residue_name} to {ff_dir}/aminoacids.hdb...")
    hdb_added = add_hdb_entry_to_forcefield(ff_dir, args.residue_name)
    if hdb_added:
        print(f"  Successfully added {args.residue_name} to aminoacids.hdb")
    else:
        print(f"  {args.residue_name} already exists in aminoacids.hdb (or hdb not found)")

    # Add residue to residuetypes.dat so pdb2gmx recognizes it as a protein residue
    add_residue_to_residuetypes(output_dir, args.residue_name, "Protein")

    if name_map:
        print(f"  Renamed atoms: {name_map}")

    # Extract GAFF2 atomtypes
    print("\nStep 4: Extracting GAFF2 atomtypes...")

    # Parse acpype ITP for accurate LJ parameters if provided
    acpype_atomtypes = None
    if args.acpype_itp:
        acpype_itp_path = Path(args.acpype_itp)
        if acpype_itp_path.exists():
            print(f"  Parsing atomtypes from acpype: {acpype_itp_path.name}")
            acpype_atomtypes = parse_acpype_atomtypes(acpype_itp_path)
            print(f"  Found {len(acpype_atomtypes)} atomtypes in acpype output")
        else:
            print(f"  WARNING: acpype ITP not found: {acpype_itp_path}")

    gaff2_types = extract_gaff2_atomtypes(atoms, categories, acpype_atomtypes)
    print(f"  GAFF2 types needed: {len(gaff2_types)}")
    print(f"  Types: {[t['name'] for t in gaff2_types]}")

    # Save standalone atomtypes file
    atomtypes_itp = res_dir / "gaff2_atomtypes.itp"
    with open(atomtypes_itp, 'w') as f:
        f.write("; GAFF2 atomtypes for ligand portion of CYL residue\n")
        f.write("[ atomtypes ]\n")
        f.write("; name  mass     charge  ptype  sigma      epsilon\n")
        for at in gaff2_types:
            f.write(f"  {at['name']:<4s}  {at['mass']:<8.4f} {at['charge']:<7.4f} "
                   f"{at['ptype']}  {at['sigma']:.5e}  {at['epsilon']:.5e}\n")
    print(f"  Standalone atomtypes: {atomtypes_itp}")

    # Add to force field
    print(f"  Adding GAFF2 atomtypes to {ff_dir}...")
    ff_gaff2_file = add_gaff2_atomtypes_to_forcefield(ff_dir, gaff2_types)
    print(f"  Created: {ff_gaff2_file}")

    # Create PDB
    print("\nStep 5: Creating CYL residue PDB...")
    pdb_content = create_cyl_pdb(atoms, categories, name_map, args.residue_name)

    pdb_file = res_dir / f"{args.residue_name.lower()}_residue.pdb"
    with open(pdb_file, 'w') as f:
        f.write(pdb_content)
    print(f"  Written: {pdb_file}")

    # Save metadata
    meta = {
        'mol2_file': str(mol2_path),
        'residue_name': args.residue_name,
        'force_field': str(ff_dir),
        'gmx_top_dir': str(gmx_top_dir),
        'charge_mode': 'AM1-BCC' if not args.use_amber_backbone_charges else 'AMBER-backbone',
        'n_backbone_atoms': len(categories['backbone']),
        'n_ligand_atoms': len(categories['ligand']),
        'n_cap_atoms_removed': len(categories['n_cap']) + len(categories['c_cap']),
        'charges': charges,
        'gaff2_types': [t['name'] for t in gaff2_types],
        'name_map': name_map
    }

    meta_file = res_dir / f"{args.residue_name.lower()}_meta.json"
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    print()
    print("=" * 60)
    print("Output Files")
    print("=" * 60)
    print()
    print("Force field (ready to use):")
    print(f"  {ff_dir}/")
    print(f"    - aminoacids.rtp (now includes {args.residue_name})")
    print(f"    - gaff2_atomtypes.itp (GAFF2 atom types)")
    print(f"    - ffnonbonded.itp (updated to include GAFF2)")
    print()
    print("Standalone files (for reference):")
    print(f"  {res_dir}/")
    print(f"    - {args.residue_name.lower()}.rtp")
    print(f"    - gaff2_atomtypes.itp")
    print(f"    - {args.residue_name.lower()}_residue.pdb")
    print(f"    - {args.residue_name.lower()}_meta.json")

    print()
    print("=" * 60)
    print("Next Steps")
    print("=" * 60)
    print()
    print("1. Run pdb2gmx with the custom force field:")
    print(f"   gmx pdb2gmx -f protein_with_{args.residue_name}.pdb -o protein.gro \\")
    print(f"       -p topol.top -ff {ff_name} -water tip3p")
    print()
    print("   Make sure to set GMXLIB to include the output directory:")
    print(f"   export GMXLIB={output_dir}:$GMXLIB")
    print()
    print("2. For RBFE (CYL1 -> CYL2):")
    print("   - Run this script for each ligand:")
    print(f"     python b05_create_cyl_residue.py --adduct-mol2 ligand1.mol2 --residue-name CYL1 \\")
    print(f"         --output-dir {output_dir} --skip-ff-copy")
    print(f"     python b05_create_cyl_residue.py --adduct-mol2 ligand2.mol2 --residue-name CYL2 \\")
    print(f"         --output-dir {output_dir} --skip-ff-copy")
    print("   - Use pmx hybrid topology to setup the transformation")
    print("   - Backbone atoms are conserved, only ligand atoms transform")
    print()

    return 0


if __name__ == '__main__':
    exit(main())
