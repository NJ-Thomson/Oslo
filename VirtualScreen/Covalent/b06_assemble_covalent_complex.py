#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
b06_assemble_covalent_complex.py

Assemble a covalent complex PDB where the CYL residue replaces CYS in the protein.

For pdb2gmx to work with the CYL residue, the complex must have:
1. The CYS backbone atoms (N, H, CA, HA, CB, HB2, HB3, SG) from the protein
2. The ligand atoms (with correct names matching the RTP) attached to SG
3. All atoms as ATOM records in the same chain (not HETATM)
4. Residue name CYL instead of CYS

This script:
1. Reads the docked complex (protein + ligand from gnina)
2. Reads the parameterized ligand (full hydrogens from mol2/acpype)
3. Replaces CYS with CYL at the covalent position
4. Outputs a PDB ready for pdb2gmx

Usage:
    python b06_assemble_covalent_complex.py \\
        --complex Outputs/Covalent/docking/complex.pdb \\
        --ligand-mol2 Outputs/Covalent/params/adduct_gaff2.mol2 \\
        --cyl-meta Outputs/Covalent/md_prep/cyl_residue/cyl_meta.json \\
        --cys-resid 1039 \\
        --output Outputs/Covalent/md_prep/complex_cyl.pdb
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


def parse_pdb_atoms(pdb_path: Path) -> List[dict]:
    """Parse PDB file and return list of atom dicts."""
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atoms.append({
                    'record': line[:6].strip(),
                    'serial': int(line[6:11]),
                    'name': line[12:16].strip(),
                    'altloc': line[16],
                    'resname': line[17:20].strip(),
                    'chain': line[21],
                    'resseq': int(line[22:26]),
                    'icode': line[26],
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]) if line[54:60].strip() else 1.0,
                    'tempfactor': float(line[60:66]) if len(line) > 60 and line[60:66].strip() else 0.0,
                    'element': line[76:78].strip() if len(line) > 76 else '',
                    'line': line
                })
    return atoms


def parse_mol2(mol2_path: Path) -> Tuple[List[dict], List[Tuple[int, int]]]:
    """Parse mol2 file to extract atoms and bonds."""
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


def identify_ligand_atoms(atoms: List[dict], bonds: List[Tuple[int, int]]) -> set:
    """
    Identify ligand atoms in the mol2 (everything attached to SG that isn't CYS backbone).
    """
    # Build adjacency list
    adj = {a['idx']: [] for a in atoms}
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)

    # Find key atoms by name
    atom_by_name = {a['name']: a['idx'] for a in atoms}

    sg_idx = atom_by_name.get('SG')
    cb_idx = atom_by_name.get('CB')

    if not sg_idx or not cb_idx:
        raise ValueError("Could not find SG or CB atoms")

    # CYS backbone atom names
    backbone_names = {'N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'SG'}
    backbone_indices = {atom_by_name[n] for n in backbone_names if n in atom_by_name}

    # Cap atoms (attached to N or C terminus) - exclude these
    n_idx = atom_by_name.get('N')
    cap_indices = set()
    if n_idx:
        for neighbor in adj[n_idx]:
            atom = next(a for a in atoms if a['idx'] == neighbor)
            if atom['name'] not in backbone_names:
                cap_indices.add(neighbor)
                for nn in adj[neighbor]:
                    if nn not in backbone_indices:
                        cap_indices.add(nn)

    # C-cap
    cc_idx = atom_by_name.get('CC')
    if cc_idx:
        cap_indices.add(cc_idx)
        for neighbor in adj[cc_idx]:
            atom = next(a for a in atoms if a['idx'] == neighbor)
            if atom['name'].startswith('HC'):
                cap_indices.add(neighbor)

    # Ligand: everything attached to SG that isn't CB
    ligand_indices = set()

    def find_ligand_atoms(start_idx, visited):
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

    for neighbor in adj[sg_idx]:
        if neighbor != cb_idx:
            find_ligand_atoms(neighbor, set())

    return ligand_indices


def superimpose_ligand(protein_sg_coords: np.ndarray, mol2_atoms: List[dict],
                        ligand_indices: set) -> Dict[int, np.ndarray]:
    """
    Calculate translation to align mol2 ligand with protein SG position.

    Returns dict mapping atom idx to new coordinates.
    """
    # Find SG in mol2
    sg_atom = next(a for a in mol2_atoms if a['name'] == 'SG')
    sg_mol2 = np.array([sg_atom['x'], sg_atom['y'], sg_atom['z']])

    # Translation vector
    translation = protein_sg_coords - sg_mol2

    # Apply to ligand atoms
    new_coords = {}
    for a in mol2_atoms:
        if a['idx'] in ligand_indices:
            coords = np.array([a['x'], a['y'], a['z']])
            new_coords[a['idx']] = coords + translation

    return new_coords


def format_pdb_atom(serial: int, name: str, resname: str, chain: str, resseq: int,
                    x: float, y: float, z: float, element: str = '') -> str:
    """
    Format an ATOM line in strict PDB format.

    PDB column positions (1-indexed):
      1-6:   Record name "ATOM  "
      7-11:  Atom serial number (right-justified)
      12:    Space
      13-16: Atom name (special formatting: element symbol in cols 13-14)
      17:    Alternate location indicator (space)
      18-20: Residue name (right-justified)
      21:    Space
      22:    Chain identifier
      23-26: Residue sequence number (right-justified)
      27:    iCode (space)
      28-30: Spaces
      31-38: X coordinate (8.3f)
      39-46: Y coordinate (8.3f)
      47-54: Z coordinate (8.3f)
      55-60: Occupancy (6.2f)
      61-66: Temperature factor (6.2f)
      67-76: Spaces
      77-78: Element symbol (right-justified)
    """
    # Atom name formatting:
    # - 1-char element names (C, N, O, S, H): start at column 14, left-pad with space
    # - 2-char element names or 4-char atom names: start at column 13
    if len(name) == 4:
        name_field = name
    elif len(name) == 3:
        name_field = f" {name}"
    elif len(name) == 2:
        name_field = f" {name} "
    else:  # len == 1
        name_field = f" {name}  "

    # Ensure name_field is exactly 4 characters
    name_field = name_field[:4].ljust(4)

    # Build the line with exact column positions
    # Columns: 1-6 (ATOM  ) + 7-11 (serial) + 12 (space) + 13-16 (name) + 17 (altloc)
    #        + 18-20 (resname) + 21 (space) + 22 (chain) + 23-26 (resseq) + 27 (icode)
    #        + 28-30 (spaces) + 31-38 (x) + 39-46 (y) + 47-54 (z)
    #        + 55-60 (occ) + 61-66 (temp) + 67-76 (spaces) + 77-78 (element)
    line = (
        f"ATOM  "              # 1-6
        f"{serial:5d}"         # 7-11
        f" "                   # 12
        f"{name_field}"        # 13-16
        f" "                   # 17 (altloc)
        f"{resname:>3s}"       # 18-20
        f" "                   # 21
        f"{chain}"             # 22
        f"{resseq:4d}"         # 23-26
        f" "                   # 27 (icode)
        f"   "                 # 28-30
        f"{x:8.3f}"            # 31-38
        f"{y:8.3f}"            # 39-46
        f"{z:8.3f}"            # 47-54
        f"{1.0:6.2f}"          # 55-60 (occupancy)
        f"{0.0:6.2f}"          # 61-66 (temp factor)
        f"          "          # 67-76
        f"{element:>2s}"       # 77-78
        f"\n"
    )
    return line


def assemble_covalent_complex(
    complex_pdb: Path,
    mol2_path: Path,
    cyl_meta: dict,
    cys_resid: int,
    output_pdb: Path,
    cyl_resname: str = "CYL"
) -> dict:
    """
    Assemble the covalent complex with CYL residue.

    Args:
        complex_pdb: Docked complex from gnina
        mol2_path: Parameterized adduct mol2
        cyl_meta: Metadata from b05_create_cyl_residue.py
        cys_resid: Residue number of covalent cysteine
        output_pdb: Output PDB path
        cyl_resname: Residue name for CYL (default: CYL)

    Returns:
        dict with metadata
    """
    # Parse inputs
    complex_atoms = parse_pdb_atoms(complex_pdb)
    mol2_atoms, mol2_bonds = parse_mol2(mol2_path)

    # Identify ligand atoms in mol2
    ligand_indices = identify_ligand_atoms(mol2_atoms, mol2_bonds)

    # Get name mapping from CYL metadata
    name_map = cyl_meta.get('name_map', {})

    # Find protein SG coordinates
    sg_atom = None
    for a in complex_atoms:
        if a['resseq'] == cys_resid and a['name'] == 'SG':
            sg_atom = a
            break

    if not sg_atom:
        raise ValueError(f"Could not find SG atom at residue {cys_resid}")

    protein_sg_coords = np.array([sg_atom['x'], sg_atom['y'], sg_atom['z']])

    # Translate ligand coordinates to align with protein
    ligand_new_coords = superimpose_ligand(protein_sg_coords, mol2_atoms, ligand_indices)

    # Build output
    output_lines = []
    serial = 0
    protein_chain = sg_atom['chain']
    cyl_written = False
    atoms_written = {'protein': 0, 'cyl_backbone': 0, 'cyl_ligand': 0}

    for a in complex_atoms:
        # Skip ligand HETATM (we'll use mol2 coordinates)
        if a['record'] == 'HETATM':
            continue

        # Check if this is the covalent cysteine
        if a['resseq'] == cys_resid and a['resname'] in ('CYS', 'CYX'):
            if not cyl_written:
                # Write CYL residue (backbone from protein, ligand from mol2)

                # CYS backbone atoms in order
                backbone_order = ['N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'SG', 'C', 'O']

                # Get backbone heavy atoms from protein
                cys_atoms = {atom['name']: atom for atom in complex_atoms
                            if atom['resseq'] == cys_resid and atom['resname'] in ('CYS', 'CYX')}

                # Get backbone atoms from mol2 (for hydrogens H, HA, HB2, HB3)
                mol2_backbone = {a['name']: a for a in mol2_atoms if a['name'] in backbone_order}

                # Calculate translation from mol2 SG to protein SG
                mol2_sg = next(a for a in mol2_atoms if a['name'] == 'SG')
                mol2_sg_coords = np.array([mol2_sg['x'], mol2_sg['y'], mol2_sg['z']])
                translation = protein_sg_coords - mol2_sg_coords

                # Write backbone atoms
                for name in backbone_order:
                    if name in cys_atoms:
                        # Use protein coordinates for atoms present in protein
                        atom = cys_atoms[name]
                        serial += 1
                        output_lines.append(format_pdb_atom(
                            serial, name, cyl_resname, protein_chain, cys_resid,
                            atom['x'], atom['y'], atom['z'],
                            name[0] if name[0] in 'CNOSH' else ''
                        ))
                    elif name in mol2_backbone:
                        # Use mol2 coordinates (translated) for atoms missing from protein (hydrogens)
                        mol2_atom = mol2_backbone[name]
                        coords = np.array([mol2_atom['x'], mol2_atom['y'], mol2_atom['z']]) + translation
                        serial += 1
                        output_lines.append(format_pdb_atom(
                            serial, name, cyl_resname, protein_chain, cys_resid,
                            coords[0], coords[1], coords[2],
                            'H' if name[0] == 'H' else name[0]
                        ))
                        atoms_written['cyl_backbone'] += 1

                # Write ligand atoms (from mol2 with translated coordinates)
                mol2_idx_to_atom = {a['idx']: a for a in mol2_atoms}
                for idx in sorted(ligand_indices):
                    atom = mol2_idx_to_atom[idx]
                    coords = ligand_new_coords[idx]

                    # Apply name mapping
                    atom_name = name_map.get(atom['name'], atom['name'])

                    serial += 1
                    element = atom['name'][0] if atom['name'][0] in 'CNOSH' else 'C'
                    output_lines.append(format_pdb_atom(
                        serial, atom_name, cyl_resname, protein_chain, cys_resid,
                        coords[0], coords[1], coords[2], element
                    ))
                    atoms_written['cyl_ligand'] += 1

                cyl_written = True

            # Skip remaining CYS atoms (we've written them above)
            continue

        # Write normal protein atom
        serial += 1
        output_lines.append(format_pdb_atom(
            serial, a['name'], a['resname'], a['chain'], a['resseq'],
            a['x'], a['y'], a['z'], a['element']
        ))
        atoms_written['protein'] += 1

    # Add TER and END
    # TER format: columns 1-6 "TER   ", 7-11 serial, 12-17 spaces, 18-20 resname, 21 space, 22 chain, 23-26 resseq
    output_lines.append(f"TER   {serial+1:5d}      {cyl_resname:>3s} {protein_chain}{cys_resid:4d}\n")
    output_lines.append("END\n")

    # Write output
    with open(output_pdb, 'w') as f:
        f.writelines(output_lines)

    meta = {
        'input_complex': str(complex_pdb),
        'input_mol2': str(mol2_path),
        'output_pdb': str(output_pdb),
        'cys_resid': cys_resid,
        'cyl_resname': cyl_resname,
        'atoms_written': atoms_written,
        'total_atoms': serial,
        'ligand_atoms': len(ligand_indices)
    }

    return meta


def main():
    ap = argparse.ArgumentParser(
        description='Assemble covalent complex with CYL residue for pdb2gmx',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python b06_assemble_covalent_complex.py \\
        --complex Outputs/Covalent/docking/complex.pdb \\
        --ligand-mol2 Outputs/Covalent/params/adduct_gaff2.mol2 \\
        --cyl-meta Outputs/Covalent/md_prep/cyl_residue/cyl_meta.json \\
        --cys-resid 1039 \\
        --output Outputs/Covalent/md_prep/complex_cyl.pdb

Then run pdb2gmx:
    cd Outputs/Covalent/md_prep
    gmx pdb2gmx -f complex_cyl.pdb -o protein.gro -p topol.top \\
        -ff amber99sb-ildn-cyl -water tip3p -ignh
"""
    )

    ap.add_argument('--complex', required=True,
                    help='Docked complex PDB from gnina')
    ap.add_argument('--ligand-mol2', required=True,
                    help='Parameterized adduct mol2 (from b05)')
    ap.add_argument('--cyl-meta', required=True,
                    help='CYL metadata JSON from b05')
    ap.add_argument('--cys-resid', type=int, required=True,
                    help='Residue number of covalent cysteine')
    ap.add_argument('--output', required=True,
                    help='Output PDB path')
    ap.add_argument('--cyl-resname', default='CYL',
                    help='Residue name for CYL (default: CYL)')

    args = ap.parse_args()

    complex_pdb = Path(args.complex)
    mol2_path = Path(args.ligand_mol2)
    cyl_meta_path = Path(args.cyl_meta)
    output_pdb = Path(args.output)

    # Validate inputs
    for path, name in [(complex_pdb, 'complex'), (mol2_path, 'mol2'), (cyl_meta_path, 'metadata')]:
        if not path.exists():
            print(f"ERROR: {name} file not found: {path}")
            return 1

    # Load CYL metadata
    with open(cyl_meta_path) as f:
        cyl_meta = json.load(f)

    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Assembling Covalent Complex")
    print("=" * 60)
    print(f"Complex PDB:    {complex_pdb}")
    print(f"Ligand mol2:    {mol2_path}")
    print(f"CYL metadata:   {cyl_meta_path}")
    print(f"Covalent CYS:   {args.cys_resid}")
    print(f"Output:         {output_pdb}")
    print()

    meta = assemble_covalent_complex(
        complex_pdb=complex_pdb,
        mol2_path=mol2_path,
        cyl_meta=cyl_meta,
        cys_resid=args.cys_resid,
        output_pdb=output_pdb,
        cyl_resname=args.cyl_resname
    )

    print("Results:")
    print(f"  Protein atoms:     {meta['atoms_written']['protein']}")
    print(f"  CYL backbone:      {meta['atoms_written']['cyl_backbone']}")
    print(f"  CYL ligand:        {meta['atoms_written']['cyl_ligand']}")
    print(f"  Total atoms:       {meta['total_atoms']}")
    print()
    print(f"Output written to: {output_pdb}")
    print()
    print("=" * 60)
    print("Next Steps")
    print("=" * 60)
    print()
    print("Run pdb2gmx with the custom force field:")
    print()
    print(f"  cd {output_pdb.parent}")
    print(f"  gmx pdb2gmx -f {output_pdb.name} -o protein.gro -p topol.top \\")
    print(f"      -ff amber99sb-ildn-cyl -water tip3p -ignh")
    print()

    return 0


if __name__ == '__main__':
    exit(main())
