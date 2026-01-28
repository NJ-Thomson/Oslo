#!/usr/bin/env python3
"""
RBFE Setup using pmx for Covalent Ligand Transformations

Complete workflow for setting up Relative Binding Free Energy calculations
for covalent ligand transformations (e.g., CX1 → CX2) using pmx.

Supports two modes:
1. RTP mode (recommended): Uses residue definitions from force field RTP files
2. ITP mode: Uses standalone ACPYPE ITP files (legacy)

Workflow:
1. Extract ligand portions from covalent residue definitions
2. Run pmx atomMapping to find atom correspondences
3. Run pmx ligandHybrid to create hybrid topology
4. Integrate hybrid topology into protein system
5. Set up lambda windows (default: 14 windows, 3ns each)
6. Generate run scripts for all simulations

Usage (RTP mode - recommended):
    python setup_rbfe_pmx.py \\
        --stateA_dir Outputs/Covalent/Inhib_32_acry/md_simulation \\
        --stateB_dir Outputs/Covalent/Inhib_32_chlo/md_simulation \\
        --stateA_resname CX1 \\
        --stateB_resname CX2 \\
        --output_dir Outputs/Covalent/RBFE_CX1_to_CX2 \\
        --n_windows 14 \\
        --prod_time 3

Usage (ITP mode - legacy):
    python setup_rbfe_pmx.py \\
        --stateA_dir Outputs/Covalent/Inhib_32_acry/md_simulation \\
        --stateB_dir Outputs/Covalent/Inhib_32_chlo/md_simulation \\
        --stateA_itp path/to/acpype/adduct_merged_GMX.itp \\
        --stateB_itp path/to/acpype/adduct_merged_GMX.itp \\
        --stateA_gro path/to/acpype/adduct_merged_GMX.gro \\
        --stateB_gro path/to/acpype/adduct_merged_GMX.gro \\
        --output_dir Outputs/Covalent/RBFE_CX1_to_CX2
"""

import argparse
import os
import sys
import shutil
import subprocess
import re
from pathlib import Path


# ============================================================================
# Utility Functions
# ============================================================================

def find_gmx():
    """Find GROMACS executable."""
    for cmd in ['gmx', 'gmx_mpi']:
        try:
            result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                return cmd
        except FileNotFoundError:
            continue
    return None


def run_command(cmd, cwd=None, check=True):
    """Run a shell command and return output."""
    print(f"  Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(
        cmd, shell=isinstance(cmd, str), cwd=cwd,
        capture_output=True, text=True
    )
    if check and result.returncode != 0:
        print(f"  STDOUT: {result.stdout}")
        print(f"  STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed with return code {result.returncode}")
    return result


# ============================================================================
# ITP Parsing Functions
# ============================================================================

def parse_itp_full(itp_path):
    """Parse ITP file and extract all sections."""
    with open(itp_path) as f:
        content = f.read()

    result = {
        'atomtypes': [],
        'atoms': [],
        'bonds': [],
        'angles': [],
        'propers': [],
        'impropers': []
    }

    # Parse atomtypes
    atomtypes_match = re.search(r'\[ atomtypes \]\s*\n(.*?)(?:\n\[|\Z)', content, re.DOTALL)
    if atomtypes_match:
        for line in atomtypes_match.group(1).strip().split('\n'):
            if line.strip() and not line.startswith(';'):
                result['atomtypes'].append(line)

    # Parse atoms
    atoms_match = re.search(r'\[ atoms \]\s*\n(.*?)(?:\n\[|\Z)', content, re.DOTALL)
    if atoms_match:
        for line in atoms_match.group(1).strip().split('\n'):
            if line.strip() and not line.startswith(';'):
                parts = line.split()
                if len(parts) >= 7:
                    result['atoms'].append({
                        'nr': int(parts[0]),
                        'type': parts[1],
                        'resnr': int(parts[2]),
                        'resname': parts[3],
                        'atom': parts[4],
                        'cgnr': int(parts[5]),
                        'charge': float(parts[6]),
                        'mass': float(parts[7]) if len(parts) > 7 else 0.0
                    })

    # Parse bonds
    bonds_match = re.search(r'\[ bonds \]\s*\n(.*?)(?:\n\[|\Z)', content, re.DOTALL)
    if bonds_match:
        for line in bonds_match.group(1).strip().split('\n'):
            if line.strip() and not line.startswith(';'):
                parts = line.split()
                if len(parts) >= 5:
                    result['bonds'].append({
                        'i': int(parts[0]), 'j': int(parts[1]),
                        'funct': int(parts[2]),
                        'r': float(parts[3]), 'k': float(parts[4])
                    })

    # Parse angles
    angles_match = re.search(r'\[ angles \]\s*\n(.*?)(?:\n\[|\Z)', content, re.DOTALL)
    if angles_match:
        for line in angles_match.group(1).strip().split('\n'):
            if line.strip() and not line.startswith(';'):
                parts = line.split()
                if len(parts) >= 6:
                    result['angles'].append({
                        'i': int(parts[0]), 'j': int(parts[1]), 'k': int(parts[2]),
                        'funct': int(parts[3]),
                        'theta': float(parts[4]), 'cth': float(parts[5])
                    })

    # Parse proper dihedrals
    dih_match = re.search(r'\[ dihedrals \].*?; propers\s*\n(.*?)(?:\n\[ dihedrals \]|\Z)', content, re.DOTALL)
    if dih_match:
        for line in dih_match.group(1).strip().split('\n'):
            if line.strip() and not line.startswith(';'):
                parts = line.split()
                if len(parts) >= 8:
                    result['propers'].append({
                        'i': int(parts[0]), 'j': int(parts[1]),
                        'k': int(parts[2]), 'l': int(parts[3]),
                        'funct': int(parts[4]),
                        'phase': float(parts[5]), 'kd': float(parts[6]), 'pn': int(parts[7])
                    })

    # Parse improper dihedrals
    imp_match = re.search(r'\[ dihedrals \].*?; impropers\s*\n(.*?)(?:\n\[|\Z)', content, re.DOTALL)
    if imp_match:
        for line in imp_match.group(1).strip().split('\n'):
            if line.strip() and not line.startswith(';'):
                parts = line.split()
                if len(parts) >= 8:
                    result['impropers'].append({
                        'i': int(parts[0]), 'j': int(parts[1]),
                        'k': int(parts[2]), 'l': int(parts[3]),
                        'funct': int(parts[4]),
                        'phase': float(parts[5]), 'kd': float(parts[6]), 'pn': int(parts[7])
                    })

    return result


# ============================================================================
# RTP Parsing Functions
# ============================================================================

def parse_rtp_residue(rtp_path, resname):
    """
    Parse a residue definition from an RTP file.

    RTP format:
        [ RESNAME ]
        [ atoms ]
        atomname  atomtype  charge  cgnr
        ...
        [ bonds ]
        atom1  atom2
        ...
        [ impropers ] (optional)
        ...

    Args:
        rtp_path: Path to aminoacids.rtp file
        resname: Residue name to extract (e.g., 'CX1', 'CX2')

    Returns:
        dict with 'atoms' and 'bonds' lists
    """
    with open(rtp_path) as f:
        content = f.read()

    # Find the residue block
    pattern = rf'\[ {resname} \](.*?)(?:\n\[(?! )|$)'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Residue '{resname}' not found in {rtp_path}")

    residue_block = match.group(1)

    result = {
        'atoms': [],
        'bonds': [],
        'impropers': []
    }

    # Parse atoms section
    atoms_match = re.search(r'\[ atoms \]\s*\n(.*?)(?:\n \[|\Z)', residue_block, re.DOTALL)
    if atoms_match:
        for line in atoms_match.group(1).strip().split('\n'):
            line = line.strip()
            if line and not line.startswith(';'):
                parts = line.split()
                if len(parts) >= 4:
                    result['atoms'].append({
                        'name': parts[0],
                        'type': parts[1],
                        'charge': float(parts[2]),
                        'cgnr': int(parts[3])
                    })

    # Parse bonds section
    bonds_match = re.search(r'\[ bonds \]\s*\n(.*?)(?:\n \[|\Z)', residue_block, re.DOTALL)
    if bonds_match:
        for line in bonds_match.group(1).strip().split('\n'):
            line = line.strip()
            if line and not line.startswith(';'):
                parts = line.split()
                if len(parts) >= 2:
                    # Skip backbone bonds (-C, +N)
                    if not (parts[0].startswith('-') or parts[0].startswith('+') or
                            parts[1].startswith('-') or parts[1].startswith('+')):
                        result['bonds'].append((parts[0], parts[1]))

    # Parse impropers section (if present)
    imp_match = re.search(r'\[ impropers \]\s*\n(.*?)(?:\n \[|\Z)', residue_block, re.DOTALL)
    if imp_match:
        for line in imp_match.group(1).strip().split('\n'):
            line = line.strip()
            if line and not line.startswith(';'):
                parts = line.split()
                if len(parts) >= 4:
                    result['impropers'].append((parts[0], parts[1], parts[2], parts[3]))

    return result


def parse_ffbonded(ff_dir):
    """
    Parse bond, angle, and dihedral parameters from force field.

    Args:
        ff_dir: Path to force field directory (e.g., amber99sb-ildn-cx1.ff)

    Returns:
        dict with 'bondtypes', 'angletypes', 'dihedraltypes'
    """
    result = {
        'bondtypes': {},
        'angletypes': {},
        'dihedraltypes': {}
    }

    ffbonded_path = Path(ff_dir) / 'ffbonded.itp'
    if not ffbonded_path.exists():
        print(f"  WARNING: {ffbonded_path} not found")
        return result

    with open(ffbonded_path) as f:
        content = f.read()

    # Parse bondtypes
    bondtypes_match = re.search(r'\[ bondtypes \]\s*\n(.*?)(?:\n\[|\Z)', content, re.DOTALL)
    if bondtypes_match:
        for line in bondtypes_match.group(1).strip().split('\n'):
            line = line.strip()
            if line and not line.startswith(';'):
                parts = line.split()
                if len(parts) >= 5:
                    key = tuple(sorted([parts[0], parts[1]]))
                    result['bondtypes'][key] = {
                        'funct': int(parts[2]),
                        'r': float(parts[3]),
                        'k': float(parts[4])
                    }

    # Parse angletypes
    angletypes_match = re.search(r'\[ angletypes \]\s*\n(.*?)(?:\n\[|\Z)', content, re.DOTALL)
    if angletypes_match:
        for line in angletypes_match.group(1).strip().split('\n'):
            line = line.strip()
            if line and not line.startswith(';'):
                parts = line.split()
                if len(parts) >= 6:
                    # Angle key: i-j-k where j is central
                    key = (parts[0], parts[1], parts[2])
                    result['angletypes'][key] = {
                        'funct': int(parts[3]),
                        'theta': float(parts[4]),
                        'cth': float(parts[5])
                    }

    return result


def extract_ligand_from_rtp(rtp_data, ff_params, atom_masses):
    """
    Extract ligand portion from RTP residue data.

    Args:
        rtp_data: Output from parse_rtp_residue()
        ff_params: Output from parse_ffbonded()
        atom_masses: dict mapping atom type to mass

    Returns:
        dict with ligand atoms, bonds, angles (pmx-compatible format)
    """
    # Backbone atoms to exclude
    backbone_names = {'N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'SG', 'C', 'O'}

    # Build atom name to index mapping
    name_to_idx = {}
    atom_types = {}
    for i, atom in enumerate(rtp_data['atoms'], 1):
        name_to_idx[atom['name']] = i
        atom_types[atom['name']] = atom['type']

    # Identify ligand atoms (everything except backbone)
    ligand_atom_names = set()
    for atom in rtp_data['atoms']:
        if atom['name'] not in backbone_names:
            ligand_atom_names.add(atom['name'])

    # Create index mapping for ligand atoms
    old_to_new = {}
    new_idx = 1
    for atom in rtp_data['atoms']:
        if atom['name'] in ligand_atom_names:
            old_to_new[atom['name']] = new_idx
            new_idx += 1

    # Extract ligand atoms
    ligand_atoms = []
    for atom in rtp_data['atoms']:
        if atom['name'] in ligand_atom_names:
            mass = atom_masses.get(atom['type'].lower(), 12.0)  # Default to carbon mass
            ligand_atoms.append({
                'nr': old_to_new[atom['name']],
                'type': atom['type'],
                'resnr': 1,
                'resname': 'LIG',
                'atom': atom['name'],
                'cgnr': old_to_new[atom['name']],
                'charge': atom['charge'],
                'mass': mass
            })

    # Extract ligand bonds with parameters
    ligand_bonds = []
    for atom1, atom2 in rtp_data['bonds']:
        if atom1 in ligand_atom_names and atom2 in ligand_atom_names:
            # Look up bond parameters
            type1 = atom_types[atom1]
            type2 = atom_types[atom2]
            key = tuple(sorted([type1, type2]))

            params = ff_params['bondtypes'].get(key, {'funct': 1, 'r': 0.15, 'k': 250000.0})

            ligand_bonds.append({
                'i': old_to_new[atom1],
                'j': old_to_new[atom2],
                'funct': params['funct'],
                'r': params['r'],
                'k': params['k']
            })

    # Generate angles from bonds
    ligand_angles = []
    bond_graph = {}
    for bond in ligand_bonds:
        if bond['i'] not in bond_graph:
            bond_graph[bond['i']] = []
        if bond['j'] not in bond_graph:
            bond_graph[bond['j']] = []
        bond_graph[bond['i']].append(bond['j'])
        bond_graph[bond['j']].append(bond['i'])

    # Find all i-j-k angle triplets
    for j in bond_graph:
        neighbors = bond_graph[j]
        for idx1, i in enumerate(neighbors):
            for k in neighbors[idx1+1:]:
                # Look up atom types for angle parameters
                atom_i = ligand_atoms[i-1]['type']
                atom_j = ligand_atoms[j-1]['type']
                atom_k = ligand_atoms[k-1]['type']

                # Try both orderings
                key1 = (atom_i, atom_j, atom_k)
                key2 = (atom_k, atom_j, atom_i)

                params = (ff_params['angletypes'].get(key1) or
                         ff_params['angletypes'].get(key2) or
                         {'funct': 1, 'theta': 109.5, 'cth': 418.4})

                ligand_angles.append({
                    'i': i, 'j': j, 'k': k,
                    'funct': params['funct'],
                    'theta': params['theta'],
                    'cth': params['cth']
                })

    return {
        'atoms': ligand_atoms,
        'bonds': ligand_bonds,
        'angles': ligand_angles,
        'propers': [],  # Auto-generated by GROMACS
        'impropers': [],
        'atomtypes': [],  # Already in force field
        'old_to_new': old_to_new,
        'ligand_indices': {old_to_new[n] for n in ligand_atom_names}
    }


def get_atom_masses():
    """Return standard atomic masses for GAFF atom types."""
    return {
        # Common types
        'c': 12.01, 'c2': 12.01, 'c3': 12.01, 'cc': 12.01, 'cd': 12.01, 'ce': 12.01,
        'c6': 12.01, 'ca': 12.01, 'cp': 12.01,
        'n': 14.01, 'ns': 14.01, 'na': 14.01, 'nb': 14.01, 'nc': 14.01, 'nd': 14.01,
        'ne': 14.01, 'nf': 14.01, 'nu': 14.01, 'nh': 14.01,
        'o': 16.00, 'oh': 16.00, 'os': 16.00,
        's': 32.07, 'ss': 32.07, 'sh': 32.07,
        'h1': 1.008, 'h2': 1.008, 'h3': 1.008, 'h4': 1.008, 'h5': 1.008,
        'ha': 1.008, 'hc': 1.008, 'hn': 1.008, 'ho': 1.008, 'hp': 1.008,
        # Standard AMBER types
        'ct': 12.01, 'c*': 12.01, 'cw': 12.01,
        'n2': 14.01, 'n3': 14.01,
        'hw': 1.008, 'h': 1.008,
        'ow': 16.00,
    }


def write_ligand_pdb_from_gro(ligand_data, gro_path, output_path, resname, mol_name='LIG'):
    """
    Extract ligand coordinates from equilibrated GRO file and write PDB.

    Args:
        ligand_data: Ligand data with atom names
        gro_path: Path to equilibrated GRO file (e.g., em.gro or npt.gro)
        output_path: Output PDB path
        resname: Residue name to find in GRO (e.g., 'CX1', 'CX2')
        mol_name: Output molecule name for PDB
    """
    with open(gro_path) as f:
        lines = f.readlines()

    n_atoms = int(lines[1].strip())

    # Build coordinate lookup from GRO
    gro_coords = {}
    for i in range(2, 2 + n_atoms):
        line = lines[i]
        gro_resname = line[5:10].strip()
        atomname = line[10:15].strip()
        if gro_resname == resname:
            x = float(line[20:28]) * 10  # nm to Å
            y = float(line[28:36]) * 10
            z = float(line[36:44]) * 10
            gro_coords[atomname] = (x, y, z)

    if not gro_coords:
        raise ValueError(f"No atoms found for residue '{resname}' in {gro_path}")

    with open(output_path, 'w') as f:
        f.write(f"REMARK   Ligand extracted from {resname} for pmx\n")
        for atom in ligand_data['atoms']:
            name = atom['atom']
            if name in gro_coords:
                x, y, z = gro_coords[name]
            else:
                print(f"  WARNING: No coordinates for atom {name}, using origin")
                x, y, z = 0.0, 0.0, 0.0

            elem = atom['type'][0].upper() if atom['type'] else 'C'
            f.write(f"ATOM  {atom['nr']:5d} {name:>4s} {mol_name:>3s} A{1:4d}    "
                   f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}\n")
        f.write("END\n")

    return output_path


# ============================================================================
# Ligand Extraction for pmx (ITP mode - legacy)
# ============================================================================

def extract_ligand_portion(itp_data, ligand_start_atom='C1'):
    """
    Extract ligand portion from adduct ITP.

    The ligand portion starts at C1 (bonded to SG) and includes everything
    connected to C1 except SG and the backbone.
    """
    # Backbone atoms to exclude
    backbone_names = {'CN', 'HN1', 'HN2', 'HN3', 'N', 'H', 'CA', 'HA',
                      'CB', 'HB2', 'HB3', 'SG', 'CC', 'HC1', 'HC2', 'HC3'}

    # Find all ligand atom indices
    ligand_indices = set()
    for atom in itp_data['atoms']:
        if atom['atom'] not in backbone_names:
            ligand_indices.add(atom['nr'])

    # Create index mapping
    old_to_new = {}
    new_idx = 1
    for atom in itp_data['atoms']:
        if atom['nr'] in ligand_indices:
            old_to_new[atom['nr']] = new_idx
            new_idx += 1

    # Extract and renumber atoms
    ligand_atoms = []
    for atom in itp_data['atoms']:
        if atom['nr'] in ligand_indices:
            new_atom = atom.copy()
            new_atom['nr'] = old_to_new[atom['nr']]
            new_atom['cgnr'] = old_to_new[atom['nr']]
            ligand_atoms.append(new_atom)

    # Extract bonds
    ligand_bonds = []
    for bond in itp_data['bonds']:
        if bond['i'] in ligand_indices and bond['j'] in ligand_indices:
            new_bond = bond.copy()
            new_bond['i'] = old_to_new[bond['i']]
            new_bond['j'] = old_to_new[bond['j']]
            ligand_bonds.append(new_bond)

    # Extract angles
    ligand_angles = []
    for angle in itp_data['angles']:
        if all(idx in ligand_indices for idx in [angle['i'], angle['j'], angle['k']]):
            new_angle = angle.copy()
            new_angle['i'] = old_to_new[angle['i']]
            new_angle['j'] = old_to_new[angle['j']]
            new_angle['k'] = old_to_new[angle['k']]
            ligand_angles.append(new_angle)

    # Extract proper dihedrals
    ligand_propers = []
    for dih in itp_data['propers']:
        if all(idx in ligand_indices for idx in [dih['i'], dih['j'], dih['k'], dih['l']]):
            new_dih = dih.copy()
            new_dih['i'] = old_to_new[dih['i']]
            new_dih['j'] = old_to_new[dih['j']]
            new_dih['k'] = old_to_new[dih['k']]
            new_dih['l'] = old_to_new[dih['l']]
            ligand_propers.append(new_dih)

    # Extract improper dihedrals
    ligand_impropers = []
    for dih in itp_data['impropers']:
        if all(idx in ligand_indices for idx in [dih['i'], dih['j'], dih['k'], dih['l']]):
            new_dih = dih.copy()
            new_dih['i'] = old_to_new[dih['i']]
            new_dih['j'] = old_to_new[dih['j']]
            new_dih['k'] = old_to_new[dih['k']]
            new_dih['l'] = old_to_new[dih['l']]
            ligand_impropers.append(new_dih)

    return {
        'atoms': ligand_atoms,
        'bonds': ligand_bonds,
        'angles': ligand_angles,
        'propers': ligand_propers,
        'impropers': ligand_impropers,
        'atomtypes': itp_data['atomtypes'],
        'old_to_new': old_to_new,
        'ligand_indices': ligand_indices
    }


def write_ligand_itp(ligand_data, output_path, mol_name='LIG'):
    """Write ligand ITP file for pmx."""
    with open(output_path, 'w') as f:
        f.write(f"; Ligand portion for pmx - {mol_name}\n\n")

        f.write("[ atomtypes ]\n")
        for line in ligand_data['atomtypes']:
            f.write(f"{line}\n")
        f.write("\n")

        f.write("[ moleculetype ]\n")
        f.write(f"{mol_name}  3\n\n")

        f.write("[ atoms ]\n")
        for atom in ligand_data['atoms']:
            f.write(f"{atom['nr']:5d} {atom['type']:>5s} {1:5d} {mol_name:>5s} "
                   f"{atom['atom']:>5s} {atom['cgnr']:5d} {atom['charge']:10.6f} {atom['mass']:10.5f}\n")
        f.write("\n")

        f.write("[ bonds ]\n")
        for bond in ligand_data['bonds']:
            f.write(f"{bond['i']:5d} {bond['j']:5d} {bond['funct']:5d} "
                   f"{bond['r']:.5e} {bond['k']:.5e}\n")
        f.write("\n")

        f.write("[ angles ]\n")
        for angle in ligand_data['angles']:
            f.write(f"{angle['i']:5d} {angle['j']:5d} {angle['k']:5d} {angle['funct']:5d} "
                   f"{angle['theta']:.4e} {angle['cth']:.4e}\n")
        f.write("\n")

        if ligand_data['propers']:
            f.write("[ dihedrals ] ; propers\n")
            for dih in ligand_data['propers']:
                f.write(f"{dih['i']:5d} {dih['j']:5d} {dih['k']:5d} {dih['l']:5d} "
                       f"{dih['funct']:5d} {dih['phase']:.2f} {dih['kd']:.5e} {dih['pn']:5d}\n")
            f.write("\n")

        if ligand_data['impropers']:
            f.write("[ dihedrals ] ; impropers\n")
            for dih in ligand_data['impropers']:
                f.write(f"{dih['i']:5d} {dih['j']:5d} {dih['k']:5d} {dih['l']:5d} "
                       f"{dih['funct']:5d} {dih['phase']:.2f} {dih['kd']:.5e} {dih['pn']:5d}\n")

    return output_path


def write_ligand_pdb(ligand_data, gro_path, output_path, mol_name='LIG'):
    """Extract ligand atoms from GRO and write PDB for pmx."""
    with open(gro_path) as f:
        lines = f.readlines()

    n_atoms = int(lines[1].strip())

    # Build lookup from GRO
    gro_atoms = {}
    for i in range(2, 2 + n_atoms):
        line = lines[i]
        resname = line[5:10].strip()
        atomname = line[10:15].strip()
        x = float(line[20:28]) * 10  # nm to Å
        y = float(line[28:36]) * 10
        z = float(line[36:44]) * 10
        if resname == 'MOL':
            gro_atoms[atomname] = (x, y, z)

    with open(output_path, 'w') as f:
        f.write(f"REMARK   Ligand for pmx - {mol_name}\n")
        for atom in ligand_data['atoms']:
            name = atom['atom']
            x, y, z = gro_atoms.get(name, (0.0, 0.0, 0.0))
            elem = atom['type'][0].upper() if atom['type'] else 'C'
            f.write(f"ATOM  {atom['nr']:5d} {name:>4s} {mol_name:>3s} A{1:4d}    "
                   f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}\n")
        f.write("END\n")

    return output_path


# ============================================================================
# MDP File Generation
# ============================================================================

def write_fep_mdp(output_path, n_windows, lambda_idx, sim_type='em',
                  nsteps=None, dt=0.002, ref_temp=300):
    """
    Write MDP file for FEP simulation at a specific lambda.

    Args:
        output_path: Output file path
        n_windows: Total number of lambda windows
        lambda_idx: Index of this lambda window (0 to n_windows-1)
        sim_type: 'em', 'nvt', 'npt', or 'prod'
        nsteps: Number of steps (auto-calculated if None)
        dt: Timestep in ps
        ref_temp: Reference temperature in K
    """
    # Generate lambda schedule
    lambdas = [i / (n_windows - 1) for i in range(n_windows)]
    lambda_str = ' '.join(f'{l:.4f}' for l in lambdas)

    # Base settings for each simulation type
    if sim_type == 'em':
        base = f"""; Energy minimization for FEP - lambda {lambda_idx}
integrator          = steep
emtol               = 1000.0
emstep              = 0.01
nsteps              = {nsteps or 5000}

nstlog              = 100
nstenergy           = 100
"""
    elif sim_type == 'nvt':
        base = f"""; NVT equilibration for FEP - lambda {lambda_idx}
integrator          = md
dt                  = {dt}
nsteps              = {nsteps or 50000}  ; {(nsteps or 50000) * dt / 1000:.1f} ns

nstlog              = 1000
nstenergy           = 1000
nstxout-compressed  = 5000

tcoupl              = V-rescale
tc-grps             = System
tau_t               = 0.1
ref_t               = {ref_temp}

pcoupl              = no

gen_vel             = yes
gen_temp            = {ref_temp}
gen_seed            = -1

constraints         = h-bonds
constraint_algorithm = LINCS

define              = -DPOSRES
"""
    elif sim_type == 'npt':
        base = f"""; NPT equilibration for FEP - lambda {lambda_idx}
integrator          = md
dt                  = {dt}
nsteps              = {nsteps or 250000}  ; {(nsteps or 250000) * dt / 1000:.1f} ns

nstlog              = 1000
nstenergy           = 1000
nstxout-compressed  = 5000

tcoupl              = V-rescale
tc-grps             = System
tau_t               = 0.1
ref_t               = {ref_temp}

pcoupl              = Parrinello-Rahman
pcoupltype          = isotropic
tau_p               = 2.0
ref_p               = 1.0
compressibility     = 4.5e-5

gen_vel             = no
continuation        = yes

constraints         = h-bonds
constraint_algorithm = LINCS

define              = -DPOSRES
"""
    else:  # prod
        base = f"""; Production FEP - lambda {lambda_idx}
integrator          = md
dt                  = {dt}
nsteps              = {nsteps or 1500000}  ; {(nsteps or 1500000) * dt / 1000:.1f} ns

nstlog              = 5000
nstenergy           = 1000
nstxout-compressed  = 5000

tcoupl              = V-rescale
tc-grps             = System
tau_t               = 0.1
ref_t               = {ref_temp}

pcoupl              = Parrinello-Rahman
pcoupltype          = isotropic
tau_p               = 2.0
ref_p               = 1.0
compressibility     = 4.5e-5

gen_vel             = no
continuation        = yes

constraints         = h-bonds
constraint_algorithm = LINCS
"""

    # Common settings
    common = f"""
; Neighbor searching
cutoff-scheme       = Verlet
nstlist             = 20
pbc                 = xyz

; Electrostatics
coulombtype         = PME
rcoulomb            = 1.0
fourierspacing      = 0.12

; VdW
vdwtype             = Cut-off
rvdw                = 1.0
DispCorr            = EnerPres
"""

    # FEP settings
    fep = f"""
; Free energy settings
free_energy         = yes
init_lambda_state   = {lambda_idx}

; Lambda vectors (bonded, coul, vdw, restraint)
bonded_lambdas      = {lambda_str}
coul_lambdas        = {lambda_str}
vdw_lambdas         = {lambda_str}
; restraint_lambdas   = {lambda_str}

; Soft-core settings
sc-alpha            = 0.5
sc-power            = 1
sc-sigma            = 0.3
sc-coul             = yes

; FEP output
nstdhdl             = 100
calc-lambda-neighbors = -1
"""

    with open(output_path, 'w') as f:
        f.write(base)
        f.write(common)
        f.write(fep)

    return output_path


# ============================================================================
# Run Script Generation
# ============================================================================

def write_lambda_run_script(output_dir, lambda_idx, gmx='gmx', gpu=False):
    """Write run script for a single lambda window."""
    gpu_flag = '-nb gpu -pme gpu' if gpu else ''

    script = f"""#!/bin/bash
# FEP simulation for lambda window {lambda_idx}
# Run this script from within the lambda directory
set -e

GMX="{gmx}"

echo "Lambda window {lambda_idx}"
echo "========================"

# Energy minimization
if [ ! -f em.gro ]; then
    echo "Running energy minimization..."
    $GMX grompp -f em.mdp -c ../input/hybrid.gro -p ../input/topol.top -o em.tpr -maxwarn 5
    $GMX mdrun -deffnm em -v {gpu_flag}
fi

# NVT equilibration
if [ ! -f nvt.gro ]; then
    echo "Running NVT equilibration..."
    $GMX grompp -f nvt.mdp -c em.gro -r em.gro -p ../input/topol.top -o nvt.tpr -maxwarn 5
    $GMX mdrun -deffnm nvt -v {gpu_flag}
fi

# NPT equilibration
if [ ! -f npt.gro ]; then
    echo "Running NPT equilibration..."
    $GMX grompp -f npt.mdp -c nvt.gro -r nvt.gro -p ../input/topol.top -t nvt.cpt -o npt.tpr -maxwarn 5
    $GMX mdrun -deffnm npt -v {gpu_flag}
fi

# Production
if [ ! -f prod.gro ]; then
    echo "Running production..."
    $GMX grompp -f prod.mdp -c npt.gro -p ../input/topol.top -t npt.cpt -o prod.tpr -maxwarn 5
    $GMX mdrun -deffnm prod -v {gpu_flag}
fi

echo "Lambda {lambda_idx} complete!"
"""
    script_path = output_dir / 'run.sh'
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    return script_path


def write_master_run_script(output_dir, n_windows, gmx='gmx'):
    """Write master script to run all lambda windows."""
    script = f"""#!/bin/bash
# Master script to run all lambda windows
# Can be run sequentially or submitted to a queue

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "RBFE Simulation - {n_windows} Lambda Windows"
echo "=========================================="

for i in $(seq 0 {n_windows - 1}); do
    lambda_dir=$(printf "lambda%02d" $i)
    echo ""
    echo "Running $lambda_dir..."
    cd $lambda_dir
    ./run.sh
    cd ..
done

echo ""
echo "=========================================="
echo "All lambda windows complete!"
echo "=========================================="
echo ""
echo "To analyze results, run:"
echo "  python analyze_fep.py"
"""
    script_path = output_dir / 'run_all.sh'
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    return script_path


def write_analysis_script(output_dir, n_windows, gmx='gmx'):
    """Write FEP analysis script."""
    script = f"""#!/usr/bin/env python3
\"\"\"
FEP Analysis Script

Analyzes free energy results from all lambda windows using GROMACS BAR
or alchemlyb (if available).
\"\"\"

import os
import subprocess
import glob

# Use the directory where this script is located
output_dir = os.path.dirname(os.path.abspath(__file__)) or "."
n_windows = {n_windows}
gmx = "{gmx}"

print("=" * 60)
print("FEP Analysis")
print("=" * 60)

# Collect all dhdl files
dhdl_files = []
for i in range(n_windows):
    dhdl = os.path.join(output_dir, f"lambda{{i:02d}}", "prod.xvg")
    if os.path.exists(dhdl):
        dhdl_files.append(dhdl)
    else:
        print(f"WARNING: Missing dhdl file for lambda {{i}}")

if len(dhdl_files) < n_windows:
    print(f"ERROR: Only found {{len(dhdl_files)}}/{{n_windows}} dhdl files")
    exit(1)

print(f"Found {{len(dhdl_files)}} dhdl files")

# Try alchemlyb first
try:
    from alchemlyb.parsing.gmx import extract_dHdl, extract_u_nk
    from alchemlyb.estimators import BAR, MBAR, TI
    import pandas as pd

    print("\\nUsing alchemlyb for analysis...")

    # Extract data
    dhdl_data = pd.concat([extract_dHdl(f, T=300) for f in sorted(dhdl_files)])

    # BAR estimator
    bar = BAR()
    bar.fit(extract_u_nk(sorted(dhdl_files)[0], T=300))  # This needs u_nk

    print(f"\\nBAR estimate: {{bar.delta_f_.iloc[0, -1]:.3f}} +/- {{bar.d_delta_f_.iloc[0, -1]:.3f}} kT")

except ImportError:
    print("\\nalchemlyb not found, using GROMACS BAR...")

    # Use gmx bar
    xvg_list = " ".join(sorted(dhdl_files))
    cmd = f"{{gmx}} bar -f {{xvg_list}} -o bar.xvg -oi barint.xvg"

    print(f"Running: {{cmd}}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           cwd=output_dir)

    if result.returncode == 0:
        print("\\nBAR analysis complete!")
        print("Results saved to bar.xvg and barint.xvg")
        print(result.stdout)
    else:
        print("ERROR:", result.stderr)

print("\\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)
"""
    script_path = output_dir / 'analyze_fep.py'
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    return script_path


# ============================================================================
# Main Setup Function
# ============================================================================

def setup_rbfe(stateA_dir, stateB_dir, stateA_itp, stateB_itp,
               stateA_gro, stateB_gro, output_dir,
               n_windows=14, prod_time_ns=3, ref_temp=300, gpu=False):
    """
    Set up complete RBFE calculation using pmx.

    Args:
        stateA_dir: Path to state A (e.g., CX1) MD simulation directory
        stateB_dir: Path to state B (e.g., CX2) MD simulation directory
        stateA_itp: Path to state A ACPYPE ITP file
        stateB_itp: Path to state B ACPYPE ITP file
        stateA_gro: Path to state A ACPYPE GRO file
        stateB_gro: Path to state B ACPYPE GRO file
        output_dir: Output directory for RBFE setup
        n_windows: Number of lambda windows (default: 14)
        prod_time_ns: Production time per window in ns (default: 3)
        ref_temp: Reference temperature in K (default: 300)
        gpu: Use GPU acceleration
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gmx = find_gmx()
    if gmx is None:
        print("WARNING: GROMACS not found in PATH")
        print("         Setup will continue but run scripts will need gmx in PATH")
        gmx = 'gmx'  # Use default name in scripts

    print("=" * 70)
    print("RBFE Setup using pmx for Covalent Ligand Transformation")
    print("=" * 70)
    print(f"State A (λ=0): {stateA_dir}")
    print(f"State B (λ=1): {stateB_dir}")
    print(f"Output: {output_dir}")
    print(f"Lambda windows: {n_windows}")
    print(f"Production time: {prod_time_ns} ns per window")
    print(f"Total simulation time: {n_windows * prod_time_ns} ns")
    print()

    # Create subdirectories
    pmx_dir = output_dir / 'pmx_input'
    input_dir = output_dir / 'input'
    pmx_dir.mkdir(exist_ok=True)
    input_dir.mkdir(exist_ok=True)

    # ========================================================================
    # Step 1: Extract ligand portions for pmx
    # ========================================================================
    print("[Step 1/6] Extracting ligand portions for pmx...")

    # State A
    print("  Parsing state A...")
    stateA_data = parse_itp_full(stateA_itp)
    stateA_ligand = extract_ligand_portion(stateA_data)
    write_ligand_itp(stateA_ligand, pmx_dir / 'lig1.itp', mol_name='LG1')
    write_ligand_pdb(stateA_ligand, stateA_gro, pmx_dir / 'lig1.pdb', mol_name='LG1')
    print(f"    State A: {len(stateA_ligand['atoms'])} ligand atoms")

    # State B
    print("  Parsing state B...")
    stateB_data = parse_itp_full(stateB_itp)
    stateB_ligand = extract_ligand_portion(stateB_data)
    write_ligand_itp(stateB_ligand, pmx_dir / 'lig2.itp', mol_name='LG2')
    write_ligand_pdb(stateB_ligand, stateB_gro, pmx_dir / 'lig2.pdb', mol_name='LG2')
    print(f"    State B: {len(stateB_ligand['atoms'])} ligand atoms")

    # ========================================================================
    # Step 2: Run pmx atomMapping
    # ========================================================================
    print("\n[Step 2/6] Running pmx atomMapping...")

    try:
        # pmx atomMapping needs to run from pmx_dir with relative paths
        result = run_command([
            'pmx', 'atomMapping',
            '-i1', 'lig1.pdb',
            '-i2', 'lig2.pdb',
            '-o1', 'pairs1.dat',
            '-o2', 'pairs2.dat',
            '-opdb1', 'lig1_aligned.pdb',
            '-opdb2', 'lig2_aligned.pdb',
            '-log', 'mapping.log'
        ], cwd=str(pmx_dir))
        print("  atomMapping complete!")
    except Exception as e:
        print(f"  ERROR: pmx atomMapping failed: {e}")
        return 1

    # ========================================================================
    # Step 3: Run pmx ligandHybrid
    # ========================================================================
    print("\n[Step 3/6] Running pmx ligandHybrid...")

    try:
        # pmx ligandHybrid needs to run from pmx_dir with relative paths
        result = run_command([
            'pmx', 'ligandHybrid',
            '-i1', 'lig1.pdb',
            '-i2', 'lig2.pdb',
            '-itp1', 'lig1.itp',
            '-itp2', 'lig2.itp',
            '-pairs', 'pairs1.dat',
            '-oA', 'mergedA.pdb',
            '-oB', 'mergedB.pdb',
            '-oitp', 'merged.itp',
            '-offitp', 'ffmerged.itp',
            '-log', 'hybrid.log'
        ], cwd=str(pmx_dir))
        print("  ligandHybrid complete!")
    except Exception as e:
        print(f"  ERROR: pmx ligandHybrid failed: {e}")
        return 1

    # ========================================================================
    # Step 4: Copy and prepare input files
    # ========================================================================
    print("\n[Step 4/6] Preparing input files...")

    stateA_dir = Path(stateA_dir)

    # Copy structure (using state A as base)
    if (stateA_dir / 'em.gro').exists():
        shutil.copy(stateA_dir / 'em.gro', input_dir / 'hybrid.gro')
        print(f"  Copied structure from state A")

    # Copy topology
    if (stateA_dir / 'topol.top').exists():
        shutil.copy(stateA_dir / 'topol.top', input_dir / 'topol.top')
        print(f"  Copied topology")

    # Copy force field
    ff_dirs = list(stateA_dir.glob('*.ff'))
    if ff_dirs:
        ff_dir = ff_dirs[0]
        dst_ff = input_dir / ff_dir.name
        if dst_ff.exists():
            shutil.rmtree(dst_ff)
        shutil.copytree(ff_dir, dst_ff)
        print(f"  Copied force field: {ff_dir.name}")

        # Add dummy atomtypes from pmx
        if (pmx_dir / 'ffmerged.itp').exists():
            with open(pmx_dir / 'ffmerged.itp') as f:
                dummy_types = f.read()

            ffnonbonded = dst_ff / 'ffnonbonded.itp'
            if ffnonbonded.exists():
                with open(ffnonbonded, 'a') as f:
                    f.write("\n; Dummy atom types from pmx\n")
                    f.write(dummy_types)
                print(f"  Added dummy atom types to force field")

    # Copy position restraints
    for posre in stateA_dir.glob('posre*.itp'):
        shutil.copy(posre, input_dir / posre.name)

    # Copy other itp files
    for itp in stateA_dir.glob('*.itp'):
        if not (input_dir / itp.name).exists():
            shutil.copy(itp, input_dir / itp.name)

    # Copy residuetypes.dat
    if (stateA_dir / 'residuetypes.dat').exists():
        shutil.copy(stateA_dir / 'residuetypes.dat', input_dir / 'residuetypes.dat')

    # Copy hybrid topology
    shutil.copy(pmx_dir / 'merged.itp', input_dir / 'hybrid_ligand.itp')
    print(f"  Copied hybrid ligand topology")

    # ========================================================================
    # Step 5: Create lambda windows and MDP files
    # ========================================================================
    print(f"\n[Step 5/6] Creating {n_windows} lambda windows...")

    dt = 0.002  # 2 fs timestep
    prod_steps = int(prod_time_ns * 1000000 / dt)  # ns to steps

    for i in range(n_windows):
        lambda_dir = output_dir / f'lambda{i:02d}'
        lambda_dir.mkdir(exist_ok=True)

        # Write MDP files
        write_fep_mdp(lambda_dir / 'em.mdp', n_windows, i, 'em')
        write_fep_mdp(lambda_dir / 'nvt.mdp', n_windows, i, 'nvt',
                      nsteps=50000, dt=dt, ref_temp=ref_temp)
        write_fep_mdp(lambda_dir / 'npt.mdp', n_windows, i, 'npt',
                      nsteps=250000, dt=dt, ref_temp=ref_temp)
        write_fep_mdp(lambda_dir / 'prod.mdp', n_windows, i, 'prod',
                      nsteps=prod_steps, dt=dt, ref_temp=ref_temp)

        # Write run script
        write_lambda_run_script(lambda_dir, i, gmx=gmx, gpu=gpu)

        print(f"  Created lambda{i:02d}/")

    # ========================================================================
    # Step 6: Write master scripts
    # ========================================================================
    print("\n[Step 6/6] Writing run and analysis scripts...")

    write_master_run_script(output_dir, n_windows, gmx=gmx)
    write_analysis_script(output_dir, n_windows, gmx=gmx)

    # Write README
    readme = f"""# RBFE Setup for Covalent Ligand Transformation

## Overview
- State A (λ=0): {stateA_dir}
- State B (λ=1): {stateB_dir}
- Lambda windows: {n_windows}
- Production time: {prod_time_ns} ns per window
- Total simulation time: {n_windows * prod_time_ns} ns

## Directory Structure
- `pmx_input/`: pmx intermediate files (atom mapping, hybrid topology)
- `input/`: Input files for simulations (structure, topology, force field)
- `lambda00/` to `lambda{n_windows-1:02d}/`: Individual lambda window directories

## Running Simulations

### Run all windows sequentially:
```bash
./run_all.sh
```

### Run a single window:
```bash
cd lambda00
./run.sh
```

### Submit to SLURM (example):
```bash
for i in $(seq 0 {n_windows-1}); do
    sbatch --job-name=fep_$i --wrap="cd lambda$(printf '%02d' $i) && ./run.sh"
done
```

## Analysis
After all windows complete:
```bash
python analyze_fep.py
```

Or use GROMACS BAR directly:
```bash
gmx bar -f lambda*/prod.xvg -o bar.xvg
```

## Lambda Schedule
{n_windows} windows with λ = {', '.join(f'{i/(n_windows-1):.3f}' for i in range(n_windows))}
"""
    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme)

    print("\n" + "=" * 70)
    print("RBFE Setup Complete!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nTo run simulations:")
    print(f"  cd {output_dir}")
    print(f"  ./run_all.sh")
    print(f"\nOr run individual windows:")
    print(f"  cd {output_dir}/lambda00")
    print(f"  ./run.sh")

    return 0


def setup_rbfe_rtp(stateA_dir, stateB_dir, stateA_resname, stateB_resname,
                   output_dir, n_windows=14, prod_time_ns=3, ref_temp=300, gpu=False):
    """
    Set up RBFE calculation using RTP residue definitions from force field.

    This is the recommended mode as it uses the residue definitions already
    present in the force field's aminoacids.rtp files.

    Args:
        stateA_dir: Path to state A MD simulation directory
        stateB_dir: Path to state B MD simulation directory
        stateA_resname: State A residue name (e.g., 'CX1')
        stateB_resname: State B residue name (e.g., 'CX2')
        output_dir: Output directory for RBFE setup
        n_windows: Number of lambda windows (default: 14)
        prod_time_ns: Production time per window in ns (default: 3)
        ref_temp: Reference temperature in K (default: 300)
        gpu: Use GPU acceleration
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stateA_dir = Path(stateA_dir)
    stateB_dir = Path(stateB_dir)

    gmx = find_gmx()
    if gmx is None:
        print("WARNING: GROMACS not found in PATH")
        print("         Setup will continue but run scripts will need gmx in PATH")
        gmx = 'gmx'

    print("=" * 70)
    print("RBFE Setup using pmx (RTP Mode)")
    print("=" * 70)
    print(f"State A (λ=0): {stateA_dir} [{stateA_resname}]")
    print(f"State B (λ=1): {stateB_dir} [{stateB_resname}]")
    print(f"Output: {output_dir}")
    print(f"Lambda windows: {n_windows}")
    print(f"Production time: {prod_time_ns} ns per window")
    print(f"Total simulation time: {n_windows * prod_time_ns} ns")
    print()

    # Create subdirectories
    pmx_dir = output_dir / 'pmx_input'
    input_dir = output_dir / 'input'
    pmx_dir.mkdir(exist_ok=True)
    input_dir.mkdir(exist_ok=True)

    # Find force field directories
    stateA_ff = list(stateA_dir.glob(f'*{stateA_resname.lower()}*.ff'))
    stateB_ff = list(stateB_dir.glob(f'*{stateB_resname.lower()}*.ff'))

    if not stateA_ff:
        # Try general pattern
        stateA_ff = list(stateA_dir.glob('*.ff'))
    if not stateB_ff:
        stateB_ff = list(stateB_dir.glob('*.ff'))

    if not stateA_ff or not stateB_ff:
        print("ERROR: Could not find force field directories")
        print(f"  State A: {list(stateA_dir.glob('*.ff'))}")
        print(f"  State B: {list(stateB_dir.glob('*.ff'))}")
        return 1

    stateA_ff = stateA_ff[0]
    stateB_ff = stateB_ff[0]
    print(f"Using force fields: {stateA_ff.name}, {stateB_ff.name}")

    # ========================================================================
    # Step 1: Parse RTP files and extract ligand portions
    # ========================================================================
    print("\n[Step 1/6] Extracting ligand portions from RTP files...")

    stateA_rtp = stateA_ff / 'aminoacids.rtp'
    stateB_rtp = stateB_ff / 'aminoacids.rtp'

    if not stateA_rtp.exists() or not stateB_rtp.exists():
        print(f"ERROR: RTP files not found")
        print(f"  State A: {stateA_rtp}")
        print(f"  State B: {stateB_rtp}")
        return 1

    # Parse RTP files
    print(f"  Parsing {stateA_resname} from {stateA_rtp.name}...")
    stateA_rtp_data = parse_rtp_residue(stateA_rtp, stateA_resname)
    print(f"    Found {len(stateA_rtp_data['atoms'])} atoms, {len(stateA_rtp_data['bonds'])} bonds")

    print(f"  Parsing {stateB_resname} from {stateB_rtp.name}...")
    stateB_rtp_data = parse_rtp_residue(stateB_rtp, stateB_resname)
    print(f"    Found {len(stateB_rtp_data['atoms'])} atoms, {len(stateB_rtp_data['bonds'])} bonds")

    # Parse force field parameters
    print("  Loading force field parameters...")
    stateA_ff_params = parse_ffbonded(stateA_ff)
    stateB_ff_params = parse_ffbonded(stateB_ff)
    atom_masses = get_atom_masses()

    # Extract ligand portions
    print("  Extracting ligand portions...")
    stateA_ligand = extract_ligand_from_rtp(stateA_rtp_data, stateA_ff_params, atom_masses)
    stateB_ligand = extract_ligand_from_rtp(stateB_rtp_data, stateB_ff_params, atom_masses)
    print(f"    State A ({stateA_resname}): {len(stateA_ligand['atoms'])} ligand atoms")
    print(f"    State B ({stateB_resname}): {len(stateB_ligand['atoms'])} ligand atoms")

    # Write ITP files for pmx
    write_ligand_itp(stateA_ligand, pmx_dir / 'lig1.itp', mol_name='LG1')
    write_ligand_itp(stateB_ligand, pmx_dir / 'lig2.itp', mol_name='LG2')

    # Find equilibrated structures
    stateA_gro = stateA_dir / 'em.gro'
    stateB_gro = stateB_dir / 'em.gro'

    if not stateA_gro.exists():
        stateA_gro = stateA_dir / 'npt.gro'
    if not stateB_gro.exists():
        stateB_gro = stateB_dir / 'npt.gro'

    if not stateA_gro.exists() or not stateB_gro.exists():
        print(f"ERROR: Structure files not found")
        print(f"  State A: looking for em.gro or npt.gro in {stateA_dir}")
        print(f"  State B: looking for em.gro or npt.gro in {stateB_dir}")
        return 1

    # Write PDB files for pmx
    write_ligand_pdb_from_gro(stateA_ligand, stateA_gro, pmx_dir / 'lig1.pdb',
                              stateA_resname, mol_name='LG1')
    write_ligand_pdb_from_gro(stateB_ligand, stateB_gro, pmx_dir / 'lig2.pdb',
                              stateB_resname, mol_name='LG2')

    # ========================================================================
    # Step 2: Run pmx atomMapping
    # ========================================================================
    print("\n[Step 2/6] Running pmx atomMapping...")

    try:
        result = run_command([
            'pmx', 'atomMapping',
            '-i1', 'lig1.pdb',
            '-i2', 'lig2.pdb',
            '-o1', 'pairs1.dat',
            '-o2', 'pairs2.dat',
            '-opdb1', 'lig1_aligned.pdb',
            '-opdb2', 'lig2_aligned.pdb',
            '-log', 'mapping.log'
        ], cwd=str(pmx_dir))
        print("  atomMapping complete!")
    except Exception as e:
        print(f"  ERROR: pmx atomMapping failed: {e}")
        return 1

    # ========================================================================
    # Step 3: Run pmx ligandHybrid
    # ========================================================================
    print("\n[Step 3/6] Running pmx ligandHybrid...")

    try:
        result = run_command([
            'pmx', 'ligandHybrid',
            '-i1', 'lig1.pdb',
            '-i2', 'lig2.pdb',
            '-itp1', 'lig1.itp',
            '-itp2', 'lig2.itp',
            '-pairs', 'pairs1.dat',
            '-oA', 'mergedA.pdb',
            '-oB', 'mergedB.pdb',
            '-oitp', 'merged.itp',
            '-offitp', 'ffmerged.itp',
            '-log', 'hybrid.log'
        ], cwd=str(pmx_dir))
        print("  ligandHybrid complete!")
    except Exception as e:
        print(f"  ERROR: pmx ligandHybrid failed: {e}")
        return 1

    # ========================================================================
    # Step 4: Copy and prepare input files
    # ========================================================================
    print("\n[Step 4/6] Preparing input files...")

    # Copy structure (using state A as base)
    if (stateA_dir / 'em.gro').exists():
        shutil.copy(stateA_dir / 'em.gro', input_dir / 'hybrid.gro')
        print(f"  Copied structure from state A")

    # Copy topology
    if (stateA_dir / 'topol.top').exists():
        shutil.copy(stateA_dir / 'topol.top', input_dir / 'topol.top')
        print(f"  Copied topology")

    # Copy force field
    dst_ff = input_dir / stateA_ff.name
    if dst_ff.exists():
        shutil.rmtree(dst_ff)
    shutil.copytree(stateA_ff, dst_ff)
    print(f"  Copied force field: {stateA_ff.name}")

    # Add dummy atomtypes from pmx
    if (pmx_dir / 'ffmerged.itp').exists():
        with open(pmx_dir / 'ffmerged.itp') as f:
            dummy_types = f.read()

        ffnonbonded = dst_ff / 'ffnonbonded.itp'
        if ffnonbonded.exists():
            with open(ffnonbonded, 'a') as f:
                f.write("\n; Dummy atom types from pmx\n")
                f.write(dummy_types)
            print(f"  Added dummy atom types to force field")

    # Copy position restraints and other itp files
    for itp in stateA_dir.glob('*.itp'):
        if not (input_dir / itp.name).exists():
            shutil.copy(itp, input_dir / itp.name)

    # Copy residuetypes.dat
    if (stateA_dir / 'residuetypes.dat').exists():
        shutil.copy(stateA_dir / 'residuetypes.dat', input_dir / 'residuetypes.dat')

    # Copy hybrid topology
    shutil.copy(pmx_dir / 'merged.itp', input_dir / 'hybrid_ligand.itp')
    print(f"  Copied hybrid ligand topology")

    # ========================================================================
    # Step 5: Create lambda windows and MDP files
    # ========================================================================
    print(f"\n[Step 5/6] Creating {n_windows} lambda windows...")

    dt = 0.002
    prod_steps = int(prod_time_ns * 1000000 / dt)

    for i in range(n_windows):
        lambda_dir = output_dir / f'lambda{i:02d}'
        lambda_dir.mkdir(exist_ok=True)

        write_fep_mdp(lambda_dir / 'em.mdp', n_windows, i, 'em')
        write_fep_mdp(lambda_dir / 'nvt.mdp', n_windows, i, 'nvt',
                      nsteps=50000, dt=dt, ref_temp=ref_temp)
        write_fep_mdp(lambda_dir / 'npt.mdp', n_windows, i, 'npt',
                      nsteps=250000, dt=dt, ref_temp=ref_temp)
        write_fep_mdp(lambda_dir / 'prod.mdp', n_windows, i, 'prod',
                      nsteps=prod_steps, dt=dt, ref_temp=ref_temp)

        write_lambda_run_script(lambda_dir, i, gmx=gmx, gpu=gpu)
        print(f"  Created lambda{i:02d}/")

    # ========================================================================
    # Step 6: Write master scripts
    # ========================================================================
    print("\n[Step 6/6] Writing run and analysis scripts...")

    write_master_run_script(output_dir, n_windows, gmx=gmx)
    write_analysis_script(output_dir, n_windows, gmx=gmx)

    # Write README
    readme = f"""# RBFE Setup for Covalent Ligand Transformation (RTP Mode)

## Overview
- State A (λ=0): {stateA_dir} [{stateA_resname}]
- State B (λ=1): {stateB_dir} [{stateB_resname}]
- Lambda windows: {n_windows}
- Production time: {prod_time_ns} ns per window
- Total simulation time: {n_windows * prod_time_ns} ns

## Source
Ligand topologies extracted from force field RTP files:
- State A: {stateA_ff}/aminoacids.rtp [{stateA_resname}]
- State B: {stateB_ff}/aminoacids.rtp [{stateB_resname}]

## Directory Structure
- `pmx_input/`: pmx intermediate files (atom mapping, hybrid topology)
- `input/`: Input files for simulations (structure, topology, force field)
- `lambda00/` to `lambda{n_windows-1:02d}/`: Individual lambda window directories

## Running Simulations

### Run all windows sequentially:
```bash
./run_all.sh
```

### Run a single window:
```bash
cd lambda00
./run.sh
```

### Submit to SLURM (example):
```bash
for i in $(seq 0 {n_windows-1}); do
    sbatch --job-name=fep_$i --wrap="cd lambda$(printf '%02d' $i) && ./run.sh"
done
```

## Analysis
After all windows complete:
```bash
python analyze_fep.py
```

Or use GROMACS BAR directly:
```bash
gmx bar -f lambda*/prod.xvg -o bar.xvg
```

## Lambda Schedule
{n_windows} windows with λ = {', '.join(f'{i/(n_windows-1):.3f}' for i in range(n_windows))}
"""
    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme)

    print("\n" + "=" * 70)
    print("RBFE Setup Complete! (RTP Mode)")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nTo run simulations:")
    print(f"  cd {output_dir}")
    print(f"  ./run_all.sh")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Set up RBFE calculations using pmx for covalent ligand transformation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Two modes are supported:

1. RTP Mode (recommended):
   Uses residue definitions from force field RTP files.
   Requires: --stateA_resname, --stateB_resname

   Example:
     python setup_rbfe_pmx.py \\
         --stateA_dir Outputs/Covalent/Inhib_32_acry/md_simulation \\
         --stateB_dir Outputs/Covalent/Inhib_32_chlo/md_simulation \\
         --stateA_resname CX1 \\
         --stateB_resname CX2 \\
         --output_dir Outputs/Covalent/RBFE_CX1_to_CX2

2. ITP Mode (legacy):
   Uses standalone ACPYPE ITP files.
   Requires: --stateA_itp, --stateB_itp, --stateA_gro, --stateB_gro

   Example:
     python setup_rbfe_pmx.py \\
         --stateA_dir Outputs/Covalent/Inhib_32_acry/md_simulation \\
         --stateB_dir Outputs/Covalent/Inhib_32_chlo/md_simulation \\
         --stateA_itp path/to/adduct_merged_GMX.itp \\
         --stateB_itp path/to/adduct_merged_GMX.itp \\
         --stateA_gro path/to/adduct_merged_GMX.gro \\
         --stateB_gro path/to/adduct_merged_GMX.gro \\
         --output_dir Outputs/Covalent/RBFE_CX1_to_CX2
        """
    )

    # Required arguments
    parser.add_argument('--stateA_dir', required=True,
                        help='Path to state A MD simulation directory')
    parser.add_argument('--stateB_dir', required=True,
                        help='Path to state B MD simulation directory')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for RBFE setup')

    # RTP mode arguments
    rtp_group = parser.add_argument_group('RTP Mode (recommended)')
    rtp_group.add_argument('--stateA_resname',
                           help='State A residue name (e.g., CX1)')
    rtp_group.add_argument('--stateB_resname',
                           help='State B residue name (e.g., CX2)')

    # ITP mode arguments
    itp_group = parser.add_argument_group('ITP Mode (legacy)')
    itp_group.add_argument('--stateA_itp',
                           help='Path to state A ACPYPE ITP file')
    itp_group.add_argument('--stateB_itp',
                           help='Path to state B ACPYPE ITP file')
    itp_group.add_argument('--stateA_gro',
                           help='Path to state A ACPYPE GRO file')
    itp_group.add_argument('--stateB_gro',
                           help='Path to state B ACPYPE GRO file')

    # Common arguments
    parser.add_argument('--n_windows', type=int, default=14,
                        help='Number of lambda windows (default: 14)')
    parser.add_argument('--prod_time', type=float, default=3,
                        help='Production time per window in ns (default: 3)')
    parser.add_argument('--temp', type=float, default=300,
                        help='Reference temperature in K (default: 300)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU acceleration')

    args = parser.parse_args()

    # Determine mode
    rtp_mode = args.stateA_resname and args.stateB_resname
    itp_mode = args.stateA_itp and args.stateB_itp and args.stateA_gro and args.stateB_gro

    if rtp_mode:
        print("Using RTP mode (extracting topology from force field)")
        return setup_rbfe_rtp(
            stateA_dir=args.stateA_dir,
            stateB_dir=args.stateB_dir,
            stateA_resname=args.stateA_resname,
            stateB_resname=args.stateB_resname,
            output_dir=args.output_dir,
            n_windows=args.n_windows,
            prod_time_ns=args.prod_time,
            ref_temp=args.temp,
            gpu=args.gpu
        )
    elif itp_mode:
        print("Using ITP mode (using ACPYPE ITP files)")
        return setup_rbfe(
            stateA_dir=args.stateA_dir,
            stateB_dir=args.stateB_dir,
            stateA_itp=args.stateA_itp,
            stateB_itp=args.stateB_itp,
            stateA_gro=args.stateA_gro,
            stateB_gro=args.stateB_gro,
            output_dir=args.output_dir,
            n_windows=args.n_windows,
            prod_time_ns=args.prod_time,
            ref_temp=args.temp,
            gpu=args.gpu
        )
    else:
        print("ERROR: Must specify either RTP mode or ITP mode arguments")
        print()
        print("RTP Mode (recommended):")
        print("  --stateA_resname CX1 --stateB_resname CX2")
        print()
        print("ITP Mode (legacy):")
        print("  --stateA_itp <path> --stateB_itp <path> --stateA_gro <path> --stateB_gro <path>")
        return 1


if __name__ == '__main__':
    sys.exit(main())
