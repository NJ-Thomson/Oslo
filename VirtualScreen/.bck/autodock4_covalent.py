#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Covalent Docking with AutoDock4 - CHARMM-GUI Style (v2)

Fixed version with proper AutoDock4 atom types.
"""

import argparse
import subprocess
import os
import sys
import tempfile
from pathlib import Path
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D
except ImportError:
    print("ERROR: RDKit required")
    sys.exit(1)


# AutoDock4 atom type mapping
AD4_ATOM_TYPES = {
    ('C', False, False): 'C',   # Aliphatic carbon
    ('C', True, False): 'A',    # Aromatic carbon
    ('N', False, False): 'N',   # Nitrogen (not acceptor)
    ('N', False, True): 'NA',   # Nitrogen acceptor
    ('O', False, True): 'OA',   # Oxygen acceptor
    ('S', False, True): 'SA',   # Sulfur acceptor
    ('H', False, False): 'H',   # Hydrogen (non-polar)
    ('H', True, False): 'HD',   # Hydrogen donor (on N or O)
    ('F', False, False): 'F',
    ('Cl', False, False): 'Cl',
    ('Br', False, False): 'Br',
    ('I', False, False): 'I',
    ('P', False, False): 'P',
}


def get_ad4_atom_type(atom, mol):
    """Determine AutoDock4 atom type for an atom."""
    symbol = atom.GetSymbol()
    is_aromatic = atom.GetIsAromatic()
    
    # Check if acceptor (has lone pairs)
    is_acceptor = symbol in ('O', 'N', 'S') and not atom.GetFormalCharge() > 0
    
    # Special case for H - check if donor (attached to N or O)
    if symbol == 'H':
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() in ('N', 'O'):
                return 'HD'
        return 'H'
    
    # Carbon - aromatic vs aliphatic
    if symbol == 'C':
        return 'A' if is_aromatic else 'C'
    
    # Nitrogen
    if symbol == 'N':
        # NA if acceptor (has lone pair available)
        if is_acceptor and atom.GetTotalNumHs() < 3:
            return 'NA'
        return 'N'
    
    # Oxygen - almost always acceptor
    if symbol == 'O':
        return 'OA'
    
    # Sulfur
    if symbol == 'S':
        return 'SA'
    
    # Halogens and others
    return symbol if len(symbol) <= 2 else symbol[:2]


def mol_to_pdbqt(mol, output_file, is_ligand=True):
    """Convert RDKit mol to PDBQT with proper AD4 atom types and Gasteiger charges."""
    
    # Compute Gasteiger charges
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except:
        pass
    
    conf = mol.GetConformer()
    
    lines = []
    if is_ligand:
        lines.append("ROOT\n")
    
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        symbol = atom.GetSymbol()
        ad4_type = get_ad4_atom_type(atom, mol)
        
        # Get Gasteiger charge
        try:
            charge = float(atom.GetProp('_GasteigerCharge'))
            if np.isnan(charge) or np.isinf(charge):
                charge = 0.0
        except:
            charge = 0.0
        
        name = f"{symbol}{i+1}"[:4].ljust(4)
        
        record = "HETATM" if is_ligand else "ATOM  "
        resname = "LIG" if is_ligand else "RES"
        
        line = (
            f"{record}{i+1:5d} {name} {resname} A   1    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
            f"  1.00  0.00    {charge:+.3f} {ad4_type:<2s}\n"
        )
        lines.append(line)
    
    if is_ligand:
        lines.append("ENDROOT\n")
        lines.append("TORSDOF 0\n")
    
    with open(output_file, 'w') as f:
        f.writelines(lines)
    
    return output_file


def pdb_to_pdbqt_receptor(pdb_file, output_file, cys_chain=None, cys_resid=None):
    """Convert receptor PDB to PDBQT with AD4 atom types and Gasteiger-like charges."""
    lines = []
    
    # Simple charge assignment based on atom type
    CHARGES = {
        'N': -0.350,   # Backbone N
        'CA': 0.100,   # Alpha carbon
        'C': 0.550,    # Carbonyl carbon
        'O': -0.550,   # Carbonyl oxygen
        'CB': 0.000,   # Beta carbon
        'CG': 0.000,
        'CD': 0.000,
        'CE': 0.000,
        'CZ': 0.000,
        'OG': -0.490,  # Ser/Thr hydroxyl
        'OD': -0.550,  # Asp carboxyl
        'OE': -0.550,  # Glu carboxyl
        'OH': -0.490,  # Tyr hydroxyl
        'ND': -0.350,  # His nitrogen
        'NE': -0.350,  # Arg/His nitrogen
        'NZ': -0.350,  # Lys nitrogen
        'SG': -0.230,  # Cys sulfur
        'SD': -0.230,  # Met sulfur
        'H': 0.250,    # Amide hydrogen
        'HN': 0.250,
    }
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            
            # Skip waters
            res_name = line[17:20].strip()
            if res_name in ('HOH', 'WAT'):
                continue
            
            # Skip Cys HG if specified
            if cys_chain and cys_resid:
                chain_id = line[21]
                try:
                    res_num = int(line[22:26].strip())
                except:
                    res_num = 0
                atom_name = line[12:16].strip()
                
                if (chain_id == cys_chain and res_num == cys_resid and 
                    res_name in ('CYS', 'CYM') and atom_name.startswith('HG')):
                    continue
            
            # Get atom name and determine element
            atom_name = line[12:16].strip()
            
            # Get element from atom name
            if atom_name[0].isdigit():
                element = atom_name[1] if len(atom_name) > 1 else 'H'
            else:
                element = atom_name[0]
            
            # Determine AD4 type
            if element == 'C':
                ad4_type = 'C'
            elif element == 'N':
                # NA for acceptor nitrogens (backbone N, His, etc.)
                if atom_name in ('N',) or atom_name.startswith(('ND', 'NE')):
                    ad4_type = 'NA'
                else:
                    ad4_type = 'N'
            elif element == 'O':
                ad4_type = 'OA'
            elif element == 'S':
                ad4_type = 'SA'
            elif element == 'H':
                # HD for polar hydrogens (on N or O)
                if atom_name.startswith(('H', 'HN', 'HE', 'HZ', 'HD', 'HH', 'HG1', 'HO')):
                    ad4_type = 'HD'
                else:
                    ad4_type = 'H'
            else:
                ad4_type = element[:2] if len(element) >= 2 else element
            
            # Get charge
            charge = CHARGES.get(atom_name, 0.0)
            if charge == 0.0:
                # Try first two characters
                charge = CHARGES.get(atom_name[:2], 0.0)
            if charge == 0.0:
                # Default by element
                if element == 'O':
                    charge = -0.400
                elif element == 'N':
                    charge = -0.200
                elif element == 'S':
                    charge = -0.200
                elif element == 'H':
                    charge = 0.150
            
            # Build PDBQT line with proper formatting
            # Format: columns must be exact for AutoDock4
            coords = line[30:54]  # X, Y, Z coordinates
            
            # PDBQT format: ...coords... occupancy bfactor charge atomtype
            new_line = (
                f"{line[:54]}"  # Record through coordinates
                f"  1.00  0.00    "  # Occupancy, B-factor, spaces
                f"{charge:+.3f}"  # Partial charge with sign
                f" {ad4_type:<2s}\n"  # Atom type
            )
            lines.append(new_line)
    
    with open(output_file, 'w') as f:
        f.writelines(lines)
    
    return output_file


def get_cys_geometry(pdb_file, chain, resid):
    """Get Cys atom coordinates."""
    coords = {}
    with open(pdb_file, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            res_name = line[17:20].strip()
            if res_name not in ('CYS', 'CYM', 'CYX'):
                continue
            try:
                res_num = int(line[22:26].strip())
                chain_id = line[21]
            except:
                continue
            
            if chain_id == chain and res_num == resid:
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords[atom_name] = np.array([x, y, z])
    
    return coords if 'SG' in coords else None


def create_covalent_adduct(mol, warhead_type='acrylamide'):
    """Create Michael addition adduct with -SCH3."""
    
    if warhead_type == 'acrylamide':
        pattern = Chem.MolFromSmarts('[CX3](=O)[CX3]=[CX3]')
    else:
        pattern = Chem.MolFromSmarts('[CX4][Cl]')
    
    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        raise ValueError("Warhead not found")
    
    match = matches[0]
    rw = Chem.RWMol(Chem.Mol(mol))
    
    if warhead_type == 'acrylamide':
        # Find the C=C double bond
        alpha_idx = None
        beta_idx = None
        for i, idx in enumerate(match):
            atom = rw.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() != 6:
                continue
            for nb in atom.GetNeighbors():
                if nb.GetIdx() in match and nb.GetAtomicNum() == 6:
                    bond = rw.GetBondBetweenAtoms(idx, nb.GetIdx())
                    if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                        # Check which is connected to carbonyl
                        for nb2 in atom.GetNeighbors():
                            if nb2.GetAtomicNum() == 8:
                                alpha_idx = nb.GetIdx()
                                beta_idx = idx
                                break
                        if alpha_idx is None:
                            alpha_idx = idx
                            beta_idx = nb.GetIdx()
                        break
            if alpha_idx:
                break
        
        if alpha_idx is None:
            alpha_idx = match[-2]
            beta_idx = match[-1]
        
        # Convert C=C to C-C
        bond = rw.GetBondBetweenAtoms(alpha_idx, beta_idx)
        if bond:
            bond.SetBondType(Chem.BondType.SINGLE)
        
        # Add S-CH3
        s_idx = rw.AddAtom(Chem.Atom(16))
        rw.AddBond(beta_idx, s_idx, Chem.BondType.SINGLE)
        c_idx = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(s_idx, c_idx, Chem.BondType.SINGLE)
        
        anchor_idx = s_idx
    
    else:  # chloroacetamide
        cl_idx = None
        c_idx = None
        for idx in match:
            atom = rw.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() == 17:
                cl_idx = idx
                for nb in atom.GetNeighbors():
                    if nb.GetAtomicNum() == 6:
                        c_idx = nb.GetIdx()
                        break
                break
        
        rw.RemoveAtom(cl_idx)
        if cl_idx < c_idx:
            c_idx -= 1
        
        s_idx = rw.AddAtom(Chem.Atom(16))
        rw.AddBond(c_idx, s_idx, Chem.BondType.SINGLE)
        ch3_idx = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(s_idx, ch3_idx, Chem.BondType.SINGLE)
        
        anchor_idx = s_idx
    
    adduct = rw.GetMol()
    Chem.SanitizeMol(adduct)
    adduct = Chem.AddHs(adduct)
    
    return adduct, anchor_idx


def position_adduct(adduct, anchor_idx, sg_pos, cb_pos, receptor_pdb=None):
    """Generate 3D and position S at SG location, with optional clash avoidance."""
    
    # Generate conformers
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(adduct, params)
    AllChem.MMFFOptimizeMolecule(adduct, maxIters=500)
    
    # Translate S to SG position
    conf = adduct.GetConformer()
    s_pos = np.array(conf.GetAtomPosition(anchor_idx))
    translation = sg_pos - s_pos
    
    for i in range(adduct.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(pos.x + translation[0],
                                        pos.y + translation[1],
                                        pos.z + translation[2]))
    
    # If receptor provided, rotate to minimize clashes
    if receptor_pdb:
        adduct = optimize_orientation(adduct, anchor_idx, sg_pos, cb_pos, receptor_pdb)
    
    return adduct


def get_receptor_coords(pdb_file):
    """Extract heavy atom coordinates from receptor."""
    coords = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            atom_name = line[12:16].strip()
            if atom_name[0] == 'H':
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append(np.array([x, y, z]))
    return np.array(coords)


def count_clashes(ligand_coords, receptor_coords, clash_dist=1.5):
    """Count steric clashes between ligand and receptor."""
    clashes = 0
    for lig_coord in ligand_coords:
        dists = np.linalg.norm(receptor_coords - lig_coord, axis=1)
        clashes += np.sum(dists < clash_dist)
    return clashes


def rotate_around_axis(mol, axis_start, axis_end, angle_deg, conf_id=0):
    """Rotate molecule around an axis."""
    conf = mol.GetConformer(conf_id)
    
    # Normalize axis
    axis = axis_end - axis_start
    axis = axis / np.linalg.norm(axis)
    
    # Rotation matrix (Rodrigues)
    angle = np.radians(angle_deg)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    # Rotate all atoms around axis_start
    for i in range(mol.GetNumAtoms()):
        pos = np.array(conf.GetAtomPosition(i))
        pos_shifted = pos - axis_start
        pos_rotated = np.dot(R, pos_shifted) + axis_start
        conf.SetAtomPosition(i, Point3D(*pos_rotated))
    
    return mol


def optimize_orientation(adduct, anchor_idx, sg_pos, cb_pos, receptor_pdb):
    """Rotate adduct around S-C bond to minimize clashes with receptor."""
    
    receptor_coords = get_receptor_coords(receptor_pdb)
    
    # Remove receptor atoms very close to SG (the Cys itself)
    dists = np.linalg.norm(receptor_coords - sg_pos, axis=1)
    receptor_coords = receptor_coords[dists > 3.0]
    
    # Find the C bonded to S (not the methyl)
    s_atom = adduct.GetAtomWithIdx(anchor_idx)
    c_bonded_idx = None
    for nb in s_atom.GetNeighbors():
        if nb.GetSymbol() == 'C' and nb.GetDegree() > 1:
            c_bonded_idx = nb.GetIdx()
            break
    
    if c_bonded_idx is None:
        print("    WARNING: Could not find rotation axis")
        return adduct
    
    conf = adduct.GetConformer()
    axis_start = np.array(conf.GetAtomPosition(anchor_idx))  # S
    axis_end = np.array(conf.GetAtomPosition(c_bonded_idx))  # C
    
    # Try different rotation angles
    best_angle = 0
    best_clashes = float('inf')
    best_mol = Chem.Mol(adduct)
    
    print("    Optimizing orientation (rotating around S-C bond)...")
    
    for angle in range(0, 360, 10):
        test_mol = Chem.Mol(adduct)
        test_mol = rotate_around_axis(test_mol, axis_start, axis_end, angle)
        
        # Get ligand heavy atom coords
        test_conf = test_mol.GetConformer()
        lig_coords = []
        for i, atom in enumerate(test_mol.GetAtoms()):
            if atom.GetSymbol() != 'H':
                lig_coords.append(np.array(test_conf.GetAtomPosition(i)))
        lig_coords = np.array(lig_coords)
        
        clashes = count_clashes(lig_coords, receptor_coords, clash_dist=1.8)
        
        if clashes < best_clashes:
            best_clashes = clashes
            best_angle = angle
            best_mol = Chem.Mol(test_mol)
    
    print(f"    Best rotation: {best_angle} deg with {best_clashes} clashes")
    
    # If still clashing, try more aggressive sampling with multiple torsions
    if best_clashes > 5:
        print("    Trying conformer sampling...")
        best_mol, best_clashes = sample_conformers_with_rotation(
            adduct, anchor_idx, receptor_coords, sg_pos
        )
        print(f"    After conformer sampling: {best_clashes} clashes")
    
    return best_mol


def sample_conformers_with_rotation(adduct, anchor_idx, receptor_coords, sg_pos):
    """Generate multiple conformers and test each with rotations."""
    
    # Generate conformers
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    
    mol_h = Chem.Mol(adduct)
    cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=50, params=params)
    
    best_mol = Chem.Mol(adduct)
    best_clashes = float('inf')
    
    # Find rotation axis atoms
    s_atom = mol_h.GetAtomWithIdx(anchor_idx)
    c_bonded_idx = None
    for nb in s_atom.GetNeighbors():
        if nb.GetSymbol() == 'C' and nb.GetDegree() > 1:
            c_bonded_idx = nb.GetIdx()
            break
    
    for cid in cids:
        # Translate S to SG position
        conf = mol_h.GetConformer(cid)
        s_pos = np.array(conf.GetAtomPosition(anchor_idx))
        translation = sg_pos - s_pos
        
        for i in range(mol_h.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, Point3D(pos.x + translation[0],
                                            pos.y + translation[1],
                                            pos.z + translation[2]))
        
        # Create single-conformer mol for rotation
        test_mol = Chem.Mol(mol_h)
        test_conf = test_mol.GetConformer()
        for i in range(test_mol.GetNumAtoms()):
            test_conf.SetAtomPosition(i, conf.GetAtomPosition(i))
        
        axis_start = np.array(test_conf.GetAtomPosition(anchor_idx))
        axis_end = np.array(test_conf.GetAtomPosition(c_bonded_idx)) if c_bonded_idx else axis_start + np.array([1,0,0])
        
        # Test rotations
        for angle in range(0, 360, 30):
            rot_mol = Chem.Mol(test_mol)
            rot_mol = rotate_around_axis(rot_mol, axis_start, axis_end, angle)
            
            rot_conf = rot_mol.GetConformer()
            lig_coords = []
            for i, atom in enumerate(rot_mol.GetAtoms()):
                if atom.GetSymbol() != 'H':
                    lig_coords.append(np.array(rot_conf.GetAtomPosition(i)))
            lig_coords = np.array(lig_coords)
            
            clashes = count_clashes(lig_coords, receptor_coords, clash_dist=1.8)
            
            if clashes < best_clashes:
                best_clashes = clashes
                best_mol = Chem.Mol(rot_mol)
                
                if best_clashes == 0:
                    return best_mol, best_clashes
    
    return best_mol, best_clashes


def create_gpf(receptor_pdbqt, ligand_pdbqt, center, box_size, output_gpf, output_dir):
    """Create Grid Parameter File."""
    
    # Get ligand atom types
    atom_types = set()
    with open(ligand_pdbqt, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                parts = line.split()
                if len(parts) >= 12:
                    atype = parts[-1].strip()
                    if atype and len(atype) <= 2:
                        atom_types.add(atype)
    
    # Ensure common types are present
    atom_types.update(['A', 'C', 'OA', 'N', 'HD', 'SA'])
    
    npts = int(box_size / 0.375)
    if npts % 2 == 1:
        npts += 1
    
    receptor_name = os.path.basename(receptor_pdbqt)
    
    content = f"""npts {npts} {npts} {npts}
gridfld {receptor_name.replace('.pdbqt', '.maps.fld')}
spacing 0.375
receptor_types A C N NA OA SA HD H
ligand_types {' '.join(sorted(atom_types))}
receptor {receptor_name}
gridcenter {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}
smooth 0.5
"""
    
    # Add map lines for each type
    for atype in sorted(atom_types):
        content += f"map {receptor_name.replace('.pdbqt', f'.{atype}.map')}\n"
    
    content += f"""elecmap {receptor_name.replace('.pdbqt', '.e.map')}
dsolvmap {receptor_name.replace('.pdbqt', '.d.map')}
dielectric -0.1465
"""
    
    gpf_path = os.path.join(output_dir, output_gpf)
    with open(gpf_path, 'w') as f:
        f.write(content)
    
    return gpf_path


def create_dpf(receptor_pdbqt, ligand_pdbqt, output_dpf, output_dir, ga_runs=20):
    """Create Docking Parameter File."""
    
    receptor_name = os.path.basename(receptor_pdbqt)
    ligand_name = os.path.basename(ligand_pdbqt)
    
    content = f"""autodock_parameter_version 4.2
outlev 1
intelec
seed pid time
ligand_types A C N NA OA SA HD H
fld {receptor_name.replace('.pdbqt', '.maps.fld')}
map {receptor_name.replace('.pdbqt', '.A.map')}
map {receptor_name.replace('.pdbqt', '.C.map')}
map {receptor_name.replace('.pdbqt', '.N.map')}
map {receptor_name.replace('.pdbqt', '.NA.map')}
map {receptor_name.replace('.pdbqt', '.OA.map')}
map {receptor_name.replace('.pdbqt', '.SA.map')}
map {receptor_name.replace('.pdbqt', '.HD.map')}
map {receptor_name.replace('.pdbqt', '.H.map')}
elecmap {receptor_name.replace('.pdbqt', '.e.map')}
desolvmap {receptor_name.replace('.pdbqt', '.d.map')}
move {ligand_name}
about 0.0 0.0 0.0
tran0 random
axisangle0 random
dihe0 random
torsdof 0
rmstol 2.0
extnrg 1000.0
e0max 0.0 10000
ga_pop_size 150
ga_num_evals 250000
ga_num_generations 27000
ga_elitism 1
ga_mutation_rate 0.02
ga_crossover_rate 0.8
ga_window_size 10
set_ga
sw_max_its 300
sw_max_succ 4
sw_max_fail 4
sw_rho 1.0
sw_lb_rho 0.01
ls_search_freq 0.06
set_psw1
unbound_model bound
ga_run {ga_runs}
analysis
"""
    
    dpf_path = os.path.join(output_dir, output_dpf)
    with open(dpf_path, 'w') as f:
        f.write(content)
    
    return dpf_path


def run_autodock(output_dir):
    """Run autogrid4 and autodock4."""
    
    # Find executables
    autogrid = None
    autodock = None
    
    for exe, var in [('autogrid4', 'autogrid'), ('autodock4', 'autodock')]:
        result = subprocess.run(['which', exe], capture_output=True, text=True)
        if result.returncode == 0:
            if var == 'autogrid':
                autogrid = result.stdout.strip()
            else:
                autodock = result.stdout.strip()
    
    if not autogrid or not autodock:
        print("\n  AutoDock4 not found in PATH")
        print("  Run manually:")
        print(f"    cd {output_dir}")
        print("    autogrid4 -p receptor.gpf -l autogrid.log")
        print("    autodock4 -p receptor.dpf -l autodock.dlg")
        return False
    
    # Run autogrid
    print("\n  Running autogrid4...")
    result = subprocess.run(
        [autogrid, '-p', 'receptor.gpf', '-l', 'autogrid.log'],
        cwd=output_dir, capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print(f"  ERROR: autogrid4 failed")
        print(result.stderr)
        return False
    print("  OK - Grid generated")
    
    # Run autodock
    print("  Running autodock4...")
    result = subprocess.run(
        [autodock, '-p', 'receptor.dpf', '-l', 'autodock.dlg'],
        cwd=output_dir, capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print(f"  ERROR: autodock4 failed")
        print(result.stderr)
        return False
    print("  OK - Docking complete")
    
    return True


def write_complex(receptor_pdb, adduct, output_pdb, cys_chain, cys_resid, anchor_idx):
    """Write complex PDB with CONECT records."""
    
    lines = []
    sg_serial = None
    last_serial = 0
    
    with open(receptor_pdb, 'r') as f:
        for line in f:
            if line.startswith('END'):
                continue
            if line.startswith('ATOM'):
                serial = int(line[6:11].strip())
                last_serial = max(last_serial, serial)
                
                chain_id = line[21]
                try:
                    res_num = int(line[22:26].strip())
                except:
                    res_num = 0
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                
                if (chain_id == cys_chain and res_num == cys_resid and 
                    res_name in ('CYS', 'CYM') and atom_name == 'SG'):
                    sg_serial = serial
            
            lines.append(line)
    
    lines.append("TER\n")
    
    # Add ligand
    conf = adduct.GetConformer()
    idx_to_serial = {}
    
    for i, atom in enumerate(adduct.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        symbol = atom.GetSymbol()
        serial = last_serial + 1 + i
        idx_to_serial[i] = serial
        
        name = f"{symbol}{i+1}"[:4].ljust(4)
        line = f"HETATM{serial:5d} {name} LIG L   1    " \
               f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}" \
               f"  1.00  0.00          {symbol:>2s}\n"
        lines.append(line)
    
    # CONECT - find C bonded to S (anchor)
    reactive_serial = None
    s_atom = adduct.GetAtomWithIdx(anchor_idx)
    for nb in s_atom.GetNeighbors():
        if nb.GetSymbol() == 'C' and nb.GetDegree() > 1:  # Not the methyl
            reactive_serial = idx_to_serial[nb.GetIdx()]
            break
    
    if sg_serial and reactive_serial:
        lines.append(f"CONECT{sg_serial:5d}{reactive_serial:5d}\n")
        lines.append(f"CONECT{reactive_serial:5d}{sg_serial:5d}\n")
    
    lines.append("END\n")
    
    with open(output_pdb, 'w') as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(description="AutoDock4 Covalent Docking (v2)")
    parser.add_argument('--ligand', '-l', required=True)
    parser.add_argument('--receptor', '-r', required=True)
    parser.add_argument('--cys_chain', default='A')
    parser.add_argument('--cys_resid', type=int, required=True)
    parser.add_argument('--warhead', '-w', default='acrylamide')
    parser.add_argument('--outdir', '-o', default='ad4_covalent')
    parser.add_argument('--box_size', type=float, default=22.0)
    parser.add_argument('--ga_runs', type=int, default=20)
    parser.add_argument('--skip_docking', action='store_true', help='Only prepare files')
    
    args = parser.parse_args()
    
    print("="*60)
    print("AutoDock4 Covalent Docking - Adduct Approach (v2)")
    print("="*60)
    
    # Setup output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {outdir.absolute()}")
    
    # Get Cys geometry
    print("\n[1] Loading cysteine...")
    cys_coords = get_cys_geometry(args.receptor, args.cys_chain, args.cys_resid)
    if not cys_coords:
        print(f"ERROR: Cys {args.cys_chain}:{args.cys_resid} not found")
        sys.exit(1)
    sg_pos = cys_coords['SG']
    cb_pos = cys_coords.get('CB', sg_pos + np.array([1.5, 0, 0]))
    print(f"    SG: ({sg_pos[0]:.2f}, {sg_pos[1]:.2f}, {sg_pos[2]:.2f})")
    
    # Load ligand
    print("\n[2] Loading ligand...")
    suppl = Chem.SDMolSupplier(args.ligand, removeHs=False)
    mol = next(suppl, None)
    if not mol:
        print("ERROR: Could not load ligand")
        sys.exit(1)
    mol = Chem.AddHs(mol)
    print(f"    Atoms: {mol.GetNumAtoms()}")
    
    # Create adduct
    print("\n[3] Creating covalent adduct...")
    adduct, anchor_idx = create_covalent_adduct(mol, args.warhead)
    print(f"    Adduct atoms: {adduct.GetNumAtoms()}")
    print(f"    S anchor index: {anchor_idx}")
    
    # Position adduct
    print("\n[4] Positioning adduct at Cys (with clash avoidance)...")
    adduct = position_adduct(adduct, anchor_idx, sg_pos, cb_pos, args.receptor)
    
    # Check S position
    conf = adduct.GetConformer()
    s_final = np.array(conf.GetAtomPosition(anchor_idx))
    print(f"    S position: ({s_final[0]:.2f}, {s_final[1]:.2f}, {s_final[2]:.2f})")
    print(f"    Distance S to SG: {np.linalg.norm(s_final - sg_pos):.3f} A")
    
    # Save adduct
    adduct_sdf = str(outdir / "adduct.sdf")
    writer = Chem.SDWriter(adduct_sdf)
    writer.write(adduct)
    writer.close()
    print(f"    Saved: {adduct_sdf}")
    
    # Write complex PDB (for visualization)
    complex_pdb = str(outdir / "complex_preopt.pdb")
    write_complex(args.receptor, adduct, complex_pdb, args.cys_chain, args.cys_resid, anchor_idx)
    print(f"    Complex: {complex_pdb}")
    
    # Prepare PDBQT files
    print("\n[5] Preparing AutoDock4 files...")
    
    receptor_pdbqt = str(outdir / "receptor.pdbqt")
    pdb_to_pdbqt_receptor(args.receptor, receptor_pdbqt, args.cys_chain, args.cys_resid)
    print(f"    Receptor: {receptor_pdbqt}")
    
    ligand_pdbqt = str(outdir / "ligand.pdbqt")
    mol_to_pdbqt(adduct, ligand_pdbqt)
    print(f"    Ligand: {ligand_pdbqt}")
    
    # Create parameter files
    print("\n[6] Creating parameter files...")
    gpf = create_gpf(receptor_pdbqt, ligand_pdbqt, sg_pos, args.box_size, 'receptor.gpf', str(outdir))
    dpf = create_dpf(receptor_pdbqt, ligand_pdbqt, 'receptor.dpf', str(outdir), args.ga_runs)
    print(f"    GPF: {gpf}")
    print(f"    DPF: {dpf}")
    
    # Run docking
    if not args.skip_docking:
        print("\n[7] Running AutoDock4...")
        run_autodock(str(outdir))
    
    # Summary
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"\nFiles in: {outdir.absolute()}")
    print(f"\nVisualize pre-positioned complex:")
    print(f"  pymol {args.receptor} {complex_pdb}")
    print(f"\nThe S atom in the adduct is positioned at the Cys SG location.")
    print(f"This ensures correct covalent bond geometry.")


if __name__ == "__main__":
    main()