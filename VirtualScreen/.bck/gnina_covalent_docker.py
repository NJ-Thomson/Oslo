#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Covalent Docking with GNINA (Docker version)

Uses GNINA via Docker container for covalent docking.

Usage:
    python gnina_covalent_docker.py \
        --ligand ligand.sdf \
        --receptor receptor.pdb \
        --cys_chain D --cys_resid 1039 \
        --output gnina_covalent

Requirements:
    - Docker with gnina/gnina image
    - docker pull gnina/gnina
"""

import argparse
import subprocess
import os
import sys
from pathlib import Path
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D
except ImportError:
    print("ERROR: RDKit required")
    sys.exit(1)


def check_docker_gnina():
    """Check if Docker and GNINA image are available."""
    # Check Docker
    result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: Docker not found")
        print("Install with: sudo apt install docker.io")
        return False
    print(f"  Docker: {result.stdout.strip()}")
    
    # Check GNINA image
    result = subprocess.run(['docker', 'images', 'gnina/gnina', '-q'], 
                          capture_output=True, text=True)
    if not result.stdout.strip():
        print("  GNINA image not found, pulling...")
        subprocess.run(['docker', 'pull', 'gnina/gnina'])
    else:
        print("  GNINA image: OK")
    
    return True


def run_gnina_docker(receptor_pdb, ligand_sdf, output_sdf, center, 
                     box_size=20, exhaustiveness=32, num_modes=20,
                     cnn_scoring='rescore', autobox_ligand=None,
                     minimize=False, score_only=False):
    """
    Run GNINA docking via Docker.
    
    All paths must be absolute or relative to current working directory.
    """
    workdir = os.getcwd()
    
    # Convert to absolute paths
    receptor_abs = os.path.abspath(receptor_pdb)
    ligand_abs = os.path.abspath(ligand_sdf)
    output_abs = os.path.abspath(output_sdf)
    
    # Find common parent directory to mount
    common_path = os.path.commonpath([receptor_abs, ligand_abs, output_abs])
    
    # Convert to container paths
    receptor_container = '/data' + receptor_abs[len(common_path):]
    ligand_container = '/data' + ligand_abs[len(common_path):]
    output_container = '/data' + output_abs[len(common_path):]
    
    # Build command
    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{common_path}:/data',
        'gnina/gnina',
        'gnina',
        '-r', receptor_container,
        '-l', ligand_container,
        '-o', output_container,
    ]
    
    if score_only:
        cmd.append('--score_only')
    elif minimize:
        cmd.extend(['--minimize', '--minimize_iters', '100'])
    else:
        # Full docking
        cmd.extend([
            '--center_x', str(center[0]),
            '--center_y', str(center[1]),
            '--center_z', str(center[2]),
            '--size_x', str(box_size),
            '--size_y', str(box_size),
            '--size_z', str(box_size),
            '--exhaustiveness', str(exhaustiveness),
            '--num_modes', str(num_modes),
        ])
    
    if autobox_ligand:
        autobox_abs = os.path.abspath(autobox_ligand)
        autobox_container = '/data' + autobox_abs[len(common_path):]
        cmd.extend(['--autobox_ligand', autobox_container])
    
    cmd.extend([
        '--cnn_scoring', cnn_scoring,
        '--cnn', 'crossdock_default2018',
    ])
    
    print(f"    Running GNINA via Docker...")
    print(f"    Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"    ERROR: GNINA failed (exit code {result.returncode})")
        return False
    
    return True


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
            chain_id = line[21]
            try:
                res_num = int(line[22:26].strip())
            except:
                continue
            
            if chain_id == chain and res_num == resid:
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords[atom_name] = np.array([x, y, z])
    
    return coords if 'SG' in coords else None


def create_michael_adduct(mol):
    """
    Create Michael addition product with -SCH3 attached to beta carbon.
    Returns: (adduct_mol, sulfur_idx, beta_carbon_idx)
    """
    # Find acrylamide pattern: N-C(=O)-C=C
    pattern = Chem.MolFromSmarts('[NX3][CX3](=O)[CX3]=[CX3]')
    matches = mol.GetSubstructMatches(pattern)
    
    if not matches:
        # Try simpler C=C adjacent to carbonyl
        pattern = Chem.MolFromSmarts('[CX3](=O)[CX3]=[CX3]')
        matches = mol.GetSubstructMatches(pattern)
    
    if not matches:
        raise ValueError("Acrylamide/vinyl pattern not found in molecule")
    
    match = matches[0]
    print(f"    Warhead atoms: {match}")
    
    # Create editable molecule
    rw = Chem.RWMol(Chem.Mol(mol))
    
    # Find the C=C double bond
    if len(match) == 4:  # Full pattern with N
        alpha_idx = match[2]
        beta_idx = match[3]
    else:  # Shorter pattern
        alpha_idx = match[1]
        beta_idx = match[2]
    
    print(f"    Alpha C: {alpha_idx}, Beta C (reactive): {beta_idx}")
    
    # Convert C=C to C-C
    bond = rw.GetBondBetweenAtoms(alpha_idx, beta_idx)
    if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
        bond.SetBondType(Chem.BondType.SINGLE)
    
    # Add S atom bonded to beta carbon
    s_idx = rw.AddAtom(Chem.Atom(16))  # Sulfur
    rw.AddBond(beta_idx, s_idx, Chem.BondType.SINGLE)
    
    # Add CH3 to S (dummy group representing rest of Cys)
    c_idx = rw.AddAtom(Chem.Atom(6))  # Carbon
    rw.AddBond(s_idx, c_idx, Chem.BondType.SINGLE)
    
    # Sanitize and add hydrogens
    adduct = rw.GetMol()
    try:
        Chem.SanitizeMol(adduct)
    except Exception as e:
        print(f"    WARNING: Sanitization issue: {e}")
    
    adduct = Chem.AddHs(adduct)
    
    # Generate 3D coordinates
    AllChem.EmbedMolecule(adduct, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(adduct, maxIters=500)
    
    print(f"    Adduct: {adduct.GetNumAtoms()} atoms (S idx: {s_idx})")
    
    return adduct, s_idx, beta_idx


def position_adduct_at_cys(adduct, s_idx, sg_pos, cb_pos):
    """Position the adduct so S is at the Cys SG position."""
    conf = adduct.GetConformer()
    
    # Get current S position
    s_pos = np.array(conf.GetAtomPosition(s_idx))
    
    # Translate to SG position
    translation = sg_pos - s_pos
    for i in range(adduct.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(pos.x + translation[0],
                                        pos.y + translation[1],
                                        pos.z + translation[2]))
    
    return adduct


def get_receptor_heavy_atoms(pdb_file, exclude_chain=None, exclude_resid=None):
    """Get heavy atom coordinates from receptor."""
    coords = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            atom_name = line[12:16].strip()
            if atom_name[0] == 'H':
                continue
            
            chain_id = line[21]
            try:
                res_num = int(line[22:26].strip())
            except:
                res_num = 0
            
            if exclude_chain and exclude_resid:
                if chain_id == exclude_chain and res_num == exclude_resid:
                    continue
            
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])
    
    return np.array(coords)


def rotate_molecule_around_axis(mol, center, axis, angle_rad):
    """Rotate molecule around an axis passing through center."""
    conf = mol.GetConformer()
    
    axis = axis / np.linalg.norm(axis)
    
    # Rodrigues rotation
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * np.dot(K, K)
    
    for i in range(mol.GetNumAtoms()):
        pos = np.array(conf.GetAtomPosition(i))
        pos_shifted = pos - center
        pos_rotated = np.dot(R, pos_shifted) + center
        conf.SetAtomPosition(i, Point3D(*pos_rotated))
    
    return mol


def count_clashes(mol, receptor_coords, clash_dist=2.0):
    """Count clashes between ligand heavy atoms and receptor."""
    conf = mol.GetConformer()
    clashes = 0
    
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() == 1:
            continue
        
        pos = np.array(conf.GetAtomPosition(i))
        dists = np.linalg.norm(receptor_coords - pos, axis=1)
        clashes += np.sum(dists < clash_dist)
    
    return clashes


def optimize_orientation(adduct, s_idx, sg_pos, receptor_coords, n_rotations=36):
    """Rotate adduct to minimize clashes."""
    conf = adduct.GetConformer()
    s_pos = np.array(conf.GetAtomPosition(s_idx))
    
    # Find C bonded to S (not methyl)
    s_atom = adduct.GetAtomWithIdx(s_idx)
    c_idx = None
    for nb in s_atom.GetNeighbors():
        if nb.GetSymbol() == 'C' and nb.GetDegree() > 1:
            c_idx = nb.GetIdx()
            break
    
    if c_idx is None:
        return adduct, count_clashes(adduct, receptor_coords)
    
    c_pos = np.array(conf.GetAtomPosition(c_idx))
    axis = s_pos - c_pos
    
    best_mol = Chem.Mol(adduct)
    best_clashes = count_clashes(adduct, receptor_coords)
    
    for i in range(n_rotations):
        angle = 2 * np.pi * i / n_rotations
        test_mol = Chem.Mol(adduct)
        test_mol = rotate_molecule_around_axis(test_mol, s_pos, axis, angle)
        
        clashes = count_clashes(test_mol, receptor_coords)
        if clashes < best_clashes:
            best_clashes = clashes
            best_mol = Chem.Mol(test_mol)
    
    return best_mol, best_clashes


def filter_poses_by_sulfur(docked_sdf, target_s_pos, max_dist=3.5):
    """Filter poses keeping only those with S near target."""
    suppl = Chem.SDMolSupplier(docked_sdf, removeHs=False)
    
    poses = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        
        # Find S atom
        s_idx = None
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'S':
                s_idx = atom.GetIdx()
                break
        
        if s_idx is None:
            continue
        
        conf = mol.GetConformer()
        s_pos = np.array(conf.GetAtomPosition(s_idx))
        dist = np.linalg.norm(s_pos - target_s_pos)
        
        # Get scores
        try:
            cnn_score = float(mol.GetProp('CNNscore')) if mol.HasProp('CNNscore') else 0
            cnn_affinity = float(mol.GetProp('CNNaffinity')) if mol.HasProp('CNNaffinity') else 0
            vina = float(mol.GetProp('minimizedAffinity')) if mol.HasProp('minimizedAffinity') else 0
        except:
            cnn_score = cnn_affinity = vina = 0
        
        poses.append({
            'mol': mol,
            'idx': i,
            's_dist': dist,
            'cnn_score': cnn_score,
            'cnn_affinity': cnn_affinity,
            'vina': vina
        })
    
    # Filter by distance
    good_poses = [p for p in poses if p['s_dist'] <= max_dist]
    
    # Sort by CNN score (higher is better)
    good_poses.sort(key=lambda x: x['cnn_score'], reverse=True)
    
    return good_poses


def write_complex_pdb(receptor_pdb, ligand_mol, output_pdb, cys_chain, cys_resid, beta_idx):
    """Write complex with CONECT records."""
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
                
                # Skip HG
                if (chain_id == cys_chain and res_num == cys_resid and 
                    res_name in ('CYS', 'CYM') and atom_name.startswith('HG')):
                    continue
                
                if (chain_id == cys_chain and res_num == cys_resid and 
                    atom_name == 'SG'):
                    sg_serial = serial
            
            lines.append(line)
    
    lines.append("TER\n")
    
    # Add ligand
    conf = ligand_mol.GetConformer()
    beta_serial = None
    
    for i in range(ligand_mol.GetNumAtoms()):
        atom = ligand_mol.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)
        symbol = atom.GetSymbol()
        serial = last_serial + 1 + i
        
        if i == beta_idx:
            beta_serial = serial
        
        name = f"{symbol}{i+1}"[:4]
        line = f"HETATM{serial:5d}  {name:<3s} LIG X   1    " \
               f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}" \
               f"  1.00  0.00          {symbol:>2s}\n"
        lines.append(line)
    
    if sg_serial and beta_serial:
        lines.append(f"CONECT{sg_serial:5d}{beta_serial:5d}\n")
        lines.append(f"CONECT{beta_serial:5d}{sg_serial:5d}\n")
    
    lines.append("END\n")
    
    with open(output_pdb, 'w') as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(description="Covalent docking with GNINA (Docker)")
    parser.add_argument('--ligand', '-l', required=True, help='Ligand SDF file')
    parser.add_argument('--receptor', '-r', required=True, help='Receptor PDB file')
    parser.add_argument('--cys_chain', default='A', help='Chain ID of target Cys')
    parser.add_argument('--cys_resid', type=int, required=True, help='Residue number of Cys')
    parser.add_argument('--output', '-o', default='gnina_covalent', help='Output directory')
    parser.add_argument('--box_size', type=float, default=22, help='Box size (Angstrom)')
    parser.add_argument('--exhaustiveness', type=int, default=32)
    parser.add_argument('--num_modes', type=int, default=20)
    parser.add_argument('--local_only', action='store_true', help='Local optimization only')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Covalent Docking with GNINA (Docker)")
    print("="*60)
    
    # Check Docker
    if not check_docker_gnina():
        sys.exit(1)
    
    # Create output directory
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Get Cys geometry
    print("\n[1] Loading cysteine geometry...")
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
    mol = suppl[0]
    if mol is None:
        print("ERROR: Could not load ligand")
        sys.exit(1)
    mol = Chem.AddHs(mol)
    print(f"    Atoms: {mol.GetNumAtoms()}")
    
    # Create adduct
    print("\n[3] Creating Michael addition adduct...")
    adduct, s_idx, beta_idx = create_michael_adduct(mol)
    
    adduct_sdf = str(outdir / "adduct.sdf")
    writer = Chem.SDWriter(adduct_sdf)
    writer.write(adduct)
    writer.close()
    print(f"    Saved: {adduct_sdf}")
    
    # Position adduct at Cys
    print("\n[4] Positioning adduct at Cys SG...")
    adduct = position_adduct_at_cys(adduct, s_idx, sg_pos, cb_pos)
    
    # Optimize orientation
    print("\n[5] Optimizing orientation to minimize clashes...")
    receptor_coords = get_receptor_heavy_atoms(args.receptor, args.cys_chain, args.cys_resid)
    adduct, n_clashes = optimize_orientation(adduct, s_idx, sg_pos, receptor_coords, n_rotations=72)
    print(f"    Best orientation: {n_clashes} clashes")
    
    positioned_sdf = str(outdir / "adduct_positioned.sdf")
    writer = Chem.SDWriter(positioned_sdf)
    writer.write(adduct)
    writer.close()
    print(f"    Saved: {positioned_sdf}")
    
    # Run GNINA
    docked_sdf = str(outdir / "docked_poses.sdf")
    
    print("\n[6] Running GNINA docking...")
    if args.local_only:
        success = run_gnina_docker(
            args.receptor, positioned_sdf, docked_sdf,
            center=sg_pos, minimize=True, cnn_scoring='rescore'
        )
    else:
        success = run_gnina_docker(
            args.receptor, positioned_sdf, docked_sdf,
            center=sg_pos, box_size=args.box_size,
            exhaustiveness=args.exhaustiveness,
            num_modes=args.num_modes,
            cnn_scoring='rescore'
        )
    
    if not success:
        print("ERROR: GNINA failed")
        sys.exit(1)
    
    # Filter and rank poses
    print("\n[7] Filtering poses by sulfur position...")
    good_poses = filter_poses_by_sulfur(docked_sdf, sg_pos, max_dist=4.0)
    
    if good_poses:
        print(f"    Found {len(good_poses)} poses with S near Cys:")
        for i, p in enumerate(good_poses[:5]):
            print(f"      {i+1}. S dist: {p['s_dist']:.2f} A, "
                  f"CNN: {p['cnn_score']:.3f}, affinity: {p['cnn_affinity']:.2f}")
        
        # Save best pose
        best = good_poses[0]
        best_sdf = str(outdir / "best_pose.sdf")
        writer = Chem.SDWriter(best_sdf)
        writer.write(best['mol'])
        writer.close()
        print(f"\n    Best pose: {best_sdf}")
        
        # Write complex
        complex_pdb = str(outdir / "complex.pdb")
        write_complex_pdb(args.receptor, best['mol'], complex_pdb,
                         args.cys_chain, args.cys_resid, beta_idx)
        print(f"    Complex: {complex_pdb}")
        
        # Save all good poses
        all_good_sdf = str(outdir / "good_poses.sdf")
        writer = Chem.SDWriter(all_good_sdf)
        for p in good_poses:
            writer.write(p['mol'])
        writer.close()
        print(f"    All good poses: {all_good_sdf}")
    else:
        print("    WARNING: No poses with S near Cys SG")
        print("    Check docked_poses.sdf manually")
    
    # Summary
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"\nOutput in: {outdir}/")
    print(f"\nVisualize:")
    print(f"  pymol {args.receptor} {outdir}/best_pose.sdf")
    print(f"\nNext: GROMACS equilibration")
    print(f"  python setup_covalent_topology.py \\")
    print(f"      --protein {args.receptor} \\")
    print(f"      --ligand {outdir}/best_pose.sdf \\")
    print(f"      --cys_resid {args.cys_resid}")


if __name__ == "__main__":
    main()
