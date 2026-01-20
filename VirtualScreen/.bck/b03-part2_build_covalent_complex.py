#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Covalent Complex - Proper Michael Addition Adduct

This script takes a docked ligand (with placeholder S) and a receptor,
and creates a proper covalent complex where:
1. The ligand's placeholder S is REMOVED
2. The ligand's beta-C is BONDED to Cys SG
3. The Cys HG (if present) is REMOVED
4. Proper CONECT records are written

The output is ready for GROMACS topology generation or visualization.

Usage:
    python build_covalent_complex.py \
        --ligand best_pose.sdf \
        --receptor protein.pdb \
        --cys_chain A --cys_resid 1039 \
        --output covalent_complex.pdb
"""

import argparse
import sys
from pathlib import Path
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("ERROR: RDKit required")
    sys.exit(1)


def find_placeholder_sulfur(mol):
    """Find the terminal sulfur (placeholder for Cys SG)."""
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'S':
            heavy_neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
            if len(heavy_neighbors) == 1:
                return atom.GetIdx()
    return None


def find_beta_carbon(mol, s_idx):
    """Find the beta carbon (the one bonded to placeholder S)."""
    s_atom = mol.GetAtomWithIdx(s_idx)
    for neighbor in s_atom.GetNeighbors():
        if neighbor.GetAtomicNum() == 6:  # Carbon
            return neighbor.GetIdx()
    return None


def remove_placeholder_sulfur(mol, s_idx):
    """
    Remove placeholder S and its hydrogens from the molecule.
    Returns new molecule and the beta carbon index in the new molecule.
    """
    # Find beta carbon before we remove anything
    beta_idx_old = find_beta_carbon(mol, s_idx)
    if beta_idx_old is None:
        raise ValueError("Could not find beta carbon attached to placeholder S")
    
    # Get atoms to remove (S and any H attached to it)
    atoms_to_remove = [s_idx]
    s_atom = mol.GetAtomWithIdx(s_idx)
    for neighbor in s_atom.GetNeighbors():
        if neighbor.GetAtomicNum() == 1:  # Hydrogen
            atoms_to_remove.append(neighbor.GetIdx())
    
    # Sort in descending order to remove from end first
    atoms_to_remove.sort(reverse=True)
    
    # Create editable molecule
    rw = Chem.RWMol(mol)
    
    # Remove atoms
    for idx in atoms_to_remove:
        rw.RemoveAtom(idx)
    
    new_mol = rw.GetMol()
    
    # Calculate new beta index (shifted by number of removed atoms with lower indices)
    removed_below_beta = sum(1 for idx in atoms_to_remove if idx < beta_idx_old)
    beta_idx_new = beta_idx_old - removed_below_beta
    
    return new_mol, beta_idx_new


def get_cys_info(pdb_file, chain, resid):
    """Get cysteine SG position and atom serial number."""
    sg_pos = None
    sg_serial = None
    hg_serials = []
    
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
            
            if chain_id != chain or res_num != resid:
                continue
            
            atom_name = line[12:16].strip()
            serial = int(line[6:11].strip())
            
            if atom_name == 'SG':
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                sg_pos = np.array([x, y, z])
                sg_serial = serial
            elif atom_name.startswith('HG'):
                hg_serials.append(serial)
    
    return {
        'sg_pos': sg_pos,
        'sg_serial': sg_serial,
        'hg_serials': hg_serials
    }


def write_covalent_complex(receptor_pdb, ligand_mol, output_pdb,
                           cys_chain, cys_resid, beta_idx):
    """
    Write a proper covalent complex PDB.
    
    - Removes Cys HG
    - Adds ligand atoms
    - Writes CONECT for covalent bond (SG - beta_C)
    """
    # Get cysteine info
    cys_info = get_cys_info(receptor_pdb, cys_chain, cys_resid)
    if cys_info['sg_serial'] is None:
        raise ValueError(f"Could not find Cys {cys_chain}:{cys_resid} SG")
    
    sg_serial = cys_info['sg_serial']
    hg_serials = set(cys_info['hg_serials'])
    
    lines = []
    last_serial = 0
    
    # Write receptor, skipping HG atoms
    with open(receptor_pdb, 'r') as f:
        for line in f:
            if line.startswith('END'):
                continue
            
            if line.startswith('ATOM') or line.startswith('HETATM'):
                serial = int(line[6:11].strip())
                last_serial = max(last_serial, serial)
                
                # Skip Cys HG atoms
                if serial in hg_serials:
                    print(f"    Removed Cys HG (serial {serial})")
                    continue
            
            # Skip existing CONECT records (we'll write our own)
            if line.startswith('CONECT'):
                continue
            
            lines.append(line)
    
    # Add TER if not present
    if lines and not lines[-1].startswith('TER'):
        lines.append("TER\n")
    
    # Write ligand atoms
    conf = ligand_mol.GetConformer()
    beta_serial = None
    ligand_start_serial = last_serial + 1
    
    print(f"    Writing ligand ({ligand_mol.GetNumAtoms()} atoms)")
    
    for i in range(ligand_mol.GetNumAtoms()):
        atom = ligand_mol.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)
        symbol = atom.GetSymbol()
        
        serial = ligand_start_serial + i
        
        if i == beta_idx:
            beta_serial = serial
            print(f"    Beta carbon at serial {serial}")
        
        # Format atom name (left-justified for 1-char elements, otherwise standard)
        if len(symbol) == 1:
            name = f" {symbol}{i+1:<2d}"[:4]
        else:
            name = f"{symbol}{i+1}"[:4]
        name = name.ljust(4)
        
        line = (f"HETATM{serial:5d} {name} LIG L   1    "
                f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
                f"  1.00  0.00          {symbol:>2s}\n")
        lines.append(line)
    
    # Write CONECT records for covalent bond
    if beta_serial is not None:
        print(f"    Covalent bond: SG (serial {sg_serial}) - C (serial {beta_serial})")
        lines.append(f"CONECT{sg_serial:5d}{beta_serial:5d}\n")
        lines.append(f"CONECT{beta_serial:5d}{sg_serial:5d}\n")
    else:
        print("    WARNING: Could not determine beta carbon serial!")
    
    lines.append("END\n")
    
    # Write output
    with open(output_pdb, 'w') as f:
        f.writelines(lines)
    
    return beta_serial is not None


def main():
    parser = argparse.ArgumentParser(
        description="Build proper covalent complex from docked ligand",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script:
1. Removes the placeholder sulfur from the docked ligand
2. Removes Cys HG from the receptor (if present)
3. Writes CONECT records for the covalent bond

The output PDB has the ligand's beta-carbon directly bonded to Cys SG.

Example:
    python %(prog)s \\
        --ligand docking_results/best_pose.sdf \\
        --receptor protein.pdb \\
        --cys_chain A --cys_resid 1039 \\
        --output covalent_complex.pdb

Visualization in PyMOL:
    load covalent_complex.pdb
    show sticks, resn LIG or (resn CYS and resi 1039)
    # PyMOL should show the covalent bond from CONECT records
        """
    )
    
    parser.add_argument('--ligand', '-l', required=True,
                        help='Docked ligand SDF (with placeholder S)')
    parser.add_argument('--receptor', '-r', required=True,
                        help='Receptor PDB')
    parser.add_argument('--cys_chain', default='A',
                        help='Chain ID of target cysteine')
    parser.add_argument('--cys_resid', type=int, required=True,
                        help='Residue number of target cysteine')
    parser.add_argument('--output', '-o', required=True,
                        help='Output PDB file')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BUILD COVALENT COMPLEX")
    print("=" * 60)
    
    # Load ligand
    print(f"\n[1] Loading ligand: {args.ligand}")
    suppl = Chem.SDMolSupplier(args.ligand, removeHs=False)
    mol = next(suppl, None)
    if mol is None:
        print("ERROR: Could not load ligand")
        sys.exit(1)
    print(f"    Loaded: {mol.GetNumAtoms()} atoms")
    
    # Find and remove placeholder S
    print("\n[2] Removing placeholder sulfur...")
    s_idx = find_placeholder_sulfur(mol)
    if s_idx is None:
        print("ERROR: Could not find placeholder sulfur in ligand")
        print("       Make sure you're using the Michael adduct (with -SH attached)")
        sys.exit(1)
    print(f"    Found placeholder S at index {s_idx}")
    
    # Get beta carbon before removing S
    beta_idx_before = find_beta_carbon(mol, s_idx)
    print(f"    Beta carbon at index {beta_idx_before}")
    
    # Remove S
    clean_ligand, beta_idx = remove_placeholder_sulfur(mol, s_idx)
    print(f"    After removal: {clean_ligand.GetNumAtoms()} atoms")
    print(f"    Beta carbon now at index {beta_idx}")
    
    # Verify the beta carbon
    beta_atom = clean_ligand.GetAtomWithIdx(beta_idx)
    print(f"    Beta carbon element: {beta_atom.GetSymbol()}")
    if beta_atom.GetSymbol() != 'C':
        print("ERROR: Beta carbon index is wrong!")
        sys.exit(1)
    
    # Write complex
    print(f"\n[3] Writing covalent complex: {args.output}")
    success = write_covalent_complex(
        args.receptor, clean_ligand, args.output,
        args.cys_chain, args.cys_resid, beta_idx
    )
    
    if not success:
        print("ERROR: Failed to write complex")
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {args.output}")
    print(f"\nVisualize in PyMOL:")
    print(f"  pymol {args.output}")
    print(f"  # Then run:")
    print(f"  show sticks, resn LIG or (resn CYS and resi {args.cys_resid})")
    print(f"  color yellow, resn LIG")
    print(f"  color green, resn CYS and resi {args.cys_resid}")
    print(f"\nThe covalent bond should be visible from CONECT records.")
    print(f"\nNote: Do NOT load best_pose.sdf alongside this - it still has")
    print(f"      the placeholder sulfur and will confuse visualization.")


if __name__ == "__main__":
    main()
