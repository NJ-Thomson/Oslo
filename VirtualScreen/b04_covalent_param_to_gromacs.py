#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automate covalent parameterization for GROMACS via AmberTools + GAFF2 + ACPYPE.

Workflow:
  1) Prepare receptor PDB: ensure target cysteine is neutral thioether (CYS, no HG/HSG).
  2) Generate ligand parameters: antechamber (GAFF2, AM1-BCC) + parmchk2.
  3) Detect acrylamide beta carbon in ligand SDF and rename that atom to 'CBL' in LIG.mol2.
  4) Build covalent S–C bond in tleap (ff14SB + GAFF2) and save prmtop/inpcrd and complex PDB.
  5) Convert Amber topology to GROMACS with ACPYPE.

Requirements:
  - AmberTools on PATH (antechamber, parmchk2, tleap)
  - ACPYPE on PATH
  - RDKit
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import shutil
from typing import Optional, Tuple, List

# RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("ERROR: RDKit is required. Install with 'pip install rdkit' or conda.")
    sys.exit(1)


def check_executable(name: str) -> Optional[str]:
    exe = shutil.which(name)
    if exe is None:
        print(f"ERROR: Required executable '{name}' not found on PATH.")
    return exe


def enforce_cys_neutral_thioether(pdb_in: str, pdb_out: str, chain_id: Optional[str], resid: int) -> None:
    """
    Ensure the target cysteine is neutral thioether for MD:
      - Rename residue to CYS (from CYM if present)
      - Remove thiol hydrogen atoms (HG/HSG)
    Keeps residue numbering/chain IDs unchanged.
    """
    lines_out = []
    with open(pdb_in, 'r') as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                lines_out.append(line)
                continue

            chn = line[21]
            try:
                resseq = int(line[22:26])
            except Exception:
                resseq = None

            if (chain_id is None or chn == chain_id) and resseq == resid:
                atom_name = line[12:16].strip()
                # Drop thiol hydrogens if any
                if atom_name in ('HG', 'HSG'):
                    continue
                # Force residue name to CYS
                line = line[:17] + 'CYS' + line[20:]
            lines_out.append(line)

    with open(pdb_out, 'w') as w:
        w.writelines(lines_out)


def rdkit_detect_beta_carbon(sdf_path: str) -> Optional[Tuple[int, List[Tuple[float, float, float]]]]:
    """
    Detect acrylamide beta carbon in the ligand SDF using SMARTS.
    Returns (beta_idx, coords_list) where coords_list is one 3D conformer’s coordinates in order.
    """
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = None
    for m in suppl:
        if m is not None:
            mol = m
            break
    if mol is None or mol.GetNumConformers() == 0:
        print("ERROR: Could not read ligand SDF or no 3D conformers present.")
        return None

    # Identify beta carbon: acrylamide N-C(=O)-C=C ; beta is terminal vinylic carbon
    patterns = [
        '[NX3][CX3](=O)[CX3]=[CH2]',
        '[NX3][CX3](=O)[CH]=[CH2]',
        '[NX3][CX3](=O)[CX3]=[CX3]',
        '[CX3](=O)[CX3]=[CH2]',
    ]
    beta_idx = None
    for smarts in patterns:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            match = matches[0]
            alpha_idx = match[-2]
            b_idx = match[-1]
            bond = mol.GetBondBetweenAtoms(alpha_idx, b_idx)
            if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                beta_idx = b_idx
                break
    if beta_idx is None:
        print("ERROR: Acrylamide beta carbon not found via SMARTS.")
        return None

    conf = mol.GetConformer()
    coords = []
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        coords.append((float(p.x), float(p.y), float(p.z)))
    return beta_idx, coords


def parse_mol2_atoms(mol2_path: str) -> List[dict]:
    """
    Parse the @<TRIPOS>ATOM block of a mol2 file.
    Returns list of dicts for each atom:
      {'idx': int, 'name': str, 'x': float, 'y': float, 'z': float, 'rest': str}
    'idx' is MOL2 atom index (1-based).
    'rest' stores the trailing fields to reconstruct the line.
    """
    atoms = []
    with open(mol2_path, 'r') as f:
        lines = f.readlines()

    in_atom = False
    for line in lines:
        if line.strip().startswith("@<TRIPOS>ATOM"):
            in_atom = True
            continue
        if line.strip().startswith("@<TRIPOS>"):
            in_atom = False
        if in_atom:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            idx = int(parts[0])
            name = parts[1]
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            # Keep the rest (substructure id, type, charges, etc.)
            rest = " ".join(parts[5:])
            atoms.append({'idx': idx, 'name': name, 'x': x, 'y': y, 'z': z, 'rest': rest})
    return atoms


def rewrite_mol2_atom_name(mol2_in: str, mol2_out: str, target_idx: int, new_name: str) -> None:
    """
    Rewrite atom name for atom with MOL2 index target_idx to new_name.
    """
    out_lines = []
    with open(mol2_in, 'r') as f:
        lines = f.readlines()

    in_atom = False
    for line in lines:
        if line.strip().startswith("@<TRIPOS>ATOM"):
            in_atom = True
            out_lines.append(line)
            continue
        if line.strip().startswith("@<TRIPOS>"):
            in_atom = False
            out_lines.append(line)
            continue

        if in_atom:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    idx = int(parts[0])
                except Exception:
                    idx = None
                if idx == target_idx:
                    parts[1] = new_name
                    line = "{:<7d} {:<4s} {:>10.4f} {:>10.4f} {:>10.4f} {}\n".format(
                        idx, parts[1], float(parts[2]), float(parts[3]), float(parts[4]),
                        " ".join(parts[5:])
                    )
        out_lines.append(line)

    with open(mol2_out, 'w') as w:
        w.writelines(out_lines)


def find_mol2_atom_index_by_coords(mol2_atoms: List[dict], target_xyz: Tuple[float, float, float],
                                   tol: float = 0.5) -> Optional[int]:
    """
    Find the MOL2 atom index whose coordinates are closest to target_xyz.
    Returns the MOL2 atom index (1-based) if within tol (Å), else None.
    """
    import math
    best_idx = None
    best_d = float('inf')
    tx, ty, tz = target_xyz
    for a in mol2_atoms:
        d = math.sqrt((a['x'] - tx)**2 + (a['y'] - ty)**2 + (a['z'] - tz)**2)
        if d < best_d:
            best_d = d
            best_idx = a['idx']
    if best_d <= tol:
        return best_idx
    print(f"WARNING: Closest MOL2 atom to target is {best_d:.3f} Å away (tol {tol:.2f} Å).")
    return best_idx  # still return the closest, but warn


def run_cmd(cmd: List[str], cwd: Optional[str] = None) -> bool:
    print(">>", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print("STDOUT:\n", res.stdout)
        print("STDERR:\n", res.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Automate covalent parameterization to GROMACS (AmberTools + ACPYPE).")
    parser.add_argument('--receptor', required=True, help='Protein receptor PDB (protonated; docking output).')
    parser.add_argument('--ligand', required=True, help='Ligand SDF (best pose).')
    parser.add_argument('--cys_resid', type=int, required=True, help='Cysteine residue number for covalent bond (tleap numbering).')
    parser.add_argument('--cys_chain', default=None, help='Optional chain ID (used only to identify Cys in receptor PDB preprocessing).')
    parser.add_argument('--outdir', default='md_params', help='Output directory.')
    parser.add_argument('--ff_protein', default='leaprc.protein.ff14SB', help='Amber protein FF leaprc.')
    parser.add_argument('--ff_ligand', default='leaprc.gaff2', help='Amber ligand FF leaprc.')
    parser.add_argument('--lig_charge', default='bcc', choices=['bcc', 'resp', 'gas'], help='Charge method for antechamber (default: bcc).')
    parser.add_argument('--atom_name', default='CBL', help='Atom name to assign to β-carbon in LIG.mol2 (default: CBL).')
    parser.add_argument('--tol_match', type=float, default=0.5, help='Tolerance (Å) to match β-carbon SDF → MOL2 coordinates.')
    parser.add_argument('--skip_acpype', action='store_true', help='Skip ACPYPE conversion to GROMACS.')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Check tools
    for tool in ['antechamber', 'parmchk2', 'tleap']:
        if check_executable(tool) is None:
            sys.exit(1)
    if not args.skip_acpype and check_executable('acpype') is None:
        print("WARNING: ACPYPE not found; will skip GROMACS conversion.")
        args.skip_acpype = True

    receptor_in = Path(args.receptor)
    ligand_sdf = Path(args.ligand)
    if not receptor_in.exists() or not ligand_sdf.exists():
        print("ERROR: Input files not found.")
        sys.exit(1)

    # 1) Prepare receptor: ensure neutral thioether CYS at target residue (rename CYM->CYS, drop HG/HSG)
    receptor_prep = outdir / "receptor_CYS_thioether.pdb"
    enforce_cys_neutral_thioether(str(receptor_in), str(receptor_prep), args.cys_chain, args.cys_resid)
    print(f"Prepared receptor: {receptor_prep}")

    # 2) Generate ligand parameters (GAFF2 + AM1-BCC)
    lig_mol2 = outdir / "LIG.mol2"
    lig_frcmod = outdir / "LIG.frcmod"

    cmd_ante = [
        'antechamber', '-i', str(ligand_sdf), '-fi', 'sdf',
        '-o', str(lig_mol2), '-fo', 'mol2',
        '-at', 'gaff2', '-c', args.lig_charge, '-s', '2'
    ]
    if not run_cmd(cmd_ante):
        print("ERROR: antechamber failed.")
        sys.exit(1)

    cmd_parmchk = ['parmchk2', '-i', str(lig_mol2), '-f', 'mol2', '-o', str(lig_frcmod)]
    if not run_cmd(cmd_parmchk):
        print("ERROR: parmchk2 failed.")
        sys.exit(1)
    print(f"Ligand parameters: {lig_mol2}, {lig_frcmod}")

    # 3) Detect β-carbon and rename that atom in LIG.mol2 to args.atom_name (CBL)
    res_beta = rdkit_detect_beta_carbon(str(ligand_sdf))
    if res_beta is None:
        print("ERROR: Could not detect acrylamide beta carbon.")
        sys.exit(1)
    beta_idx, sdf_coords = res_beta
    target_xyz = sdf_coords[beta_idx]

    mol2_atoms = parse_mol2_atoms(str(lig_mol2))
    mol2_beta_idx = find_mol2_atom_index_by_coords(mol2_atoms, target_xyz, tol=args.tol_match)
    if mol2_beta_idx is None:
        print("ERROR: Could not match β-carbon from SDF to MOL2 atom list.")
        sys.exit(1)
    lig_mol2_named = outdir / "LIG_named.mol2"
    rewrite_mol2_atom_name(str(lig_mol2), str(lig_mol2_named), mol2_beta_idx, args.atom_name)
    print(f"Renamed MOL2 atom {mol2_beta_idx} to {args.atom_name} for bonding.")

    # 4) Create leap.in to build bond and save prmtop/inpcrd
    leap_in = outdir / "leap.in"
    complex_prmtop = outdir / "complex.prmtop"
    complex_inpcrd = outdir / "complex.inpcrd"
    complex_pdb = outdir / "complex_amber.pdb"

    leap_script = f"""
source {args.ff_protein}
source {args.ff_ligand}

loadamberparams {lig_frcmod}

protein = loadpdb {receptor_prep}
lig     = loadmol2 {lig_mol2_named}

# Create the covalent bond: protein CYS SG to ligand beta carbon
bond protein.{args.cys_resid}.SG lig.1.{args.atom_name}

complex = combine {{ protein lig }}

saveamberparm complex {complex_prmtop} {complex_inpcrd}
savepdb complex {complex_pdb}
quit
"""
    with open(leap_in, 'w') as f:
        f.write(leap_script.strip() + "\n")

    print(f"Running tleap with script: {leap_in}")
    if not run_cmd(['tleap', '-f', str(leap_in)]):
        print("ERROR: tleap failed. Check leap.log in current directory for details if present.")
        sys.exit(1)

    if not (complex_prmtop.exists() and complex_inpcrd.exists()):
        print("ERROR: Amber topology files not generated.")
        sys.exit(1)

    print(f"Amber outputs: {complex_prmtop}, {complex_inpcrd}, {complex_pdb}")

    # 5) Convert to GROMACS via ACPYPE (optional)
    if not args.skip_acpype:
        print("Converting Amber topology to GROMACS with ACPYPE...")
        # -p prmtop -x inpcrd; -d to keep directory structure (double precision flag not needed here)
        if not run_cmd(['acpype', '-p', str(complex_prmtop), '-x', str(complex_inpcrd), '-d'], cwd=str(outdir)):
            print("ERROR: ACPYPE conversion failed.")
            sys.exit(1)
        # ACPYPE creates a directory with suffix _GMX; find it
        gmx_dirs = [p for p in outdir.iterdir() if p.is_dir() and p.name.endswith('_GMX')]
        if gmx_dirs:
            print(f"GROMACS files in: {gmx_dirs[0]}")
        else:
            print("WARNING: ACPYPE finished but _GMX directory not found. Check output for details.")

    print("\nDone.")
    print("Next steps in GROMACS (example):")
    print("  gmx editconf -f complex_GMX.gro -o boxed.gro -c -d 1.0 -bt cubic")
    print("  gmx solvate -cp boxed.gro -cs spc216.gro -o solv.gro -p complex_GMX.top")
    print("  gmx grompp -f ions.mdp -c solv.gro -p complex_GMX.top -o ions.tpr")
    print("  gmx genion -s ions.tpr -o solv_ions.gro -p complex_GMX.top -pname NA -nname CL -neutral")
    print("  gmx grompp -f minim.mdp -c solv_ions.gro -p complex_GMX.top -o em.tpr")
    print("  gmx mdrun -deffnm em")
    print("Then proceed with NVT/NPT/production as usual.")

if __name__ == "__main__":
    main()
