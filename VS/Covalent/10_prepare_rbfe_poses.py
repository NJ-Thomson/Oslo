#!/usr/bin/env python3
"""
04_prepare_rbfe_poses.py

Prepare poses for RBFE calculations between acrylamide and chloroacetamide warheads.

This script:
1. Finds overlapping poses between acry and chlo docking results using MCS-based RMSD
2. Creates 4 complex PDBs (ligand + receptor) for the selected poses
3. Runs parameterization (b04 + acpype) only for the 2 unique ligands
4. Outputs structures ready for b05/b06 pipeline

The 4 systems created:
  - acry_best: Best acrylamide pose (by Vina affinity)
  - chlo_to_acry: Chloroacetamide pose closest to acry_best
  - chlo_best: Best chloroacetamide pose (by Vina affinity)
  - acry_to_chlo: Acrylamide pose closest to chlo_best

For RBFE, you would run:
  - Forward from acry: acry_best → chlo_to_acry
  - Forward from chlo: acry_to_chlo → chlo_best

Usage:
    python 04_prepare_rbfe_poses.py \\
        --receptor receptor.pdb \\
        --acry_sdf docked_poses_acry.sdf \\
        --chlo_sdf docked_poses_chlo.sdf \\
        --cys_resid 1039 \\
        --output_dir Outputs/RBFE_prep
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdFMCS
except ImportError:
    print("ERROR: RDKit required. Install via conda/pip.")
    sys.exit(1)


# ============================================================================
# Pose Selection (MCS-based RMSD)
# ============================================================================

def load_poses(sdf_path: Path) -> List[Chem.Mol]:
    """Load all poses from SDF file."""
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    poses = [mol for mol in suppl if mol is not None]
    print(f"  Loaded {len(poses)} poses from {sdf_path.name}")
    return poses


def get_vina_affinity(mol: Chem.Mol) -> float:
    """Get Vina affinity from molecule properties."""
    for prop in ['minimizedAffinity', 'affinity']:
        if mol.HasProp(prop):
            try:
                return float(mol.GetProp(prop))
            except:
                pass
    return 0.0


def find_mcs_mapping(mol1: Chem.Mol, mol2: Chem.Mol) -> List[Tuple[int, int]]:
    """Find Maximum Common Substructure atom mapping between two molecules."""
    mcs = rdFMCS.FindMCS(
        [mol1, mol2],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        matchValences=False,
        ringMatchesRingOnly=False,
        completeRingsOnly=False,
        timeout=10
    )

    if mcs.canceled or mcs.numAtoms == 0:
        return []

    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    if mcs_mol is None:
        return []

    matches1 = mol1.GetSubstructMatches(mcs_mol)
    matches2 = mol2.GetSubstructMatches(mcs_mol)

    if not matches1 or not matches2:
        return []

    return list(zip(matches1[0], matches2[0]))


def calculate_rmsd(mol1: Chem.Mol, mol2: Chem.Mol, atom_mapping: List[Tuple[int, int]]) -> float:
    """Calculate RMSD between two molecules using atom mapping."""
    if not atom_mapping:
        return float('inf')

    conf1 = mol1.GetConformer()
    conf2 = mol2.GetConformer()

    sq_dists = []
    for idx1, idx2 in atom_mapping:
        p1 = conf1.GetAtomPosition(idx1)
        p2 = conf2.GetAtomPosition(idx2)
        sq_dists.append((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    return np.sqrt(np.mean(sq_dists))


def select_rbfe_poses(acry_poses: List[Chem.Mol], chlo_poses: List[Chem.Mol]) -> Dict:
    """
    Select poses for RBFE calculations.

    Returns dict with:
        - acry_best_idx, acry_best_mol, acry_best_affinity
        - chlo_best_idx, chlo_best_mol, chlo_best_affinity
        - chlo_to_acry_idx, chlo_to_acry_mol, chlo_to_acry_rmsd
        - acry_to_chlo_idx, acry_to_chlo_mol, acry_to_chlo_rmsd
    """
    # Find MCS mapping (compute once)
    print("\n  Finding Maximum Common Substructure...")
    mcs_mapping = find_mcs_mapping(acry_poses[0], chlo_poses[0])

    if mcs_mapping:
        atoms = [acry_poses[0].GetAtomWithIdx(i).GetSymbol() for i, _ in mcs_mapping]
        print(f"  MCS: {len(mcs_mapping)} atoms")
    else:
        print("  WARNING: MCS failed")

    # Get best poses by Vina affinity
    acry_affinities = [get_vina_affinity(m) for m in acry_poses]
    chlo_affinities = [get_vina_affinity(m) for m in chlo_poses]

    best_acry_idx = int(np.argmin(acry_affinities))
    best_chlo_idx = int(np.argmin(chlo_affinities))

    print(f"\n  Best ACRY: pose {best_acry_idx} ({acry_affinities[best_acry_idx]:.2f} kcal/mol)")
    print(f"  Best CHLO: pose {best_chlo_idx} ({chlo_affinities[best_chlo_idx]:.2f} kcal/mol)")

    # Find chlo pose closest to best acry
    rmsds_to_acry = []
    for chlo_mol in chlo_poses:
        rmsd = calculate_rmsd(acry_poses[best_acry_idx], chlo_mol, mcs_mapping)
        rmsds_to_acry.append(rmsd)
    chlo_to_acry_idx = int(np.argmin(rmsds_to_acry))

    # Find acry pose closest to best chlo
    rmsds_to_chlo = []
    for acry_mol in acry_poses:
        rmsd = calculate_rmsd(acry_mol, chlo_poses[best_chlo_idx], mcs_mapping)
        rmsds_to_chlo.append(rmsd)
    acry_to_chlo_idx = int(np.argmin(rmsds_to_chlo))

    print(f"\n  CHLO closest to best ACRY: pose {chlo_to_acry_idx} (RMSD={rmsds_to_acry[chlo_to_acry_idx]:.2f} Å)")
    print(f"  ACRY closest to best CHLO: pose {acry_to_chlo_idx} (RMSD={rmsds_to_chlo[acry_to_chlo_idx]:.2f} Å)")

    return {
        'acry_best': {
            'idx': best_acry_idx,
            'mol': acry_poses[best_acry_idx],
            'affinity': acry_affinities[best_acry_idx]
        },
        'chlo_best': {
            'idx': best_chlo_idx,
            'mol': chlo_poses[best_chlo_idx],
            'affinity': chlo_affinities[best_chlo_idx]
        },
        'chlo_to_acry': {
            'idx': chlo_to_acry_idx,
            'mol': chlo_poses[chlo_to_acry_idx],
            'affinity': chlo_affinities[chlo_to_acry_idx],
            'rmsd': rmsds_to_acry[chlo_to_acry_idx]
        },
        'acry_to_chlo': {
            'idx': acry_to_chlo_idx,
            'mol': acry_poses[acry_to_chlo_idx],
            'affinity': acry_affinities[acry_to_chlo_idx],
            'rmsd': rmsds_to_chlo[acry_to_chlo_idx]
        },
        'mcs_n_atoms': len(mcs_mapping) if mcs_mapping else 0
    }


# ============================================================================
# Complex Assembly
# ============================================================================

def write_ligand_pdb(mol: Chem.Mol, output_path: Path, resname: str = "LIG"):
    """Write ligand to PDB file."""
    # Set residue info
    for atom in mol.GetAtoms():
        mi = Chem.AtomPDBResidueInfo()
        mi.SetResidueName(resname)
        mi.SetResidueNumber(1)
        mi.SetChainId("A")
        mi.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(mi)

    Chem.MolToPDBFile(mol, str(output_path))


def create_complex_pdb(receptor_pdb: Path, ligand_mol: Chem.Mol,
                       output_path: Path, lig_resname: str = "LIG"):
    """Create complex PDB by combining receptor and ligand."""
    # Write ligand to temp file
    lig_tmp = output_path.parent / f"_tmp_lig_{output_path.stem}.pdb"
    write_ligand_pdb(ligand_mol, lig_tmp, lig_resname)

    # Read receptor
    with open(receptor_pdb) as f:
        receptor_lines = [l for l in f if l.startswith(('ATOM', 'HETATM', 'TER'))]

    # Read ligand
    with open(lig_tmp) as f:
        ligand_lines = [l for l in f if l.startswith(('ATOM', 'HETATM'))]

    # Combine (receptor first, then ligand as HETATM)
    with open(output_path, 'w') as f:
        for line in receptor_lines:
            if not line.startswith('TER'):
                f.write(line)
        f.write("TER\n")
        for line in ligand_lines:
            # Ensure ligand atoms are HETATM
            if line.startswith('ATOM'):
                line = 'HETATM' + line[6:]
            f.write(line)
        f.write("END\n")

    # Cleanup
    lig_tmp.unlink()


# ============================================================================
# Parameterization (b04 + acpype)
# ============================================================================

def run_b04(complex_pdb: Path, cys_resid: int, lig_resname: str,
            output_prefix: Path, cap_type: str = "aliphatic") -> Path:
    """Run b04 to extract adduct fragment."""
    script = Path(__file__).parent / "b04_extract_adduct_fragment.py"

    cmd = [
        sys.executable, str(script),
        "--complex", str(complex_pdb),
        "--cys-resid", str(cys_resid),
        "--lig-resname", lig_resname,
        "--out-prefix", str(output_prefix),
        "--cap-type", cap_type,
        "--make-mol2"
    ]

    print(f"    Running b04 for {complex_pdb.name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    ERROR: {result.stderr}")
        raise RuntimeError("b04 failed")

    return Path(f"{output_prefix}.pdb")


def run_acpype(adduct_pdb: Path, output_dir: Path, net_charge: int = 0) -> Dict[str, Path]:
    """Run acpype for GAFF2 parameterization."""
    # Check for acpype
    acpype_cmd = shutil.which("acpype")
    if not acpype_cmd:
        raise RuntimeError("acpype not found in PATH")

    print(f"    Running acpype on {adduct_pdb.name}...")

    cmd = [
        acpype_cmd,
        "-i", str(adduct_pdb),
        "-c", "bcc",  # AM1-BCC charges
        "-a", "gaff2",
        "-n", str(net_charge),
        "-o", "gmx"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)

    if result.returncode != 0:
        print(f"    STDERR: {result.stderr}")
        # acpype sometimes returns non-zero but still works

    # Find output files
    base = adduct_pdb.stem
    acpype_dir = output_dir / f"{base}.acpype"

    if not acpype_dir.exists():
        raise RuntimeError(f"acpype output not found: {acpype_dir}")

    return {
        'mol2': acpype_dir / f"{base}_NEW.mol2",
        'itp': acpype_dir / f"{base}_GMX.itp",
        'gro': acpype_dir / f"{base}_GMX.gro",
        'dir': acpype_dir
    }


# ============================================================================
# Main Workflow
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Prepare poses for RBFE calculations',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--receptor', required=True, help='Receptor PDB file')
    parser.add_argument('--acry_sdf', required=True, help='Acrylamide docked poses SDF')
    parser.add_argument('--chlo_sdf', required=True, help='Chloroacetamide docked poses SDF')
    parser.add_argument('--cys_resid', type=int, required=True, help='Covalent cysteine residue ID')
    parser.add_argument('--output_dir', default='RBFE_prep', help='Output directory')
    parser.add_argument('--lig_resname', default='LIG', help='Ligand residue name in complex')
    parser.add_argument('--cap_type', default='aliphatic', choices=['aliphatic', 'methyl'],
                        help='Cap type for b04')
    parser.add_argument('--skip_param', action='store_true',
                        help='Skip parameterization (only select poses)')
    args = parser.parse_args()

    receptor = Path(args.receptor)
    acry_sdf = Path(args.acry_sdf)
    chlo_sdf = Path(args.chlo_sdf)
    output_dir = Path(args.output_dir)

    # Validate inputs
    for p, name in [(receptor, 'receptor'), (acry_sdf, 'acry_sdf'), (chlo_sdf, 'chlo_sdf')]:
        if not p.exists():
            print(f"ERROR: {name} not found: {p}")
            return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RBFE POSE PREPARATION")
    print("=" * 70)
    print(f"\nReceptor:    {receptor}")
    print(f"ACRY poses:  {acry_sdf}")
    print(f"CHLO poses:  {chlo_sdf}")
    print(f"CYS residue: {args.cys_resid}")
    print(f"Output:      {output_dir}")

    # Step 1: Load and select poses
    print("\n" + "=" * 70)
    print("STEP 1: Selecting poses")
    print("=" * 70)

    acry_poses = load_poses(acry_sdf)
    chlo_poses = load_poses(chlo_sdf)

    if not acry_poses or not chlo_poses:
        print("ERROR: No poses found")
        return 1

    selected = select_rbfe_poses(acry_poses, chlo_poses)

    # Step 2: Create complex PDBs
    print("\n" + "=" * 70)
    print("STEP 2: Creating complex PDBs")
    print("=" * 70)

    complexes_dir = output_dir / "complexes"
    complexes_dir.mkdir(exist_ok=True)

    complex_paths = {}
    systems = [
        ('acry_best', selected['acry_best']['mol']),
        ('chlo_to_acry', selected['chlo_to_acry']['mol']),
        ('chlo_best', selected['chlo_best']['mol']),
        ('acry_to_chlo', selected['acry_to_chlo']['mol']),
    ]

    for name, mol in systems:
        out_pdb = complexes_dir / f"complex_{name}.pdb"
        create_complex_pdb(receptor, mol, out_pdb, args.lig_resname)
        complex_paths[name] = out_pdb
        print(f"  Created: {out_pdb.name}")

    # Also save individual ligand SDFs for reference
    for name, mol in systems:
        out_sdf = complexes_dir / f"ligand_{name}.sdf"
        writer = Chem.SDWriter(str(out_sdf))
        writer.write(mol)
        writer.close()

    # Step 3: Parameterization (only for best poses)
    print("\n" + "=" * 70)
    print("STEP 3: Parameterization (b04 + acpype)")
    print("=" * 70)

    if args.skip_param:
        print("  Skipping parameterization (--skip_param)")
    else:
        params_dir = output_dir / "params"
        params_dir.mkdir(exist_ok=True)

        param_results = {}

        # Only parameterize acry_best and chlo_best
        for ligand_type, system_name in [('acry', 'acry_best'), ('chlo', 'chlo_best')]:
            print(f"\n  Parameterizing {ligand_type} (from {system_name})...")

            lig_params_dir = params_dir / ligand_type
            lig_params_dir.mkdir(exist_ok=True)

            # Run b04
            adduct_prefix = lig_params_dir / "adduct"
            adduct_pdb = run_b04(
                complex_paths[system_name],
                args.cys_resid,
                args.lig_resname,
                adduct_prefix,
                args.cap_type
            )

            # Run acpype
            acpype_out = run_acpype(adduct_pdb, lig_params_dir)

            param_results[ligand_type] = {
                'adduct_pdb': str(adduct_pdb),
                'mol2': str(acpype_out['mol2']),
                'itp': str(acpype_out['itp']),
                'gro': str(acpype_out['gro']),
                'acpype_dir': str(acpype_out['dir'])
            }

            print(f"    Output: {acpype_out['dir']}")

        # Save param results
        with open(output_dir / "parameterization.json", 'w') as f:
            json.dump(param_results, f, indent=2)

    # Save selection metadata
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    metadata = {
        'receptor': str(receptor),
        'acry_sdf': str(acry_sdf),
        'chlo_sdf': str(chlo_sdf),
        'cys_resid': args.cys_resid,
        'mcs_atoms': selected['mcs_n_atoms'],
        'selections': {
            'acry_best': {
                'pose_idx': selected['acry_best']['idx'],
                'affinity': selected['acry_best']['affinity'],
                'complex_pdb': str(complex_paths['acry_best'])
            },
            'chlo_best': {
                'pose_idx': selected['chlo_best']['idx'],
                'affinity': selected['chlo_best']['affinity'],
                'complex_pdb': str(complex_paths['chlo_best'])
            },
            'chlo_to_acry': {
                'pose_idx': selected['chlo_to_acry']['idx'],
                'affinity': selected['chlo_to_acry']['affinity'],
                'rmsd_to_acry_best': selected['chlo_to_acry']['rmsd'],
                'complex_pdb': str(complex_paths['chlo_to_acry'])
            },
            'acry_to_chlo': {
                'pose_idx': selected['acry_to_chlo']['idx'],
                'affinity': selected['acry_to_chlo']['affinity'],
                'rmsd_to_chlo_best': selected['acry_to_chlo']['rmsd'],
                'complex_pdb': str(complex_paths['acry_to_chlo'])
            }
        },
        'rbfe_pairs': {
            'forward_from_acry': {
                'stateA': 'acry_best',
                'stateB': 'chlo_to_acry',
                'description': 'ACRY→CHLO starting from best acry pose'
            },
            'forward_from_chlo': {
                'stateA': 'acry_to_chlo',
                'stateB': 'chlo_best',
                'description': 'ACRY→CHLO starting from best chlo pose'
            }
        }
    }

    with open(output_dir / "rbfe_prep_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  ACRY-aligned pair (FEP from acry's best pose):")
    print(f"    State A: acry_best    (pose {selected['acry_best']['idx']}, {selected['acry_best']['affinity']:.2f} kcal/mol)")
    print(f"    State B: chlo_to_acry (pose {selected['chlo_to_acry']['idx']}, RMSD={selected['chlo_to_acry']['rmsd']:.2f} Å)")

    print(f"\n  CHLO-aligned pair (FEP from chlo's best pose):")
    print(f"    State A: acry_to_chlo (pose {selected['acry_to_chlo']['idx']}, RMSD={selected['acry_to_chlo']['rmsd']:.2f} Å)")
    print(f"    State B: chlo_best    (pose {selected['chlo_best']['idx']}, {selected['chlo_best']['affinity']:.2f} kcal/mol)")

    print(f"\n  Output files in: {output_dir}/")
    print(f"    complexes/     - 4 complex PDBs")
    if not args.skip_param:
        print(f"    params/acry/   - Acrylamide parameterization")
        print(f"    params/chlo/   - Chloroacetamide parameterization")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
  For each system, run b05 and b06 to create MD-ready structures:

  # For acry systems (acry_best and acry_to_chlo):
  python b05_create_cyl_residue.py \\
      --adduct-mol2 {output_dir}/params/acry/adduct.acpype/adduct_NEW.mol2 \\
      --acpype-itp {output_dir}/params/acry/adduct.acpype/adduct_GMX.itp \\
      --residue-name CYA \\
      --output-dir {output_dir}/md_prep

  python b06_assemble_covalent_complex.py \\
      --complex {output_dir}/complexes/complex_acry_best.pdb \\
      --ligand-mol2 {output_dir}/params/acry/adduct.acpype/adduct_NEW.mol2 \\
      --cyl-meta {output_dir}/md_prep/cya_residue/cya_meta.json \\
      --cys-resid {cys_resid} \\
      --cyl-resname CYA \\
      --output {output_dir}/md_prep/complex_acry_best_cya.pdb

  # For chlo systems (chlo_best and chlo_to_acry):
  python b05_create_cyl_residue.py \\
      --adduct-mol2 {output_dir}/params/chlo/adduct.acpype/adduct_NEW.mol2 \\
      --acpype-itp {output_dir}/params/chlo/adduct.acpype/adduct_GMX.itp \\
      --residue-name CYC \\
      --output-dir {output_dir}/md_prep --skip-ff-copy

  # Then run pdb2gmx + solvation + EM for each system
""".format(output_dir=output_dir, cys_resid=args.cys_resid))

    return 0


if __name__ == '__main__':
    sys.exit(main())
