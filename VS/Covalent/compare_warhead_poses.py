#!/usr/bin/env python3
"""
compare_warhead_poses.py

Simple script to find overlapping poses between acrylamide and chloroacetamide docking results.

Given two SDF files with docked poses:
1. Find the chlo pose with minimum scaffold RMSD to acry's best pose
2. Find the acry pose with minimum scaffold RMSD to chlo's best pose

Usage:
    python compare_warhead_poses.py \\
        --acry docked_poses_acry.sdf \\
        --chlo docked_poses_chlo.sdf \\
        --output_dir selected_poses
"""

import argparse
import sys
from pathlib import Path
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
except ImportError:
    print("ERROR: RDKit required. Install via conda/pip.")
    sys.exit(1)


def load_poses(sdf_path: Path):
    """Load all poses from SDF file."""
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    poses = [mol for mol in suppl if mol is not None]
    print(f"  Loaded {len(poses)} poses from {sdf_path.name}")
    return poses


def get_vina_affinity(mol) -> float:
    """Get Vina affinity from molecule properties."""
    for prop in ['minimizedAffinity', 'affinity']:
        if mol.HasProp(prop):
            try:
                return float(mol.GetProp(prop))
            except:
                pass
    return 0.0


def find_mcs_mapping(mol1, mol2):
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


def calculate_rmsd(mol1, mol2, atom_mapping):
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


def write_sdf(mol, path, name=None):
    """Write molecule to SDF file."""
    writer = Chem.SDWriter(str(path))
    if name:
        mol.SetProp('_Name', name)
    writer.write(mol)
    writer.close()


def main():
    parser = argparse.ArgumentParser(description='Find overlapping poses between acry and chlo')
    parser.add_argument('--acry', required=True, help='Acrylamide docked poses SDF')
    parser.add_argument('--chlo', required=True, help='Chloroacetamide docked poses SDF')
    parser.add_argument('--output_dir', default='selected_poses', help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("WARHEAD POSE COMPARISON")
    print("=" * 60)

    # Load poses
    print("\n[1] Loading poses...")
    acry_poses = load_poses(Path(args.acry))
    chlo_poses = load_poses(Path(args.chlo))

    if not acry_poses or not chlo_poses:
        print("ERROR: No poses found")
        return 1

    # Find MCS mapping (compute once, reuse for all comparisons)
    print("\n[2] Finding Maximum Common Substructure (scaffold)...")
    mcs_mapping = find_mcs_mapping(acry_poses[0], chlo_poses[0])

    if mcs_mapping:
        atoms = [acry_poses[0].GetAtomWithIdx(i).GetSymbol() for i, _ in mcs_mapping]
        print(f"  MCS: {len(mcs_mapping)} atoms ({' '.join(atoms)})")
    else:
        print("  WARNING: MCS failed, results may be unreliable")

    # Get best poses by Vina affinity
    print("\n[3] Finding best poses by Vina affinity...")
    acry_affinities = [get_vina_affinity(m) for m in acry_poses]
    chlo_affinities = [get_vina_affinity(m) for m in chlo_poses]

    best_acry_idx = int(np.argmin(acry_affinities))
    best_chlo_idx = int(np.argmin(chlo_affinities))

    print(f"  Best ACRY: pose {best_acry_idx} ({acry_affinities[best_acry_idx]:.2f} kcal/mol)")
    print(f"  Best CHLO: pose {best_chlo_idx} ({chlo_affinities[best_chlo_idx]:.2f} kcal/mol)")

    # Calculate RMSD from best acry to all chlo poses
    print("\n[4] Calculating scaffold RMSD...")
    print(f"\n  RMSDs from best ACRY (pose {best_acry_idx}) to all CHLO poses:")

    rmsds_to_acry = []
    for j, chlo_mol in enumerate(chlo_poses):
        rmsd = calculate_rmsd(acry_poses[best_acry_idx], chlo_mol, mcs_mapping)
        rmsds_to_acry.append(rmsd)

    # Show all chlo poses sorted by RMSD
    sorted_chlo = np.argsort(rmsds_to_acry)
    for rank, j in enumerate(sorted_chlo[:10]):  # Top 10
        print(f"    {rank+1}. CHLO[{j}]: RMSD = {rmsds_to_acry[j]:.2f} Å, affinity = {chlo_affinities[j]:.2f} kcal/mol")

    closest_chlo_idx = sorted_chlo[0]

    # Calculate RMSD from best chlo to all acry poses
    print(f"\n  RMSDs from best CHLO (pose {best_chlo_idx}) to all ACRY poses:")

    rmsds_to_chlo = []
    for i, acry_mol in enumerate(acry_poses):
        rmsd = calculate_rmsd(acry_mol, chlo_poses[best_chlo_idx], mcs_mapping)
        rmsds_to_chlo.append(rmsd)

    # Show all acry poses sorted by RMSD
    sorted_acry = np.argsort(rmsds_to_chlo)
    for rank, i in enumerate(sorted_acry[:10]):  # Top 10
        print(f"    {rank+1}. ACRY[{i}]: RMSD = {rmsds_to_chlo[i]:.2f} Å, affinity = {acry_affinities[i]:.2f} kcal/mol")

    closest_acry_idx = sorted_acry[0]

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n  ACRY-aligned pair (for FEP starting from acry's best pose):")
    print(f"    ACRY: pose {best_acry_idx} (best, {acry_affinities[best_acry_idx]:.2f} kcal/mol)")
    print(f"    CHLO: pose {closest_chlo_idx} (RMSD = {rmsds_to_acry[closest_chlo_idx]:.2f} Å, {chlo_affinities[closest_chlo_idx]:.2f} kcal/mol)")

    print(f"\n  CHLO-aligned pair (for FEP starting from chlo's best pose):")
    print(f"    ACRY: pose {closest_acry_idx} (RMSD = {rmsds_to_chlo[closest_acry_idx]:.2f} Å, {acry_affinities[closest_acry_idx]:.2f} kcal/mol)")
    print(f"    CHLO: pose {best_chlo_idx} (best, {chlo_affinities[best_chlo_idx]:.2f} kcal/mol)")

    # Save selected poses
    print(f"\n  Saving poses to {output_dir}/...")

    write_sdf(acry_poses[best_acry_idx], output_dir / 'acry_best.sdf', 'acry_best')
    write_sdf(chlo_poses[closest_chlo_idx], output_dir / 'chlo_closest_to_acry.sdf', 'chlo_closest_to_acry')
    write_sdf(acry_poses[closest_acry_idx], output_dir / 'acry_closest_to_chlo.sdf', 'acry_closest_to_chlo')
    write_sdf(chlo_poses[best_chlo_idx], output_dir / 'chlo_best.sdf', 'chlo_best')

    print("    acry_best.sdf")
    print("    chlo_closest_to_acry.sdf")
    print("    acry_closest_to_chlo.sdf")
    print("    chlo_best.sdf")

    return 0


if __name__ == '__main__':
    sys.exit(main())
