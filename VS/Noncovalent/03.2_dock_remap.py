#!/usr/bin/env python3
"""
Export best pose from Vina PDBQT and remap coordinates to original SDF atom order and H count.

This produces an SDF that:
- Has the exact atoms (including hydrogens) and ordering of the original SDF.
- Uses heavy-atom coordinates from the best docked pose (receptor frame).
- Places hydrogens by copying docked H coordinates when possible, else preserves original offsets.

Requirements:
- RDKit installed
- mk_export.py available on PATH (Meeko CLI compatible with: mk_export.py INPUT.pdbqt -s OUTPUT.sdf [-k])

Usage:
    python dock_remap_one.py --original ORIGINAL.sdf \
                             --pdbqt VINA_DOCKED.pdbqt \
                             --out REINDEXED_OUTPUT.sdf \
                             [--keep-names] [--mk-export MK_EXPORT_PATH]

Example:
    python dock_remap_one.py --original ../Outputs/conformers/NonCov/Inhib_42.sdf \
                             --pdbqt 4CXA_Inhib_45_docked.pdbqt \
                             --out 4CXA_Inhib_45_docked_reindexed.sdf \
                             --keep-names
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from rdkit import Chem


def run_cmd(cmd, desc):
    print(f"  {desc}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  ERROR: {desc} failed")
        print(f"  STDERR: {res.stderr}")
        sys.exit(1)
    print(f"  ✓ {desc} complete")


def get_neighbors(mol):
    return {i: [nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors()]
            for i in range(mol.GetNumAtoms())}


def remap_coordinates_to_original(orig_sdf, docked_sdf, out_sdf):
    # Load original (keep explicit Hs and order)
    orig = Chem.SDMolSupplier(str(orig_sdf), removeHs=False)[0]
    if orig is None:
        raise ValueError(f"Failed to read original SDF: {orig_sdf}")

    # Load docked best pose SDF (exported via mk_export.py)
    dock = Chem.SDMolSupplier(str(docked_sdf), removeHs=False)[0]
    if dock is None:
        raise ValueError(f"Failed to read docked SDF: {docked_sdf}")

    # Build heavy-only molecules
    orig_heavy = Chem.RemoveHs(orig)
    dock_heavy = Chem.RemoveHs(dock)

    # Compute a heavy-atom mapping: indices in dock_heavy corresponding to orig_heavy order
    match = dock_heavy.GetSubstructMatch(orig_heavy)
    if not match or len(match) != orig_heavy.GetNumAtoms():
        # Fallback: try SMARTS-based matching
        patt = Chem.MolFromSmarts(Chem.MolToSmarts(orig_heavy))
        match = dock_heavy.GetSubstructMatch(patt)
        if not match or len(match) != orig_heavy.GetNumAtoms():
            raise ValueError("Could not map docked pose to original topology. Check protonation/tautomer.")

    # Prepare conformers
    dock_conf = dock.GetConformer()
    orig_conf = orig.GetConformer()

    # Output molecule and conformer (same atom count/order as original)
    out = Chem.Mol(orig)
    out_conf = Chem.Conformer(out.GetNumAtoms())

    # Indices lists
    orig_full_heavy_idx = [i for i, a in enumerate(orig.GetAtoms()) if a.GetAtomicNum() != 1]
    orig_full_H_idx = [i for i, a in enumerate(orig.GetAtoms()) if a.GetAtomicNum() == 1]
    dock_full_heavy_idx = [i for i, a in enumerate(dock.GetAtoms()) if a.GetAtomicNum() != 1]

    # match is tuple length N; match[pos] = dock_heavy index corresponding to orig_heavy index 'pos'
    # 'pos' corresponds to the position in orig_full_heavy_idx.
    # Copy heavy-atom coordinates directly from docked SDF (receptor frame)
    for pos, o_full_idx in enumerate(orig_full_heavy_idx):
        d_heavy_idx = match[pos]  # index in dock_heavy
        d_full_idx = dock_full_heavy_idx[d_heavy_idx]  # corresponding heavy atom index in full dock
        out_conf.SetAtomPosition(o_full_idx, dock_conf.GetAtomPosition(d_full_idx))

    # Neighbor maps
    orig_neighbors = get_neighbors(orig)
    dock_neighbors = get_neighbors(dock)

    # Build map from original heavy atom index -> dock full heavy atom index
    orig_heavy_to_dock_heavy_full = {
        o_full_idx: dock_full_heavy_idx[match[pos]]
        for pos, o_full_idx in enumerate(orig_full_heavy_idx)
    }

    # For hydrogens: try to map Hs attached to each heavy atom by index order.
    # If counts differ, fall back to preserving original offset relative to the moved heavy atom.
    for o_H in orig_full_H_idx:
        # original heavy neighbor
        o_heavy_nei = next((n for n in orig_neighbors[o_H] if orig.GetAtomWithIdx(n).GetAtomicNum() != 1), None)
        if o_heavy_nei is None:
            # keep original position
            out_conf.SetAtomPosition(o_H, orig_conf.GetAtomPosition(o_H))
            continue

        d_full_heavy = orig_heavy_to_dock_heavy_full.get(o_heavy_nei, None)
        if d_full_heavy is None:
            # fallback: original offset
            orig_H_pos = orig_conf.GetAtomPosition(o_H)
            orig_Hvy_pos = orig_conf.GetAtomPosition(o_heavy_nei)
            vec = orig_H_pos - orig_Hvy_pos
            new_Hvy_pos = out_conf.GetAtomPosition(o_heavy_nei)
            out_conf.SetAtomPosition(o_H, new_Hvy_pos + vec)
            continue

        # Hydrogens attached to dock heavy
        dock_Hs = [n for n in dock_neighbors[d_full_heavy] if dock.GetAtomWithIdx(n).GetAtomicNum() == 1]
        # Hydrogens attached to original heavy
        orig_Hs = [n for n in orig_neighbors[o_heavy_nei] if orig.GetAtomWithIdx(n).GetAtomicNum() == 1]

        if len(dock_Hs) == len(orig_Hs) and len(orig_Hs) > 0:
            dock_Hs_sorted = sorted(dock_Hs)
            orig_Hs_sorted = sorted(orig_Hs)
            # position of current original H within its group
            try:
                pos_in_group = orig_Hs_sorted.index(o_H)
            except ValueError:
                pos_in_group = None
            if pos_in_group is not None:
                d_H_full_idx = dock_Hs_sorted[pos_in_group]
                out_conf.SetAtomPosition(o_H, dock_conf.GetAtomPosition(d_H_full_idx))
                continue

        # Fallback: preserve original offset relative to the (now dock-positioned) heavy atom
        orig_H_pos = orig_conf.GetAtomPosition(o_H)
        orig_Hvy_pos = orig_conf.GetAtomPosition(o_heavy_nei)
        vec = orig_H_pos - orig_Hvy_pos
        new_Hvy_pos = out_conf.GetAtomPosition(o_heavy_nei)
        out_conf.SetAtomPosition(o_H, new_Hvy_pos + vec)

    # Write output SDF
    out.RemoveAllConformers()
    out.AddConformer(out_conf, assignId=True)
    w = Chem.SDWriter(str(out_sdf))
    w.write(out)
    w.close()
    print(f"✓ Wrote reindexed SDF in docked pose: {out_sdf}")


def main():
    ap = argparse.ArgumentParser(description="Export best pose from PDBQT and remap to original SDF order/H count.")
    ap.add_argument("--original", required=True, help="Original ligand SDF (desired atom order and hydrogens)")
    ap.add_argument("--pdbqt", required=True, help="Vina docked ligand PDBQT (output from Vina)")
    ap.add_argument("--out", required=True, help="Output SDF with original atom order and docked coordinates")
    ap.add_argument("--keep-names", action="store_true", help="Pass -k to mk_export.py to keep atom names")
    ap.add_argument("--mk-export", default="mk_export.py", help="Path to mk_export.py (default: mk_export.py)")
    args = ap.parse_args()

    original = Path(args.original)
    pdbqt = Path(args.pdbqt)
    out_sdf = Path(args.out)

    if not original.exists():
        print(f"ERROR: Original SDF not found: {original}")
        sys.exit(1)
    if not pdbqt.exists():
        print(f"ERROR: PDBQT not found: {pdbqt}")
        sys.exit(1)

    print("=== Export best pose and remap to original ===")
    print(f"Original SDF: {original}")
    print(f"Docked PDBQT: {pdbqt}")
    print(f"Output SDF:   {out_sdf}")

    # Create a temporary file for the exported best pose SDF
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_sdf = Path(tmpdir) / "best_pose.sdf"
        cmd = [args.mk_export, str(pdbqt), "-s", str(tmp_sdf)]
        if args.keep_names:
            cmd += ["-k"]
        run_cmd(cmd, "Export best pose from PDBQT via mk_export.py")

        # Remap docked coordinates to original order/H count
        remap_coordinates_to_original(original, tmp_sdf, out_sdf)


if __name__ == "__main__":
    main()
