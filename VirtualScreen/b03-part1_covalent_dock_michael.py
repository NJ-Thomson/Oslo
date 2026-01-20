#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Covalent Docking Workflow for Michael Addition (Acrylamide) - Native GNINA (Covalent Mode)

Highlights:
- Protonate receptor without renumbering residues:
  - Preferred: Reduce (adds H, preserves residue numbering and grouping)
  - Fallback: PDBFixer (pH-aware; also preserves residue grouping)
- Keep target cysteine as thiolate (CYM, no HG/HSG)
- GNINA covalent docking with SMARTS-based beta-carbon identification per pose

Requirements:
  - RDKit
  - GNINA binary
  - Protonation:
      Preferred: Reduce (conda install -c conda-forge reduce)
      Fallback:  PDBFixer + OpenMM (conda install -c conda-forge pdbfixer openmm)
"""

import argparse
import subprocess
import os
import sys
from pathlib import Path
import shutil
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D
except ImportError:
    print("ERROR: RDKit required. Install with: pip install rdkit")
    sys.exit(1)


# --- Protonation helpers (protein) ---

def enforce_cys_thiolate_inline(pdb_in: str, pdb_out: str, chain_id: str, resid: int) -> None:
    """
    Rename the specified cysteine to CYM and remove thiol H (HG/HSG) in-place.
    Preserves residue numbering and chain IDs.
    """
    out_lines = []
    with open(pdb_in, 'r') as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                out_lines.append(line)
                continue
            chn = line[21]
            try:
                resseq = int(line[22:26])
            except Exception:
                resseq = None
            if chn == chain_id and resseq == resid:
                atom_name = line[12:16].strip()
                # Drop thiol hydrogen if present
                if atom_name in ('HG', 'HSG'):
                    continue
                # Rename residue to CYM (thiolate)
                line = line[:17] + 'CYM' + line[20:]
            out_lines.append(line)
    with open(pdb_out, 'w') as w:
        w.writelines(out_lines)


def protonate_with_reduce(pdb_in: str, pdb_out: str, noflip: bool = True) -> bool:
    """
    Add hydrogens using Reduce, preserving residue numbering and grouping H with the same residues.
    Returns True on success.
    """
    reduce_exe = shutil.which("reduce")
    if reduce_exe is None:
        return False
    cmd = [reduce_exe, "-BUILD"]
    if noflip:
        cmd.append("-NOFLIP")
    cmd.append(pdb_in)
    print("    Running Reduce to add hydrogens...")
    print("    Command:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not res.stdout:
        print(f"    Reduce failed:\n{res.stderr}")
        return False
    with open(pdb_out, "w") as f:
        f.write(res.stdout)
    print(f"    Hydrogens added with Reduce: {pdb_out}")
    return True


def protonate_with_pdbfixer(pdb_in: str, pdb_out: str, ph: float = 7.4, keep_hets: bool = True) -> bool:
    """
    Add hydrogens using PDBFixer at given pH. Preserves residues and chain IDs.
    """
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except Exception as e:
        print(f"    PDBFixer not available: {e}")
        return False
    try:
        fixer = PDBFixer(filename=pdb_in)
        if not keep_hets:
            fixer.removeHeterogens(keepWater=False)
        # Keep changes minimal: do not add/remove residues; just atoms and H
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=ph)
        with open(pdb_out, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        print(f"    Hydrogens added with PDBFixer at pH {ph:.2f}: {pdb_out}")
        return True
    except Exception as e:
        print(f"    PDBFixer failed: {e}")
        return False


def protonate_receptor_preserving_resnums(pdb_in: str, pdb_out: str, chain_id: str, resid: int,
                                          ph: float = 7.4, keep_hets: bool = True,
                                          method: str = "reduce") -> bool:
    """
    Preferred: Reduce (preserves residue numbering; no “new residues” for H).
    Fallback: PDBFixer (pH-aware; also preserves residues).
    Keeps the target cysteine as thiolate by renaming to CYM before adding H.
    """
    # 1) Enforce thiolate upfront so the hydrogen builder won't add HG
    tmp_thiol = pdb_out + ".thiol_in.pdb"
    enforce_cys_thiolate_inline(pdb_in, tmp_thiol, chain_id=chain_id, resid=resid)

    # 2) Try the requested method first, then fallback
    ok = False
    if method.lower() == "reduce":
        ok = protonate_with_reduce(tmp_thiol, pdb_out, noflip=True)
        if not ok:
            print("    Reduce failed or not found; falling back to PDBFixer...")
            ok = protonate_with_pdbfixer(tmp_thiol, pdb_out, ph=ph, keep_hets=keep_hets)
    else:
        ok = protonate_with_pdbfixer(tmp_thiol, pdb_out, ph=ph, keep_hets=keep_hets)
        if not ok:
            print("    PDBFixer failed; trying Reduce...")
            ok = protonate_with_reduce(tmp_thiol, pdb_out, noflip=True)

    try:
        os.remove(tmp_thiol)
    except Exception:
        pass

    if not ok:
        print("    ERROR: Could not add hydrogens with Reduce or PDBFixer.")
        return False

    # 3) Safety pass: ensure target cysteine remains thiolate
    tmp_final = pdb_out + ".tmp"
    enforce_cys_thiolate_inline(pdb_out, tmp_final, chain_id=chain_id, resid=resid)
    os.replace(tmp_final, pdb_out)
    return True


# --- Sanitization helpers (ligand) ---

def _try_kekulize(m: Chem.Mol) -> None:
    """
    Try to kekulize; if it fails, clear aromatic flags and re-perceive.
    """
    try:
        Chem.Kekulize(m, clearAromaticFlags=True)
        Chem.SetAromaticity(m)
    except Exception:
        for b in m.GetBonds():
            b.SetIsAromatic(False)
        for a in m.GetAtoms():
            a.SetIsAromatic(False)
        Chem.SanitizeMol(
            m,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES |
                        Chem.SanitizeFlags.SANITIZE_SYMMRINGS |
                        Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
        )
        Chem.SetAromaticity(m)


def sanitize_molecule(m: Chem.Mol, ensure_3d: bool = True, maxIters: int = 500) -> Chem.Mol:
    """
    Standardize a molecule for docking.
    """
    m = Chem.Mol(m)  # copy
    try:
        Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
    except Exception:
        Chem.SanitizeMol(
            m,
            sanitizeOps=(Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        )
    _try_kekulize(m)
    m = Chem.AddHs(m, addCoords=True)
    if ensure_3d and m.GetNumConformers() == 0:
        AllChem.EmbedMolecule(m, AllChem.ETKDGv3())
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=maxIters)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(m, maxIters=maxIters)
        except Exception:
            pass
    Chem.AssignStereochemistry(m, force=True, cleanIt=True)
    return m


def write_sdf_v3000(m: Chem.Mol, path: str) -> None:
    w = Chem.SDWriter(path)
    try:
        w.SetForceV3000(True)
    except Exception:
        pass
    w.write(m)
    w.close()


# --- Geometry helpers ---

def get_cys_geometry(pdb_file, chain, resid):
    """Extract cysteine atom coordinates."""
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
            except Exception:
                continue

            if chain_id == chain and res_num == resid:
                atom_name = line[12:16].strip()
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                coords[atom_name] = np.array([x, y, z], dtype=float)

    if 'SG' not in coords:
        return None
    return coords


def get_receptor_coords(pdb_file, exclude_chain=None, exclude_resid=None):
    """Get all heavy atom coordinates from receptor (for clash estimation)."""
    coords = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            atom_name = line[12:16].strip()
            if atom_name[0] == 'H' or (len(atom_name) > 1 and atom_name[1] == 'H'):
                continue
            if exclude_chain and exclude_resid:
                chain_id = line[21]
                try:
                    res_num = int(line[22:26].strip())
                except Exception:
                    res_num = 0
                if chain_id == exclude_chain and res_num == exclude_resid:
                    continue
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            coords.append([x, y, z])
    return np.array(coords, dtype=float)


def angle_deg(a, b, c):
    """Return angle at b (a-b-c) in degrees."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


# --- Acrylamide beta-carbon detection ---

def find_acrylamide_beta_carbon(mol):
    """
    Find the beta carbon of an acrylamide warhead.
    Acrylamide pattern: N-C(=O)-C=C
                             alpha beta
    Returns: (alpha_idx, beta_idx) or None
    """
    patterns = [
        '[NX3][CX3](=O)[CX3]=[CH2]',      # Terminal acrylamide
        '[NX3][CX3](=O)[CH]=[CH2]',       # With H on alpha
        '[NX3][CX3](=O)[CX3]=[CX3]',      # Substituted
        '[CX3](=O)[CX3]=[CH2]',           # Without explicit N
    ]
    for smarts in patterns:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            match = matches[0]
            alpha_idx = match[-2]
            beta_idx = match[-1]
            bond = mol.GetBondBetweenAtoms(alpha_idx, beta_idx)
            if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                return alpha_idx, beta_idx
    return None


def covalent_beta_smarts_candidates():
    """
    Candidates centered on the acrylamide beta carbon (vinylic carbon NOT the carbonyl carbon).
    Each SMARTS is written so the 'current atom' is the beta carbon.
    """
    return [
        '[$([CH2]=[CX3][CX3](=O)[NX3])]',
        '[$([CX3;!$(C=O)]=[CX3][CX3](=O)[NX3])]',
        '[$([CH2]=[CX3][CX3](=O)[N])]',
        '[$([CX3;!$(C=O)]=[CX3][CX3](=O)[N])]',
        '[$([CH2]=[CX3][CX3](=O)O)]',
        '[$([CX3;!$(C=O)]=[CX3][CX3](=O)O)]',
    ]


def pick_beta_covalent_smarts(mol, beta_idx):
    """
    Choose a covalent_lig_atom_pattern SMARTS that actually matches the known beta_idx.
    """
    for smarts in covalent_beta_smarts_candidates():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            continue
        matched_idxs = [m[0] if isinstance(m, tuple) else m for m in matches]
        if beta_idx in matched_idxs:
            print(f"    Selected covalent SMART: {smarts} (matches beta_idx={beta_idx})")
            return smarts
    fallback = '[$([CX3;!$(C=O)]=[CX3][CX3](=O)[NX3])]'
    print(f"    WARNING: None of the covalent SMARTS candidates matched; fallback: {fallback}")
    return fallback


# --- Positioning helper (optional) ---

def preposition_ligand_beta_at_SG(mol, beta_idx, sg_pos, receptor_coords=None, cb_pos=None, n_rotations=24):
    """
    Translate ligand so beta carbon sits at SG; optionally rotate around SG->CB axis to minimize clashes.
    """
    conf = mol.GetConformer()
    beta_pos = np.array(conf.GetAtomPosition(beta_idx), dtype=float)
    translation = sg_pos - beta_pos

    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(pos.x + translation[0],
                                        pos.y + translation[1],
                                        pos.z + translation[2]))

    if receptor_coords is not None and cb_pos is not None and n_rotations > 0:
        axis = cb_pos - sg_pos
        axis_norm = np.linalg.norm(axis)
        axis = axis / axis_norm if axis_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        best_clashes, best_angle = float('inf'), 0.0

        heavy_idxs = [i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
        for angle in np.linspace(0, 2*np.pi, n_rotations, endpoint=False):
            cos_a = np.cos(angle); sin_a = np.sin(angle)
            clashes = 0
            for i in heavy_idxs:
                if i == beta_idx:
                    continue
                p = np.array(conf.GetAtomPosition(i), dtype=float)
                rel = p - sg_pos
                rot = (rel * cos_a +
                       np.cross(axis, rel) * sin_a +
                       axis * np.dot(axis, rel) * (1 - cos_a))
                coord = rot + sg_pos
                d = np.linalg.norm(receptor_coords - coord, axis=1)
                clashes += np.sum(d < 2.0)
            if clashes < best_clashes:
                best_clashes, best_angle = clashes, angle

        # Apply best rotation
        cos_a = np.cos(best_angle); sin_a = np.sin(best_angle)
        for i in range(mol.GetNumAtoms()):
            if i == beta_idx:
                continue
            p = np.array(conf.GetAtomPosition(i), dtype=float)
            rel = p - sg_pos
            rot = (rel * cos_a +
                   np.cross(axis, rel) * sin_a +
                   axis * np.dot(axis, rel) * (1 - cos_a))
            conf.SetAtomPosition(i, Point3D(*(rot + sg_pos)))

        print(f"    Pre-position rotation: {np.degrees(best_angle):.1f} deg, {best_clashes} clashes")

    return mol


# --- GNINA runner (native, covalent mode) ---

def run_gnina_covalent(receptor, ligand, output, center, box_size=18,
                       exhaustiveness=16, num_modes=20, local_only=False,
                       gpu=False, device=None,
                       cnn_scoring='refine', cnn_model='crossdock_default2018',
                       seed=42,
                       covalent_rec_atom=None,
                       covalent_lig_atom_pattern=None,
                       covalent_lig_atom_position=None,
                       covalent_optimize_lig=True,
                       gnina_path="~/software/gnina/build/bin/gnina"):
    exe = os.path.expanduser(gnina_path)
    if not os.path.isfile(exe):
        exe = 'gnina'  # fallback to PATH

    cmd = [exe, '-r', receptor, '-l', ligand, '-o', output]

    if gpu:
        if device is not None:
            cmd.extend(['--device', str(device)])
    else:
        cmd.append('--no_gpu')

    if local_only:
        cmd.extend(['--local_only', '--minimize', '--minimize_iters', '200'])
    else:
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

    cmd.extend([
        '--cnn_scoring', cnn_scoring,
        '--cnn', cnn_model,
        '--seed', str(seed),
    ])

    if covalent_rec_atom:
        cmd.extend(['--covalent_rec_atom', covalent_rec_atom])
    if covalent_lig_atom_pattern:
        cmd.extend(['--covalent_lig_atom_pattern', covalent_lig_atom_pattern])
    if covalent_lig_atom_position is not None:
        x, y, z = covalent_lig_atom_position
        cmd.extend(['--covalent_lig_atom_position', f'{x:.3f},{y:.3f},{z:.3f}'])
    if covalent_optimize_lig:
        cmd.append('--covalent_optimize_lig')

    print("\n[5] Running GNINA (covalent mode)...")
    print(f"    Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
        return True

    stderr = (result.stderr or "") + "\n" + (result.stdout or "")
    print(f"    GNINA stderr/stdout:\n{stderr}")

    # CPU fallback on CUDA errors
    s = stderr.lower()
    kernel_err = "no kernel image is available for execution on the device" in s
    device_err = ("cuda error" in s) or ("unsupported gpu architecture" in s)
    if gpu and (kernel_err or device_err):
        print("    Detected CUDA/device issue. Retrying with CPU (--no_gpu)...")
        cmd = [exe, '-r', receptor, '-l', ligand, '-o', output]
        if local_only:
            cmd.extend(['--local_only', '--minimize', '--minimize_iters', '200'])
        else:
            cmd.extend([
                '--center_x', str(center[0]), '--center_y', str(center[1]), '--center_z', str(center[2]),
                '--size_x', str(box_size), '--size_y', str(box_size), '--size_z', str(box_size),
                '--exhaustiveness', str(exhaustiveness), '--num_modes', str(num_modes),
            ])
        cmd.extend(['--cnn_scoring', cnn_scoring, '--cnn', cnn_model, '--seed', str(seed), '--no_gpu'])
        if covalent_rec_atom:
            cmd.extend(['--covalent_rec_atom', covalent_rec_atom])
        if covalent_lig_atom_pattern:
            cmd.extend(['--covalent_lig_atom_pattern', covalent_lig_atom_pattern])
        if covalent_lig_atom_position is not None:
            x, y, z = covalent_lig_atom_position
            cmd.extend(['--covalent_lig_atom_position', f'{x:.3f},{y:.3f},{z:.3f}'])
        if covalent_optimize_lig:
            cmd.append('--covalent_optimize_lig')

        print(f"    Command: {' '.join(cmd)}")
        retry = subprocess.run(cmd, capture_output=True, text=True)
        if retry.returncode == 0:
            print(retry.stdout)
            return True
        else:
            print(f"    GNINA stderr (retry):\n{retry.stderr}")
            return False

    return False


# --- Filtering and output ---

def find_beta_idx_in_pose(mol, cov_smarts, target_sg_pos):
    """
    For a GNINA output pose, find the ligand beta carbon by SMARTS.
    If multiple matches, pick the one closest to SG.
    """
    patt = Chem.MolFromSmarts(cov_smarts)
    if patt is None:
        return None
    matches = mol.GetSubstructMatches(patt)
    if not matches:
        return None
    conf = mol.GetConformer()
    candidates = [m[0] if isinstance(m, tuple) else m for m in matches]
    best_idx = None
    best_dist = float('inf')
    for idx in candidates:
        p = conf.GetAtomPosition(idx)
        d = np.linalg.norm(np.array([p.x, p.y, p.z]) - target_sg_pos)
        if d < best_dist:
            best_dist = d
            best_idx = idx
    return best_idx


def filter_poses_by_geometry_beta_smart(sdf_file, target_sg_pos, cov_smarts,
                                        max_dist=3.0, cys_cb_pos=None, angle_bounds=(90, 125),
                                        debug=False):
    """
    Filter poses by:
      - betaC (identified per pose by SMARTS) within max_dist of target SG
      - optional angle check: betaC - SG - CB within bounds
    Returns a list of dicts sorted by CNNscore (desc) then distance (asc).
    """
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
    poses = []
    for i, mol in enumerate(suppl):
        if mol is None or mol.GetNumConformers() == 0:
            continue

        beta_idx = find_beta_idx_in_pose(mol, cov_smarts, target_sg_pos)
        if beta_idx is None:
            if debug:
                print(f"    Pose {i}: no beta carbon match by SMARTS; skipping")
            continue

        conf = mol.GetConformer()
        bp = conf.GetAtomPosition(beta_idx)
        beta_pos = np.array([bp.x, bp.y, bp.z], dtype=float)
        dist = np.linalg.norm(beta_pos - target_sg_pos)

        # Scores
        try:
            cnn_score = float(mol.GetProp('CNNscore')) if mol.HasProp('CNNscore') else 0.0
            cnn_affinity = float(mol.GetProp('CNNaffinity')) if mol.HasProp('CNNaffinity') else 0.0
            vina = float(mol.GetProp('minimizedAffinity')) if mol.HasProp('minimizedAffinity') else 0.0
        except Exception:
            cnn_score = cnn_affinity = vina = 0.0

        if dist > max_dist:
            if debug:
                print(f"    Pose {i}: betaC–SG dist {dist:.2f} Å > {max_dist:.2f} Å; skipping (beta_idx={beta_idx})")
            continue

        ang_ok = True
        ang_val = None
        if cys_cb_pos is not None:
            ang_val = angle_deg(beta_pos, target_sg_pos, cys_cb_pos)
            if ang_val is not None:
                lo, hi = angle_bounds
                ang_ok = (lo <= ang_val <= hi)

        if not ang_ok:
            if debug:
                print(f"    Pose {i}: angle {ang_val:.1f}° out of bounds {angle_bounds}; skipping")
            continue

        if debug:
            print(f"    Pose {i}: beta_idx={beta_idx}, dist={dist:.2f} Å, angle={ang_val if ang_val is not None else 'NA'}°, CNN={cnn_score:.3f}")

        poses.append({
            'mol': mol,
            'idx': i,
            'beta_dist': dist,
            'beta_idx': beta_idx,
            'angle': ang_val,
            'cnn_score': cnn_score,
            'cnn_affinity': cnn_affinity,
            'vina': vina
        })

    poses.sort(key=lambda x: (x['cnn_score'], -x['beta_dist']), reverse=True)
    return poses


def write_complex_pdb(receptor_pdb, ligand_mol, output_pdb,
                      cys_chain, cys_resid, beta_idx):
    """
    Write complex PDB with CONECT record for covalent bond (SG to ligand beta-carbon).
    """
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
                res_name = line[17:20].strip()
                atom_name = line[12:16].strip()
                try:
                    res_num = int(line[22:26].strip())
                except Exception:
                    res_num = 0

                # Remove displaced cysteine HG/HSG if present
                if (chain_id == cys_chain and res_num == cys_resid and
                    res_name in ('CYS', 'CYM') and (atom_name.startswith('HG') or atom_name == 'HSG')):
                    continue

                if (chain_id == cys_chain and res_num == cys_resid and atom_name == 'SG'):
                    sg_serial = serial

            lines.append(line)

    lines.append("TER\n")

    conf = ligand_mol.GetConformer()
    beta_serial = None
    serial = last_serial
    for i in range(ligand_mol.GetNumAtoms()):
        atom = ligand_mol.GetAtomWithIdx(i)
        symbol = atom.GetSymbol()

        serial += 1
        pos = conf.GetAtomPosition(i)
        if i == beta_idx:
            beta_serial = serial

        name = f"{symbol}{i+1}"[:4].ljust(4)
        line = (f"HETATM{serial:5d} {name} LIG L   1    "
                f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
                f"  1.00  0.00          {symbol:>2s}\n")
        lines.append(line)

    if sg_serial and beta_serial:
        lines.append(f"CONECT{sg_serial:5d}{beta_serial:5d}\n")
        lines.append(f"CONECT{beta_serial:5d}{sg_serial:5d}\n")

    lines.append("END\n")
    with open(output_pdb, 'w') as f:
        f.writelines(lines)

    return sg_serial is not None and beta_serial is not None


# --- I/O helpers ---

def validate_sdf(path):
    try:
        with open(path, 'r', errors='ignore') as f:
            for line in f:
                if line.strip() == "$$$$":
                    return True
    except Exception:
        return False
    return False


def load_ligand(path_str):
    lig_path = Path(path_str)
    if not lig_path.exists() or lig_path.stat().st_size == 0:
        print(f"ERROR: Ligand file not found or empty: {path_str}")
        return None

    mol = None
    ext = lig_path.suffix.lower()

    if ext in ('.sdf', '.sd') and not validate_sdf(str(lig_path)):
        print("    WARNING: SDF seems to lack a '$$$$' record separator. RDKit may fail to parse.")
        print("             Consider fixing with Open Babel: obabel input.sdf -O fixed.sdf")

    try:
        if ext in ('.sdf', '.sd'):
            suppl = Chem.SDMolSupplier(str(lig_path), removeHs=False, sanitize=True, strictParsing=False)
            for m in suppl:
                if m is not None:
                    mol = m
                    break
        elif ext in ('.mol2', '.mol'):
            mol = Chem.MolFromMolFile(str(lig_path), sanitize=True, removeHs=False)
        elif ext == '.pdb':
            mol = Chem.MolFromPDBFile(str(lig_path), sanitize=True, removeHs=False)
        elif ext in ('.smi', '.txt'):
            with open(lig_path, 'r') as f:
                for line in f:
                    smi = line.strip()
                    if smi:
                        mol = Chem.MolFromSmiles(smi)
                        if mol is not None:
                            mol = Chem.AddHs(mol)
                        break
        else:
            # Try SDF fallback
            suppl = Chem.SDMolSupplier(str(lig_path), removeHs=False, sanitize=True, strictParsing=False)
            for m in suppl:
                if m is not None:
                    mol = m
                    break
            if mol is None:
                mol = Chem.MolFromMolFile(str(lig_path), sanitize=True, removeHs=False)
    except Exception as e:
        print(f"    RDKit parse warning: {e}")

    if mol is None:
        return None

    mol = sanitize_molecule(mol, ensure_3d=True)
    return mol


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="Covalent Docking for Michael Addition (Acrylamide) - Native GNINA (Covalent Mode)"
    )

    parser.add_argument('--ligand', '-l', required=True, help='Ligand file (SDF preferred; also MOL/MOL2/PDB/SMI)')
    parser.add_argument('--receptor', '-r', required=True, help='Receptor PDB')
    parser.add_argument('--cys_chain', default='A', help='Chain ID of target cysteine')
    parser.add_argument('--cys_resid', type=int, required=True, help='Residue number of target cysteine')
    parser.add_argument('--output', '-o', default='covalent_docking', help='Output directory')
    parser.add_argument('--box_size', type=float, default=18, help='Docking box size (Angstrom)')
    parser.add_argument('--exhaustiveness', type=int, default=16, help='GNINA exhaustiveness')
    parser.add_argument('--local_only', action='store_true', help='Only local optimization (no global docking)')

    parser.add_argument('--gnina_path', default='~/software/gnina/build/bin/gnina',
                        help='Path to native GNINA binary (default: ~/software/gnina/build/bin/gnina)')
    parser.add_argument('--cnn_scoring', choices=['none', 'rescore', 'refine'], default='refine',
                        help='GNINA CNN scoring stage (default: refine)')
    parser.add_argument('--cnn_model', default='crossdock_default2018',
                        help='GNINA CNN model (default: crossdock_default2018)')
    parser.add_argument('--num_modes', type=int, default=20, help='Number of GNINA output poses (default: 20)')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed for docking (default: 2024)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU (native GNINA auto-uses GPU unless --no_gpu)')
    parser.add_argument('--device', type=int, default=None, help='GPU device index for native GNINA (e.g., 0)')
    parser.add_argument('--max_beta_dist', type=float, default=2.5,
                        help='Max betaC–SG distance for valid poses (Angstrom)')

    # Protonation options
    parser.add_argument('--protonate_receptor', action='store_true',
                        help='Add hydrogens to receptor before docking')
    parser.add_argument('--ph', type=float, default=7.4,
                        help='pH for receptor protonation (used by PDBFixer)')
    parser.add_argument('--keep_hets', action='store_true',
                        help='When using PDBFixer, keep heterogens (ligands/ions).')
    parser.add_argument('--prot_method', choices=['reduce', 'pdbfixer'], default='reduce',
                        help='Method to add hydrogens without renumbering residues (default: reduce)')

    args = parser.parse_args()

    print("=" * 60)
    print("COVALENT DOCKING - MICHAEL ADDITION (Native GNINA, Covalent Flags)")
    print("=" * 60)

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    # Optional: protonate receptor (adds H in-place; preserves residue numbering; keeps target Cys as thiolate)
    receptor_for_docking = args.receptor
    if args.protonate_receptor:
        print("\n[0] Protonating receptor (preserve residue numbering; thiolate Cys)...")
        prot_pdb = str(outdir / f"receptor_protonated_pH{args.ph:.1f}_{args.prot_method}.pdb")
        ok = protonate_receptor_preserving_resnums(
            args.receptor, prot_pdb,
            chain_id=args.cys_chain, resid=args.cys_resid,
            ph=args.ph, keep_hets=args.keep_hets, method=args.prot_method
        )
        if not ok:
            print("    WARNING: Protonation failed; proceeding with original receptor")
        else:
            receptor_for_docking = prot_pdb
            print(f"    Protonated receptor: {receptor_for_docking}")

    # Step 1: Load cysteine geometry (from protonated receptor if generated)
    print("\n[1] Loading target cysteine...")
    cys_coords = get_cys_geometry(receptor_for_docking, args.cys_chain, args.cys_resid)
    if cys_coords is None:
        print(f"ERROR: Cys {args.cys_chain}:{args.cys_resid} not found or SG missing")
        sys.exit(1)

    sg_pos = cys_coords['SG']
    cb_pos = cys_coords.get('CB', sg_pos + np.array([1.5, 0, 0], dtype=float))
    print(f"    Cys {args.cys_chain}:{args.cys_resid}")
    print(f"    SG: ({sg_pos[0]:.2f}, {sg_pos[1]:.2f}, {sg_pos[2]:.2f})")

    # Step 2: Load ligand
    print("\n[2] Loading ligand...")
    mol = load_ligand(args.ligand)
    if mol is None:
        print("ERROR: Could not parse ligand.")
        sys.exit(1)
    print(f"    Loaded: {mol.GetNumAtoms()} atoms")
    try:
        print(f"    SMILES: {Chem.MolToSmiles(Chem.RemoveHs(mol))}")
    except Exception:
        pass

    # Step 3: Identify acrylamide beta carbon
    print("\n[3] Identifying acrylamide beta carbon...")
    res = find_acrylamide_beta_carbon(mol)
    if res is None:
        print("ERROR: Acrylamide warhead not found (N-C(=O)-C=C).")
        sys.exit(1)
    alpha_idx, beta_idx = res
    print(f"    Acrylamide alpha C={alpha_idx}, beta C={beta_idx}")

    # Step 4: Optional pre-positioning (betaC at SG)
    print("\n[4] Optional pre-positioning (betaC at SG)...")
    receptor_coords = get_receptor_coords(receptor_for_docking, args.cys_chain, args.cys_resid)
    mol = preposition_ligand_beta_at_SG(mol, beta_idx, sg_pos, receptor_coords, cb_pos, n_rotations=24)

    positioned_sdf = str(outdir / "ligand_positioned.sdf")
    write_sdf_v3000(mol, positioned_sdf)
    print(f"    Saved: {positioned_sdf}")
    conf = mol.GetConformer()
    beta_pos_final = np.array(conf.GetAtomPosition(beta_idx), dtype=float)
    print(f"    betaC position: ({beta_pos_final[0]:.2f}, {beta_pos_final[1]:.2f}, {beta_pos_final[2]:.2f})")
    print(f"    Distance betaC–SG: {np.linalg.norm(beta_pos_final - sg_pos):.3f} A")

    # Step 5: GNINA covalent docking (native)
    print("\n[5] GNINA covalent settings...")
    docked_sdf = str(outdir / "docked_poses.sdf")

    cov_rec_atom = f"{args.cys_chain}:{args.cys_resid}:SG"
    cov_lig_smarts = pick_beta_covalent_smarts(mol, beta_idx)

    # Debug: list which atoms the SMARTS matches
    try:
        patt_dbg = Chem.MolFromSmarts(cov_lig_smarts)
        matches_dbg = mol.GetSubstructMatches(patt_dbg)
        if matches_dbg:
            matched = [m[0] if isinstance(m, tuple) else m for m in matches_dbg]
            print(f"    SMARTS matches ligand atom indices: {matched} (beta_idx={beta_idx})")
        else:
            print("    SMARTS matched no atoms in the input ligand (unexpected).")
    except Exception:
        pass

    success = run_gnina_covalent(
        receptor=receptor_for_docking,
        ligand=positioned_sdf,
        output=docked_sdf,
        center=sg_pos,
        box_size=args.box_size,
        exhaustiveness=args.exhaustiveness,
        num_modes=args.num_modes,
        local_only=args.local_only,
        gpu=args.gpu,
        device=args.device,
        cnn_scoring=args.cnn_scoring,
        cnn_model=args.cnn_model,
        seed=args.seed,
        covalent_rec_atom=cov_rec_atom,
        covalent_lig_atom_pattern=cov_lig_smarts,
        covalent_lig_atom_position=tuple(sg_pos),  # initial placement hint
        covalent_optimize_lig=True,
        gnina_path=args.gnina_path
    )

    if not success:
        print("WARNING: GNINA failed, using pre-positioned ligand")
        import shutil as _sh
        _sh.copy(positioned_sdf, docked_sdf)

    # Step 6: Filter poses by covalent geometry (per-pose atom identification)
    print("\n[6] Filtering poses by covalent geometry (betaC–SG distance + angle)...")
    good_poses = filter_poses_by_geometry_beta_smart(
        docked_sdf, sg_pos, cov_smarts=cov_lig_smarts,
        max_dist=args.max_beta_dist,
        cys_cb_pos=cb_pos,
        angle_bounds=(90, 125),
        debug=False
    )

    if not good_poses:
        print(f"    WARNING: No poses with betaC within {args.max_beta_dist} A of SG and passing angle filter")
        print("    Using pre-positioned ligand as fallback")
        good_poses = [{
            'mol': mol,
            'idx': 0,
            'beta_dist': 0.0,
            'angle': None,
            'cnn_score': 0.0,
            'cnn_affinity': 0.0,
            'vina': 0.0,
            'beta_idx': beta_idx
        }]

    print(f"    Found {len(good_poses)} valid poses:")
    for i, p in enumerate(good_poses[:5]):
        ang_str = f", angle: {p['angle']:.1f}°" if p.get('angle') is not None else ""
        print(f"      {i+1}. betaC–SG dist: {p['beta_dist']:.2f} A{ang_str}, "
              f"CNN: {p['cnn_score']:.3f}, affinity: {p['cnn_affinity']:.2f}")

    # Save best pose
    best = good_poses[0]
    best_sdf = str(outdir / "best_pose.sdf")
    write_sdf_v3000(best['mol'], best_sdf)
    print(f"\n    Best pose: {best_sdf}")

    # Step 7: Write complex PDB with CONECT (use the beta idx from the chosen pose if available)
    print("\n[7] Writing complex PDB...")
    complex_pdb = str(outdir / "complex.pdb")
    best_beta_idx = best.get('beta_idx', beta_idx)
    success_complex = write_complex_pdb(
        receptor_for_docking, best['mol'], complex_pdb,
        args.cys_chain, args.cys_resid, best_beta_idx
    )

    if success_complex:
        print(f"    Complex with CONECT: {complex_pdb}")
    else:
        print(f"    WARNING: Could not add CONECT records")

    # Save all good poses
    if len(good_poses) > 1:
        all_good_sdf = str(outdir / "good_poses.sdf")
        w = Chem.SDWriter(all_good_sdf)
        try:
            w.SetForceV3000(True)
        except Exception:
            pass
        for p in good_poses:
            w.write(p['mol'])
        w.close()
        print(f"    All valid poses: {all_good_sdf}")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {outdir}/")
    print(f"\nVisualize:")
    print(f"  pymol {receptor_for_docking} {best_sdf}")
    print("The complex.pdb has a CONECT record between Cys SG and the ligand beta carbon.")
    print("Next: set up MD with a covalent topology for the complex.")


if __name__ == "__main__":
    main()
