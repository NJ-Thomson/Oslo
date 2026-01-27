#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Covalent docking workflow - Supports acrylamide (Michael addition) and chloroacetamide (SN2)

Supported warheads:
  - acrylamide:      N-C(=O)-C=C  -> Michael addition at β-carbon
  - chloroacetamide: N-C(=O)-CH2-Cl -> SN2 displacement at α-carbon

For acrylamide:
  - Detects acrylamide (N-C(=O)-C=C) warhead
  - Converts terminal C=C -> C-C (Michael adduct), preserving heavy-atom coords
  - The β-carbon (terminal vinyl CH2) becomes the covalent attachment point

For chloroacetamide:
  - Detects chloroacetamide (N-C(=O)-CH2-Cl) warhead
  - Removes Cl leaving group for docking
  - The α-carbon (CH2 bearing Cl) becomes the covalent attachment point

Common:
  - Ensures hydrogens are explicit and have coordinates for GNINA
  - Protonates receptor and preserves target Cys as thiolate (CYM)
  - Marks the chosen reactive carbon with isotope=13 and uses SMARTS '[13C]' so GNINA picks exactly that atom

Scoring (GNINA 1.3):
  - CNN scoring is NOT recommended for covalent docking (CNN was not trained on
    covalent complexes - see McNutt et al. J Cheminform 2025)
  - Poses are ranked by Vina affinity (kcal/mol, more negative = better)
  - Geometry filters: distance to Cys-SG and C-SG-CB angle
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
except Exception:
    print("ERROR: RDKit required. Install via conda/pip (rdkit).")
    sys.exit(1)


# ---------- Protonation helpers ----------
def enforce_cys_thiolate_inline(pdb_in: str, pdb_out: str, chain_id: str, resid: int) -> None:
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
                # remove any hydrogen on SG
                if atom_name in ('HG', 'HSG'):
                    continue
                line = line[:17] + 'CYM' + line[20:]
            out_lines.append(line)
    with open(pdb_out, 'w') as w:
        w.writelines(out_lines)


def protonate_with_reduce(pdb_in: str, pdb_out: str, noflip: bool = True) -> bool:
    reduce_exe = shutil.which("reduce")
    if reduce_exe is None:
        return False
    cmd = [reduce_exe, "-BUILD"]
    if noflip:
        cmd.append("-NOFLIP")
    cmd.append(pdb_in)
    print("    Running Reduce to add hydrogens...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not res.stdout:
        print(f"    Reduce failed:\n{res.stderr}")
        return False
    with open(pdb_out, "w") as f:
        f.write(res.stdout)
    print(f"    Hydrogens added with Reduce: {pdb_out}")
    return True


def protonate_with_pdbfixer(pdb_in: str, pdb_out: str, ph: float = 7.4, keep_hets: bool = True) -> bool:
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
    tmp_in = pdb_out + ".thiol_in.pdb"
    enforce_cys_thiolate_inline(pdb_in, tmp_in, chain_id=chain_id, resid=resid)

    ok = False
    if method.lower() == "reduce":
        ok = protonate_with_reduce(tmp_in, pdb_out, noflip=True)
        if not ok:
            print("    Reduce failed; falling back to PDBFixer...")
            ok = protonate_with_pdbfixer(tmp_in, pdb_out, ph=ph, keep_hets=keep_hets)
    else:
        ok = protonate_with_pdbfixer(tmp_in, pdb_out, ph=ph, keep_hets=keep_hets)
        if not ok:
            print("    PDBFixer failed; trying Reduce...")
            ok = protonate_with_reduce(tmp_in, pdb_out, noflip=True)

    try:
        os.remove(tmp_in)
    except Exception:
        pass

    if not ok:
        print("    ERROR: Could not add hydrogens.")
        return False

    # Re-apply CYM to the output to ensure cysteine remains thiolate
    tmp_final = pdb_out + ".tmp"
    enforce_cys_thiolate_inline(pdb_out, tmp_final, chain_id=chain_id, resid=resid)
    os.replace(tmp_final, pdb_out)
    return True


# ---------- Ligand helpers ----------
def _try_kekulize(m: Chem.Mol) -> None:
    try:
        Chem.Kekulize(m, clearAromaticFlags=True)
        Chem.SetAromaticity(m)
    except Exception:
        for b in m.GetBonds():
            b.SetIsAromatic(False)
        for a in m.GetAtoms():
            a.SetIsAromatic(False)
        Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES |
                        Chem.SanitizeFlags.SANITIZE_SYMMRINGS |
                        Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
        Chem.SetAromaticity(m)


def sanitize_molecule(m: Chem.Mol, ensure_3d: bool = True, maxIters: int = 500) -> Chem.Mol:
    m = Chem.Mol(m)
    try:
        Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
    except Exception:
        Chem.SanitizeMol(m, sanitizeOps=(Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE))
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


# ---------- Geometry helpers ----------
def get_cys_geometry(pdb_file, chain, resid):
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
    coords = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            atom_name = line[12:16].strip()
            if atom_name and (atom_name[0] == 'H' or (len(atom_name) > 1 and atom_name[1] == 'H')):
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
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


# ---------- Warhead detection ----------
def find_acrylamide_beta_carbon(mol):
    """
    Find acrylamide warhead (N-C(=O)-C=C) and return (alpha_idx, beta_idx).
    Beta carbon is the terminal vinyl carbon where Cys-SG attacks in Michael addition.
    """
    patterns = [
        '[NX3][CX3](=O)[CX3]=[CH2]',
        '[NX3][CX3](=O)[CH]=[CH2]',
        '[NX3][CX3](=O)[CX3]=[CX3]',
        '[CX3](=O)[CX3]=[CH2]',
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


def find_chloroacetamide_alpha_carbon(mol):
    """
    Find chloroacetamide warhead (N-C(=O)-CH2-Cl) and return (carbonyl_idx, alpha_idx, cl_idx).
    Alpha carbon is the CH2 bearing Cl where Cys-SG attacks in SN2 displacement.
    """
    # Pattern: amide carbonyl - CH2 - Cl
    patterns = [
        '[NX3][CX3](=O)[CH2][Cl]',   # N-C(=O)-CH2-Cl
        '[CX3](=O)[CH2][Cl]',         # C(=O)-CH2-Cl (more general)
        '[NX3][CX3](=O)[CX4][Cl]',    # Allow substituted carbon
    ]
    for smarts in patterns:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            match = matches[0]
            # Find the carbonyl carbon, alpha carbon (CH2), and Cl
            cl_idx = None
            alpha_idx = None
            carbonyl_idx = None

            for idx in match:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == 'Cl':
                    cl_idx = idx
                elif atom.GetSymbol() == 'C':
                    # Check if carbonyl (has =O neighbor)
                    has_carbonyl_O = False
                    for nb in atom.GetNeighbors():
                        if nb.GetSymbol() == 'O':
                            bond = mol.GetBondBetweenAtoms(idx, nb.GetIdx())
                            if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                                has_carbonyl_O = True
                                break
                    if has_carbonyl_O:
                        carbonyl_idx = idx

            # Alpha carbon is the one bonded to both carbonyl and Cl
            if cl_idx is not None and carbonyl_idx is not None:
                cl_atom = mol.GetAtomWithIdx(cl_idx)
                for nb in cl_atom.GetNeighbors():
                    if nb.GetSymbol() == 'C' and nb.GetIdx() != carbonyl_idx:
                        # Check if this C is bonded to carbonyl
                        for nb2 in nb.GetNeighbors():
                            if nb2.GetIdx() == carbonyl_idx:
                                alpha_idx = nb.GetIdx()
                                break
                    if alpha_idx is not None:
                        break

            if alpha_idx is not None and cl_idx is not None and carbonyl_idx is not None:
                return carbonyl_idx, alpha_idx, cl_idx

    return None


def detect_warhead_type(mol):
    """
    Auto-detect warhead type from molecule structure.
    Returns: ('acrylamide', alpha_idx, beta_idx) or ('chloroacetamide', carbonyl_idx, alpha_idx, cl_idx) or None
    """
    # Try acrylamide first (Michael acceptor)
    acr_result = find_acrylamide_beta_carbon(mol)
    if acr_result is not None:
        return ('acrylamide', acr_result[0], acr_result[1])

    # Try chloroacetamide (SN2)
    cla_result = find_chloroacetamide_alpha_carbon(mol)
    if cla_result is not None:
        return ('chloroacetamide', cla_result[0], cla_result[1], cla_result[2])

    return None


# ---------- Convert acrylamide to Michael adduct ----------
def convert_to_michael_adduct(mol, alpha_idx, beta_idx):
    """
    Convert acrylamide C=C -> C-C (Michael adduct).
    Returns: (adduct_with_H, final_alpha_idx, final_beta_idx) or (None, None, None) on failure.
    """
    try:
        if mol is None or mol.GetNumConformers() == 0:
            print("    ERROR: input molecule has no conformer")
            return None, None, None

        conf_orig = mol.GetConformer()
        orig_beta_pos = np.array(conf_orig.GetAtomPosition(beta_idx), dtype=float)

        mol_noH = Chem.RemoveHs(mol)
        if mol_noH.GetNumConformers() == 0:
            conf_new = Chem.Conformer(mol_noH.GetNumAtoms())
            for i in range(mol_noH.GetNumAtoms()):
                try:
                    pos = conf_orig.GetAtomPosition(i)
                    conf_new.SetAtomPosition(i, Point3D(pos.x, pos.y, pos.z))
                except Exception:
                    conf_new.SetAtomPosition(i, Point3D(0.0, 0.0, 0.0))
            mol_noH.AddConformer(conf_new, assignId=True)

        emol = Chem.RWMol(mol_noH)

        patt = Chem.MolFromSmarts('[CX3](=O)[CX3]=[CX3]')
        matches = mol_noH.GetSubstructMatches(patt)
        if not matches:
            patt = Chem.MolFromSmarts('C(=O)C=C')
            matches = mol_noH.GetSubstructMatches(patt)
        if not matches:
            print("    ERROR: Could not find acrylamide pattern in H-removed molecule")
            return None, None, None

        # Prefer terminal CH2 as beta (TotalNumHs == 2)
        chosen_match = None
        chosen_beta_noH = None
        bestd = 1e12
        for match in matches:
            cand_vinyl = []
            for idx in match:
                atom = mol_noH.GetAtomWithIdx(idx)
                if atom.GetAtomicNum() != 6:
                    continue
                for nb in atom.GetNeighbors():
                    if nb.GetAtomicNum() == 6:
                        bnd = mol_noH.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
                        if bnd and bnd.GetBondType() == Chem.BondType.DOUBLE:
                            cand_vinyl.append(idx)
                            break
            if not cand_vinyl:
                continue
            for cand in cand_vinyl:
                try:
                    pos = np.array(mol_noH.GetConformer().GetAtomPosition(cand), dtype=float)
                except Exception:
                    pos = np.array([0.0, 0.0, 0.0], dtype=float)
                d = np.linalg.norm(pos - orig_beta_pos)
                try:
                    hcount = mol.GetAtomWithIdx(cand).GetTotalNumHs()
                except Exception:
                    hcount = 0
                score = d
                if hcount == 2:
                    score *= 0.001
                heavy_deg = sum(1 for nb in mol_noH.GetAtomWithIdx(cand).GetNeighbors() if nb.GetAtomicNum() > 1)
                score *= (1.0 + 0.1 * heavy_deg)
                if score < bestd:
                    bestd = score
                    chosen_match = match
                    chosen_beta_noH = cand

        if chosen_match is None or chosen_beta_noH is None:
            chosen_match = matches[0]
            chosen_beta_noH = chosen_match[-1]
            print("    WARNING: could not disambiguate terminal vinyl carbon; using fallback match ordering")

        beta_idx_noH = chosen_beta_noH
        alpha_idx_noH = None
        for idx in chosen_match:
            if idx == beta_idx_noH:
                continue
            atom = mol_noH.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() != 6:
                continue
            bond = mol_noH.GetBondBetweenAtoms(idx, beta_idx_noH)
            if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                alpha_idx_noH = idx
                break
        if alpha_idx_noH is None:
            for nb in mol_noH.GetAtomWithIdx(beta_idx_noH).GetNeighbors():
                if nb.GetAtomicNum() == 6:
                    alpha_idx_noH = nb.GetIdx()
                    break
        if alpha_idx_noH is None:
            print("    ERROR: Could not determine alpha carbon near chosen beta")
            return None, None, None

        bond = emol.GetBondBetweenAtoms(alpha_idx_noH, beta_idx_noH)
        if bond is None:
            print("    ERROR: No bond between alpha and beta in heavy-atom mol")
            return None, None, None
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            bond.SetBondType(Chem.BondType.SINGLE)
            emol.GetAtomWithIdx(alpha_idx_noH).SetHybridization(Chem.HybridizationType.SP3)
            emol.GetAtomWithIdx(beta_idx_noH).SetHybridization(Chem.HybridizationType.SP3)

        new_noH = emol.GetMol()

        conf_new = Chem.Conformer(new_noH.GetNumAtoms())
        for idx in range(new_noH.GetNumAtoms()):
            try:
                pos = conf_orig.GetAtomPosition(idx)
                conf_new.SetAtomPosition(idx, Point3D(pos.x, pos.y, pos.z))
            except Exception:
                conf_new.SetAtomPosition(idx, Point3D(0.0, 0.0, 0.0))
        new_noH.RemoveAllConformers()
        new_noH.AddConformer(conf_new, assignId=True)

        adduct_H = Chem.AddHs(new_noH, addCoords=True)

        try:
            heavy_indices = [a.GetIdx() for a in adduct_H.GetAtoms() if a.GetAtomicNum() > 1]
            props = AllChem.MMFFGetMoleculeProperties(adduct_H, mmffVariant='MMFF94s')
            if props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(adduct_H, props)
                if ff is not None:
                    for idx in heavy_indices:
                        ff.AddFixedPoint(idx)
                    ff.Initialize()
                    ff.Minimize(maxIts=200)
                    print("    Optimized H positions (heavy atoms fixed)")
        except Exception as e:
            print(f"    Note: MMFF optimization skipped or failed: {e}")

        try:
            adduct_H = Chem.AddHs(adduct_H, addCoords=True)
        except Exception:
            pass

        try:
            conf = adduct_H.GetConformer()
            coords_ok = True
            for a in adduct_H.GetAtoms():
                pos = conf.GetAtomPosition(a.GetIdx())
                if any(np.isnan([pos.x, pos.y, pos.z])):
                    coords_ok = False
                    break
        except Exception:
            coords_ok = False

        if not coords_ok:
            try:
                noH_tmp = Chem.RemoveHs(adduct_H)
                adduct_H = Chem.AddHs(noH_tmp, addCoords=True)
                try:
                    AllChem.UFFOptimizeMolecule(adduct_H, maxIters=150)
                except Exception:
                    pass
            except Exception:
                pass

        conf_final = adduct_H.GetConformer()
        final_beta_idx = None
        bestd = 1e9
        for atom in adduct_H.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            idx = atom.GetIdx()
            p = np.array(conf_final.GetAtomPosition(idx), dtype=float)
            d = np.linalg.norm(p - orig_beta_pos)
            if d < bestd:
                bestd = d
                final_beta_idx = idx

        if final_beta_idx is None:
            print("    ERROR: Could not locate final beta carbon")
            return None, None, None

        final_alpha_idx = None
        for nei in adduct_H.GetAtomWithIdx(final_beta_idx).GetNeighbors():
            if nei.GetAtomicNum() != 6:
                continue
            has_carbonyl = any(n2.GetAtomicNum() == 8 for n2 in nei.GetNeighbors())
            if has_carbonyl:
                final_alpha_idx = nei.GetIdx()
                break
        if final_alpha_idx is None:
            for nei in adduct_H.GetAtomWithIdx(final_beta_idx).GetNeighbors():
                if nei.GetAtomicNum() == 6:
                    final_alpha_idx = nei.GetIdx()
                    break

        print(f"    Michael adduct: {adduct_H.GetNumAtoms()} atoms (H count = {sum(1 for a in adduct_H.GetAtoms() if a.GetAtomicNum()==1)})")
        print(f"    Beta carbon index (for docking): {final_beta_idx}")
        return adduct_H, final_alpha_idx, final_beta_idx

    except Exception as e:
        print(f"    ERROR in convert_to_michael_adduct: {e}")
        return None, None, None


# ---------- Convert chloroacetamide to SN2 adduct ----------
def convert_to_sn2_adduct(mol, carbonyl_idx, alpha_idx, cl_idx):
    """
    Convert chloroacetamide to SN2 adduct by removing Cl.
    The alpha carbon (previously bearing Cl) becomes the covalent attachment point.

    Returns: (adduct_with_H, final_alpha_idx) or (None, None) on failure.
    """
    try:
        if mol is None or mol.GetNumConformers() == 0:
            print("    ERROR: input molecule has no conformer")
            return None, None

        conf_orig = mol.GetConformer()
        orig_alpha_pos = np.array(conf_orig.GetAtomPosition(alpha_idx), dtype=float)

        # Work with molecule copy
        mol_noH = Chem.RemoveHs(mol)
        if mol_noH.GetNumConformers() == 0:
            conf_new = Chem.Conformer(mol_noH.GetNumAtoms())
            for i in range(mol_noH.GetNumAtoms()):
                try:
                    pos = conf_orig.GetAtomPosition(i)
                    conf_new.SetAtomPosition(i, Point3D(pos.x, pos.y, pos.z))
                except Exception:
                    conf_new.SetAtomPosition(i, Point3D(0.0, 0.0, 0.0))
            mol_noH.AddConformer(conf_new, assignId=True)

        # Find Cl index in H-removed molecule (indices may shift)
        cl_idx_noH = None
        for atom in mol_noH.GetAtoms():
            if atom.GetSymbol() == 'Cl':
                cl_idx_noH = atom.GetIdx()
                break

        if cl_idx_noH is None:
            print("    ERROR: Cl not found in molecule")
            return None, None

        # Find alpha carbon (bonded to Cl) in noH molecule
        alpha_idx_noH = None
        cl_atom = mol_noH.GetAtomWithIdx(cl_idx_noH)
        for nb in cl_atom.GetNeighbors():
            if nb.GetSymbol() == 'C':
                alpha_idx_noH = nb.GetIdx()
                break

        if alpha_idx_noH is None:
            print("    ERROR: Could not find alpha carbon bonded to Cl")
            return None, None

        # Remove Cl atom
        emol = Chem.RWMol(mol_noH)
        emol.RemoveAtom(cl_idx_noH)

        # Adjust alpha index if Cl was before it
        final_alpha_idx_noH = alpha_idx_noH
        if cl_idx_noH < alpha_idx_noH:
            final_alpha_idx_noH = alpha_idx_noH - 1

        new_noH = emol.GetMol()

        # Preserve coordinates (adjust for removed atom)
        conf_new = Chem.Conformer(new_noH.GetNumAtoms())
        old_conf = mol_noH.GetConformer()
        new_idx = 0
        for old_idx in range(mol_noH.GetNumAtoms()):
            if old_idx == cl_idx_noH:
                continue
            try:
                pos = old_conf.GetAtomPosition(old_idx)
                conf_new.SetAtomPosition(new_idx, Point3D(pos.x, pos.y, pos.z))
            except Exception:
                conf_new.SetAtomPosition(new_idx, Point3D(0.0, 0.0, 0.0))
            new_idx += 1
        new_noH.RemoveAllConformers()
        new_noH.AddConformer(conf_new, assignId=True)

        # Sanitize
        try:
            Chem.SanitizeMol(new_noH)
        except Exception as e:
            print(f"    WARNING: Sanitization issue: {e}")

        # Add hydrogens
        adduct_H = Chem.AddHs(new_noH, addCoords=True)

        # Optimize H positions with heavy atoms fixed
        try:
            heavy_indices = [a.GetIdx() for a in adduct_H.GetAtoms() if a.GetAtomicNum() > 1]
            props = AllChem.MMFFGetMoleculeProperties(adduct_H, mmffVariant='MMFF94s')
            if props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(adduct_H, props)
                if ff is not None:
                    for idx in heavy_indices:
                        ff.AddFixedPoint(idx)
                    ff.Initialize()
                    ff.Minimize(maxIts=200)
                    print("    Optimized H positions (heavy atoms fixed)")
        except Exception as e:
            print(f"    Note: MMFF optimization skipped or failed: {e}")

        # Find final alpha carbon index by position
        conf_final = adduct_H.GetConformer()
        final_alpha_idx = None
        bestd = 1e9
        for atom in adduct_H.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            idx = atom.GetIdx()
            p = np.array(conf_final.GetAtomPosition(idx), dtype=float)
            d = np.linalg.norm(p - orig_alpha_pos)
            if d < bestd:
                bestd = d
                final_alpha_idx = idx

        if final_alpha_idx is None:
            print("    ERROR: Could not locate final alpha carbon")
            return None, None

        print(f"    SN2 adduct: {adduct_H.GetNumAtoms()} atoms (H count = {sum(1 for a in adduct_H.GetAtoms() if a.GetAtomicNum()==1)})")
        print(f"    Alpha carbon index (for docking): {final_alpha_idx}")
        return adduct_H, final_alpha_idx

    except Exception as e:
        print(f"    ERROR in convert_to_sn2_adduct: {e}")
        return None, None


# ---------- SMARTS fallback ----------
def pick_beta_covalent_smarts_saturated(mol, beta_idx):
    candidates = [
        '[#6;X4;H2]',
        '[#6;X4;!$(C=O)]',
        '[#6;X4]',
        '[$([CX4])]',
        '[#6]'
    ]
    for smarts in candidates:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            continue
        matched_idxs = set(m[0] if isinstance(m, tuple) else m for m in matches)
        if beta_idx in matched_idxs:
            print(f"    Selected covalent SMARTS: {smarts} (matches include beta_idx={beta_idx})")
            return smarts
    for smarts in candidates:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        if mol.GetSubstructMatches(patt):
            print(f"    WARNING: None matched beta_idx exactly; using {smarts} (first that matches somewhere)")
            return smarts
    print("    WARNING: No SMARTS matched; using generic carbon '[#6]'")
    return '[#6]'


# ---------- Pre-positioning ----------
def preposition_ligand_beta_at_SG(mol, beta_idx, sg_pos, receptor_coords=None, cb_pos=None, n_rotations=24):
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


# ---------- GNINA runner ----------
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
        exe = shutil.which('gnina') or 'gnina'
    print("Using GNINA exe:", exe)

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

    print("\n[GNINA] Command:", ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
        return True

    stderr = (result.stderr or "") + "\n" + (result.stdout or "")
    print(f"    GNINA output:\n{stderr}")

    s = stderr.lower()
    if gpu and ("cuda" in s or "device" in s or "kernel" in s):
        print("    Retrying with CPU...")
        cmd_cpu = [c for c in cmd if c != '--device' and not c.isdigit()]
        if '--no_gpu' not in cmd_cpu:
            cmd_cpu.append('--no_gpu')
        result = subprocess.run(cmd_cpu, capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
            return True

    return False


# ---------- Pose filtering ----------
def find_beta_idx_in_pose_saturated(mol, target_sg_pos):
    conf = mol.GetConformer()
    best_idx = None
    best_dist = float('inf')
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        d = np.linalg.norm(np.array([pos.x, pos.y, pos.z]) - target_sg_pos)
        if d < best_dist:
            best_dist = d
            best_idx = idx
    return best_idx


def filter_poses_by_geometry(sdf_file, target_sg_pos, max_dist=3.0,
                             cys_cb_pos=None, angle_bounds=(90, 125), debug=False):
    """
    Filter and rank docked poses by geometry and Vina affinity.

    Ranking criteria (for covalent docking):
      1. Vina affinity (more negative = better binding) - primary
      2. Distance to Cys-SG (closer = better) - secondary

    NOTE: CNN scoring is not used for ranking because the CNN was not trained
    on covalent complexes (see GNINA 1.3 paper, McNutt et al. 2025).
    """
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
    poses = []
    for i, mol in enumerate(suppl):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        beta_idx = find_beta_idx_in_pose_saturated(mol, target_sg_pos)
        if beta_idx is None:
            continue
        conf = mol.GetConformer()
        bp = conf.GetAtomPosition(beta_idx)
        beta_pos = np.array([bp.x, bp.y, bp.z], dtype=float)
        dist = np.linalg.norm(beta_pos - target_sg_pos)

        # Extract Vina affinity (primary ranking metric for covalent docking)
        # GNINA outputs this as 'minimizedAffinity' or 'affinity' (kcal/mol, more negative = better)
        vina_affinity = 0.0
        for prop_name in ['minimizedAffinity', 'affinity', 'minimized affinity']:
            if mol.HasProp(prop_name):
                try:
                    vina_affinity = float(mol.GetProp(prop_name))
                    break
                except Exception:
                    pass

        # CNN score (kept for reference but NOT used for ranking covalent poses)
        cnn_score = 0.0
        if mol.HasProp('CNNscore'):
            try:
                cnn_score = float(mol.GetProp('CNNscore'))
            except Exception:
                pass

        if dist > max_dist:
            if debug:
                print(f"    Pose {i}: dist {dist:.2f} Å > {max_dist:.2f} Å; skipping")
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
                print(f"    Pose {i}: angle {ang_val:.1f}° out of bounds; skipping")
            continue
        poses.append({
            'mol': mol,
            'idx': i,
            'beta_dist': dist,
            'beta_idx': beta_idx,
            'angle': ang_val,
            'vina_affinity': vina_affinity,
            'cnn_score': cnn_score  # kept for reference
        })

    # Sort by Vina affinity (more negative = better), then by distance (closer = better)
    poses.sort(key=lambda x: (x['vina_affinity'], x['beta_dist']))
    return poses


def write_complex_pdb(receptor_pdb, ligand_mol, output_pdb, cys_chain, cys_resid, beta_idx):
    lines = []
    sg_serial = None
    last_serial = 0
    with open(receptor_pdb, 'r') as f:
        for line in f:
            if line.startswith('END'):
                continue
            if line.startswith('ATOM'):
                try:
                    serial = int(line[6:11].strip())
                except Exception:
                    serial = 0
                last_serial = max(last_serial, serial)
                chain_id = line[21]
                res_name = line[17:20].strip()
                atom_name = line[12:16].strip()
                try:
                    res_num = int(line[22:26].strip())
                except Exception:
                    res_num = 0
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


# ---------- I/O ----------
def load_ligand(path_str):
    lig_path = Path(path_str)
    if not lig_path.exists() or lig_path.stat().st_size == 0:
        print(f"ERROR: Ligand file not found: {path_str}")
        return None
    mol = None
    ext = lig_path.suffix.lower()
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
            suppl = Chem.SDMolSupplier(str(lig_path), removeHs=False, sanitize=True, strictParsing=False)
            for m in suppl:
                if m is not None:
                    mol = m
                    break
    except Exception as e:
        print(f"    RDKit parse warning: {e}")
    if mol is None:
        return None
    mol = sanitize_molecule(mol, ensure_3d=True)
    return mol


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Covalent Docking - Supports acrylamide (Michael addition) and chloroacetamide (SN2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Warhead types:
  acrylamide      - Michael addition: N-C(=O)-C=C, attacks at beta carbon
  chloroacetamide - SN2 displacement: N-C(=O)-CH2-Cl, attacks at alpha carbon
  auto            - Auto-detect warhead type from structure

Examples:
  python %(prog)s -l ligand_acrylamide.sdf -r receptor.pdb --cys_resid 145 --warhead acrylamide
  python %(prog)s -l ligand_chloroacetamide.sdf -r receptor.pdb --cys_resid 145 --warhead chloroacetamide
  python %(prog)s -l ligand.sdf -r receptor.pdb --cys_resid 145 --warhead auto
        """
    )
    parser.add_argument('--ligand', '-l', required=True, help='Ligand file (SDF/MOL/SMILES)')
    parser.add_argument('--receptor', '-r', required=True, help='Receptor PDB')
    parser.add_argument('--cys_chain', default='A', help='Chain ID of target cysteine')
    parser.add_argument('--cys_resid', type=int, required=True, help='Residue number of target cysteine')
    parser.add_argument('--warhead', '-w', choices=['acrylamide', 'chloroacetamide', 'auto'],
                        default='auto', help='Warhead type (default: auto-detect)')
    parser.add_argument('--output', '-o', default='covalent_docking', help='Output directory')
    parser.add_argument('--box_size', type=float, default=18, help='Docking box size (Angstrom)')
    parser.add_argument('--exhaustiveness', type=int, default=16, help='GNINA exhaustiveness')
    parser.add_argument('--local_only', action='store_true', help='Only local optimization')
    parser.add_argument('--gnina_path', default='~/software/gnina/build/bin/gnina', help='Path to GNINA binary')
    # NOTE: CNN scoring is NOT recommended for covalent docking - the CNN was not
    # trained on covalent complexes (see GNINA 1.3 paper). Use Vina affinity instead.
    parser.add_argument('--cnn_scoring', choices=['none', 'rescore', 'refine'], default='none',
                        help='CNN scoring mode (default: none - recommended for covalent)')
    parser.add_argument('--cnn_model', default='crossdock_default2018')
    parser.add_argument('--num_modes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--max_covalent_dist', type=float, default=2.5,
                        help='Max distance from covalent carbon to Cys-SG (Angstrom)')
    parser.add_argument('--protonate_receptor', action='store_true')
    parser.add_argument('--ph', type=float, default=7.4)
    parser.add_argument('--keep_hets', action='store_true')
    parser.add_argument('--prot_method', choices=['reduce', 'pdbfixer'], default='reduce')
    args = parser.parse_args()

    print("=" * 70)
    print("COVALENT DOCKING - Acrylamide/Chloroacetamide")
    print("=" * 70)

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    receptor_for_docking = args.receptor
    if args.protonate_receptor:
        print("\n[0] Protonating receptor (preserving target Cys as thiolate)...")
        prot_pdb = str(outdir / f"receptor_protonated_pH{args.ph:.1f}_{args.prot_method}.pdb")
        ok = protonate_receptor_preserving_resnums(
            args.receptor, prot_pdb,
            chain_id=args.cys_chain, resid=args.cys_resid,
            ph=args.ph, keep_hets=args.keep_hets, method=args.prot_method
        )
        if not ok:
            print("ERROR: Protonation failed; aborting")
            sys.exit(1)
        receptor_for_docking = prot_pdb
        print(f"    Using protonated receptor: {receptor_for_docking}")

    print("\n[1] Loading target cysteine coords...")
    cys_coords = get_cys_geometry(receptor_for_docking, args.cys_chain, args.cys_resid)
    if cys_coords is None:
        print(f"ERROR: Cys {args.cys_chain}:{args.cys_resid} not found in {receptor_for_docking}")
        sys.exit(1)
    sg_pos = cys_coords['SG']
    cb_pos = cys_coords.get('CB', sg_pos + np.array([1.5, 0.0, 0.0], dtype=float))
    print(f"    SG: ({sg_pos[0]:.2f}, {sg_pos[1]:.2f}, {sg_pos[2]:.2f})")

    print("\n[2] Loading ligand...")
    mol = load_ligand(args.ligand)
    if mol is None:
        print("ERROR: Could not parse ligand.")
        sys.exit(1)
    print(f"    Loaded ligand: {mol.GetNumAtoms()} atoms")
    try:
        print(f"    SMILES (input): {Chem.MolToSmiles(Chem.RemoveHs(mol))}")
    except Exception:
        pass

    print("\n[3] Detecting warhead type...")
    warhead_type = args.warhead
    covalent_carbon_idx = None  # The carbon that bonds to Cys-SG

    if warhead_type == 'auto':
        detection = detect_warhead_type(mol)
        if detection is None:
            print("ERROR: Could not auto-detect warhead type.")
            print("       Expected acrylamide (N-C(=O)-C=C) or chloroacetamide (N-C(=O)-CH2-Cl)")
            sys.exit(1)
        warhead_type = detection[0]
        print(f"    Auto-detected warhead: {warhead_type}")

    if warhead_type == 'acrylamide':
        res = find_acrylamide_beta_carbon(mol)
        if res is None:
            print("ERROR: Acrylamide warhead not found (N-C(=O)-C=C)")
            sys.exit(1)
        alpha_idx, beta_idx = res
        print(f"    Found acrylamide: alpha C={alpha_idx}, beta C={beta_idx}")

        print("\n[4] Converting to Michael adduct (saturating C=C)...")
        adduct, adduct_alpha_idx, adduct_covalent_idx = convert_to_michael_adduct(mol, alpha_idx, beta_idx)
        if adduct is None:
            print("ERROR: Failed to convert to Michael adduct")
            sys.exit(1)
        covalent_carbon_idx = adduct_covalent_idx
        adduct_sdf = str(outdir / "ligand_michael_adduct.sdf")

    elif warhead_type == 'chloroacetamide':
        res = find_chloroacetamide_alpha_carbon(mol)
        if res is None:
            print("ERROR: Chloroacetamide warhead not found (N-C(=O)-CH2-Cl)")
            sys.exit(1)
        carbonyl_idx, alpha_idx, cl_idx = res
        print(f"    Found chloroacetamide: carbonyl C={carbonyl_idx}, alpha C={alpha_idx}, Cl={cl_idx}")

        print("\n[4] Converting to SN2 adduct (removing Cl)...")
        adduct, adduct_covalent_idx = convert_to_sn2_adduct(mol, carbonyl_idx, alpha_idx, cl_idx)
        if adduct is None:
            print("ERROR: Failed to convert to SN2 adduct")
            sys.exit(1)
        covalent_carbon_idx = adduct_covalent_idx
        adduct_sdf = str(outdir / "ligand_sn2_adduct.sdf")

    else:
        print(f"ERROR: Unknown warhead type: {warhead_type}")
        sys.exit(1)

    try:
        print(f"    SMILES (adduct): {Chem.MolToSmiles(Chem.RemoveHs(adduct))}")
    except Exception:
        pass

    adduct_for_write = Chem.AddHs(adduct, addCoords=True)
    write_sdf_v3000(adduct_for_write, adduct_sdf)
    adduct = adduct_for_write
    print(f"    Saved adduct: {adduct_sdf}")

    print("\n[5] Pre-positioning ligand...")
    receptor_coords = get_receptor_coords(receptor_for_docking, args.cys_chain, args.cys_resid)
    adduct = preposition_ligand_beta_at_SG(adduct, covalent_carbon_idx, sg_pos, receptor_coords, cb_pos, n_rotations=24)

    positioned_sdf = str(outdir / "ligand_positioned.sdf")

    # --- IMPORTANT: tag the chosen covalent carbon with isotope=13 so SMARTS '[13C]' matches only it
    for a in adduct.GetAtoms():
        a.SetIsotope(0)
    try:
        adduct.GetAtomWithIdx(covalent_carbon_idx).SetIsotope(13)
    except Exception:
        print("    ERROR: could not set isotope on covalent carbon")

    positioned_for_write = Chem.AddHs(adduct, addCoords=True)
    write_sdf_v3000(positioned_for_write, positioned_sdf)
    print(f"    Saved: {positioned_sdf}")
    print(f"    Adduct confs: {adduct.GetNumConformers()}  covalent_carbon_idx: {covalent_carbon_idx}")

    print("\n[6] GNINA covalent docking...")
    docked_sdf = str(outdir / "docked_poses.sdf")
    cov_rec_atom = f"{args.cys_chain}:{args.cys_resid}:SG"

    # Use isotopic SMARTS that selects only the tagged beta carbon
    cov_lig_smarts = '[13C]'
    print(f"    Using isotopic SMARTS to force single-atom match: {cov_lig_smarts}")

    # Debug: confirm match indices
    try:
        patt_dbg = Chem.MolFromSmarts(cov_lig_smarts)
        matches_dbg = adduct.GetSubstructMatches(patt_dbg)
        matched = [m[0] if isinstance(m, tuple) else m for m in matches_dbg] if matches_dbg else []
        print(f"    SMARTS matches ligand atom indices: {matched} (expected: [{adduct_beta_idx}])")
    except Exception as e:
        print(f"    SMARTS debug failed: {e}")

    h_count = sum(1 for a in adduct.GetAtoms() if a.GetAtomicNum() == 1)
    print(f"    Adduct hydrogen count: {h_count}")

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
        covalent_lig_atom_position=tuple(sg_pos),
        covalent_optimize_lig=True,
        gnina_path=args.gnina_path
    )

    if not success:
        print("WARNING: GNINA failed; using pre-positioned ligand as fallback")
        shutil.copy(positioned_sdf, docked_sdf)

    print("\n[7] Filtering poses...")
    good_poses = filter_poses_by_geometry(
        docked_sdf, sg_pos,
        max_dist=args.max_covalent_dist,
        cys_cb_pos=cb_pos,
        angle_bounds=(90, 125),
        debug=False
    )

    if not good_poses:
        print("    WARNING: No valid poses found; returning pre-positioned adduct")
        good_poses = [{
            'mol': adduct,
            'idx': 0,
            'beta_dist': 0.0,
            'angle': None,
            'vina_affinity': 0.0,
            'cnn_score': 0.0,
            'beta_idx': covalent_carbon_idx
        }]

    print(f"    Found {len(good_poses)} valid poses (post-filtering).")
    best = good_poses[0]
    best_sdf = str(outdir / "best_pose.sdf")
    write_sdf_v3000(best['mol'], best_sdf)
    print(f"    Best pose saved: {best_sdf}")
    print(f"    Best pose metrics:")
    print(f"      - Vina affinity: {best.get('vina_affinity', 0.0):.2f} kcal/mol")
    print(f"      - Distance to SG: {best.get('beta_dist', 0.0):.2f} Å")
    if best.get('angle') is not None:
        print(f"      - C-SG-CB angle: {best['angle']:.1f}°")

    print("\n[8] Writing complex PDB...")
    complex_pdb = str(outdir / "complex.pdb")
    best_covalent_idx = best.get('beta_idx', covalent_carbon_idx)
    success_complex = write_complex_pdb(receptor_for_docking, best['mol'], complex_pdb,
                                       args.cys_chain, args.cys_resid, best_covalent_idx)
    if success_complex:
        print(f"    Complex: {complex_pdb}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nWarhead type: {warhead_type}")
    print(f"Output directory: {outdir.resolve()}")
    adduct_name = "ligand_michael_adduct.sdf" if warhead_type == 'acrylamide' else "ligand_sn2_adduct.sdf"
    print(f"  - {adduct_name}")
    print("  - ligand_positioned.sdf")
    print("  - docked_poses.sdf")
    print("  - best_pose.sdf")
    print("  - complex.pdb")


if __name__ == "__main__":
    main()
