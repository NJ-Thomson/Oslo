#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
b04_extract_adduct_fragment_methyl_optimized.py

Extract covalent fragment with OPTIMIZED caps that minimize influence
on ligand charge derivation by maximizing cap-ligand distance.

Supports two cap types:
  --cap-type methyl (default):
      CH3-N-CA-C(=O)-CH3  (methylamine N-cap, methyl C-cap, keeps carbonyl)

  --cap-type aliphatic:
      CH3-NH-CA-CH3  (no carbonyls - pure aliphatic caps)
      Better for minimizing electronic polarization of ligand.

This version systematically samples cap conformations by rotating methyl H atoms
around the cap bonds. The conformation with maximum distance to ligand is selected.

Usage:
  python b04_extract_adduct_fragment_methyl_optimized.py \\
    --complex complex.pdb \\
    --cys-resname CYS --cys-resid 1039 \\
    --lig-resname LIG --lig-resid 1 \\
    --out-prefix Outputs/adduct \\
    --cap-type aliphatic \\
    --n-samples 36
"""

import argparse
import json
import math
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Geometry helpers
# ============================================================================

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return v / norm


def perpendicular_vector(v: np.ndarray) -> np.ndarray:
    """Find a vector perpendicular to v."""
    v = normalize(v)
    if abs(v[0]) < 0.9:
        perp = np.cross(v, np.array([1.0, 0.0, 0.0]))
    else:
        perp = np.cross(v, np.array([0.0, 1.0, 0.0]))
    return normalize(perp)


def rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Return rotation matrix for rotation around axis by theta radians."""
    axis = normalize(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
    ])


def rotate_point_around_axis(point: np.ndarray, axis_point: np.ndarray,
                              axis_dir: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a point around an axis."""
    # Translate to origin
    p = point - axis_point
    # Rotate
    R = rotation_matrix(axis_dir, angle)
    p_rot = R @ p
    # Translate back
    return p_rot + axis_point


def build_tetrahedral_h(center: np.ndarray, attached: np.ndarray,
                        bond_length: float = 1.09, n_hydrogens: int = 3,
                        initial_angle: float = 0.0) -> List[np.ndarray]:
    """Build hydrogens in tetrahedral geometry with controllable initial rotation."""
    vec = normalize(attached - center)
    tet_angle = math.radians(109.47)
    perp1 = perpendicular_vector(vec)
    perp2 = normalize(np.cross(vec, perp1))

    positions = []
    for i in range(n_hydrogens):
        phi = initial_angle + math.radians(i * 360 / n_hydrogens)
        axial = -vec * math.cos(math.pi - tet_angle)
        radial = (perp1 * math.cos(phi) + perp2 * math.sin(phi)) * math.sin(math.pi - tet_angle)
        h_dir = normalize(axial + radial)
        positions.append(center + h_dir * bond_length)

    return positions


def place_atom_tetrahedral(center: np.ndarray, existing: List[np.ndarray],
                           bond_length: float) -> np.ndarray:
    """Place an atom in tetrahedral geometry given existing attachments."""
    if len(existing) == 1:
        vec = normalize(existing[0] - center)
        perp = perpendicular_vector(vec)
        tet_angle = math.radians(109.47)
        return center + bond_length * normalize(-vec * math.cos(tet_angle) + perp * math.sin(tet_angle))
    elif len(existing) == 2:
        v1 = normalize(existing[0] - center)
        v2 = normalize(existing[1] - center)
        avg = normalize(v1 + v2)
        return center - avg * bond_length
    elif len(existing) == 3:
        avg = sum(normalize(e - center) for e in existing) / 3
        return center - normalize(avg) * bond_length
    else:
        return center + np.array([bond_length, 0, 0])


# ============================================================================
# Standard bond lengths (Å)
# ============================================================================
BOND = {
    'C-N': 1.335,   # amide C-N
    'C-C': 1.525,   # sp3-sp3 C-C
    'C=O': 1.229,   # carbonyl
    'C-H': 1.09,
    'N-H': 1.01,
    'N-C': 1.47,    # sp3 N-C (amine)
    'CA-N': 1.458,
    'CA-C': 1.525,
    'CA-CB': 1.530,
    'CB-SG': 1.808,
    'S-C': 1.82,
}


def place_tetrahedral_fourth(center: np.ndarray, existing: List[np.ndarray],
                              bond_length: float) -> np.ndarray:
    """Place fourth substituent in tetrahedral geometry given three existing."""
    if len(existing) != 3:
        raise ValueError("Need exactly 3 existing positions")
    # Fourth position is opposite to the average of the three
    avg_dir = np.zeros(3)
    for e in existing:
        avg_dir += normalize(e - center)
    avg_dir /= 3
    return center - normalize(avg_dir) * bond_length


def place_tetrahedral_third(center: np.ndarray, first: np.ndarray,
                             second: np.ndarray, bond_length: float,
                             prefer_dir: np.ndarray = None) -> np.ndarray:
    """Place third substituent in tetrahedral geometry given two existing.

    Returns one of two possible positions. If prefer_dir is given, returns
    the position closer to that direction.
    """
    v1 = normalize(first - center)
    v2 = normalize(second - center)
    avg = normalize(v1 + v2)

    # Perpendicular to the v1-v2 plane
    perp = np.cross(v1, v2)
    if np.linalg.norm(perp) < 1e-6:
        perp = perpendicular_vector(v1)
    else:
        perp = normalize(perp)

    # Tetrahedral angle from avg
    tet_half = math.radians(109.47 / 2)

    # Two possible positions
    pos1 = center + bond_length * normalize(-avg * math.cos(tet_half) + perp * math.sin(tet_half))
    pos2 = center + bond_length * normalize(-avg * math.cos(tet_half) - perp * math.sin(tet_half))

    if prefer_dir is not None:
        if np.dot(pos1 - center, prefer_dir) > np.dot(pos2 - center, prefer_dir):
            return pos1
        else:
            return pos2
    return pos1


# ============================================================================
# PDB parsing and writing
# ============================================================================

def parse_pdb(pdb_path: str) -> Tuple[List[Dict], List[List[int]]]:
    atoms, conects = [], []
    with open(pdb_path, 'r') as f:
        for line in f:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM"):
                try:
                    serial = int(line[6:11])
                except Exception:
                    continue
                atom = {
                    "rec": rec,
                    "serial": serial,
                    "name": line[12:16],
                    "resn": line[17:20],
                    "chain": line[21],
                    "resi": int(line[22:26]),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                    "occ": line[54:60] if len(line) > 54 else "  1.00",
                    "b": line[60:66] if len(line) > 60 else "  0.00",
                    "elem": (line[76:78].strip() if len(line) > 76 else "") or line[12:16].strip()[:1],
                }
                atoms.append(atom)
            elif rec == "CONECT":
                fields = [line[6:11], line[11:16], line[16:21], line[21:26], line[26:31]]
                try:
                    serials = [int(f.strip()) for f in fields if f.strip()]
                    if serials:
                        conects.append(serials)
                except Exception:
                    pass
    return atoms, conects


def format_pdb_line(serial: int, name: str, resn: str, chain: str, resi: int,
                    x: float, y: float, z: float, elem: str,
                    rec: str = "ATOM", occ: float = 1.00, b: float = 0.00) -> str:
    """Format a PDB ATOM/HETATM line."""
    if len(name) == 4:
        name_fmt = name
    elif len(elem) == 2 or name[0].isdigit():
        name_fmt = f"{name:<4s}"
    else:
        name_fmt = f" {name:<3s}"

    return (f"{rec:<6s}{serial:5d} {name_fmt}{resn:>3s} {chain}"
            f"{resi:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{occ:6.2f}{b:6.2f}          {elem:>2s}")


def select_ligand(atoms: List[Dict], lig_resname: str, lig_resid: int) -> List[Dict]:
    return [a for a in atoms
            if a["rec"] == "HETATM"
            and a["resn"].strip() == lig_resname.strip()
            and a["resi"] == lig_resid]


def find_cys_atom(atoms: List[Dict], cys_resname: str, cys_resid: int,
                  atom_name: str) -> Optional[Dict]:
    """Find a specific atom from the cysteine residue."""
    for a in atoms:
        if (a["rec"] == "ATOM"
            and a["resn"].strip() == cys_resname.strip()
            and a["resi"] == cys_resid
            and a["name"].strip() == atom_name):
            return a
    return None


def find_beta_from_conect(conects: List[List[int]], sg_serial: int,
                          lig_serials: set) -> Optional[int]:
    for row in conects:
        if not row:
            continue
        src = row[0]
        if src == sg_serial:
            for t in row[1:]:
                if t in lig_serials:
                    return t
        if src in lig_serials and sg_serial in row[1:]:
            return src
    return None


# ============================================================================
# Build capped fragment with conformational sampling
# ============================================================================

def build_methyl_cys_fragment(
    sg_pos: np.ndarray,
    cb_pos: np.ndarray,
    ca_pos: np.ndarray,
    n_pos: np.ndarray,
    c_pos: np.ndarray,
    o_pos: np.ndarray = None,
    ligand_pos: np.ndarray = None,
    n_term_rotation: float = 0.0,  # Rotation angle for N-terminal CH3
    c_term_rotation: float = 0.0,  # Rotation angle for C-terminal CH3
) -> Dict[str, np.ndarray]:
    """
    Build CH3-N-CA-C(=O)-CH3 structure with rotatable caps (keeps carbonyl).

    n_term_rotation: rotation angle (radians) for N-terminal methyl H atoms
    c_term_rotation: rotation angle (radians) for C-terminal methyl H atoms
    """
    atoms = {}

    # Use actual backbone positions
    atoms['SG'] = sg_pos
    atoms['CB'] = cb_pos
    atoms['CA'] = ca_pos
    atoms['N'] = n_pos
    atoms['C'] = c_pos

    # Direction from SG toward ligand
    if ligand_pos is not None:
        sg_to_lig = normalize(ligand_pos - sg_pos)
    else:
        sg_to_lig = normalize(sg_pos - cb_pos)

    away_from_lig = -sg_to_lig

    # Carbonyl oxygen
    if o_pos is not None:
        atoms['O'] = o_pos
    else:
        ca_to_c = normalize(c_pos - ca_pos)
        o_perp = perpendicular_vector(ca_to_c)
        o_dir = normalize(o_perp + away_from_lig * 0.3)
        atoms['O'] = c_pos + o_dir * BOND['C=O']

    # HA on CA
    atoms['HA'] = place_atom_tetrahedral(ca_pos, [n_pos, cb_pos, c_pos], BOND['C-H'])

    # HB2, HB3 on CB
    cb_to_sg = normalize(sg_pos - cb_pos)
    cb_to_ca = normalize(ca_pos - cb_pos)
    hb_perp = normalize(np.cross(cb_to_sg, cb_to_ca))
    hb_avg = normalize(cb_to_sg + cb_to_ca)

    hb2_dir = normalize(-hb_avg + hb_perp * 0.9)
    hb3_dir = normalize(-hb_avg - hb_perp * 0.9)
    atoms['HB2'] = cb_pos + hb2_dir * BOND['C-H']
    atoms['HB3'] = cb_pos + hb3_dir * BOND['C-H']

    # H (amide H on N)
    ca_to_n = normalize(n_pos - ca_pos)
    h_perp = perpendicular_vector(ca_to_n)
    if np.dot(h_perp, away_from_lig) < 0:
        h_perp = -h_perp
    h_dir = normalize(-ca_to_n * 0.3 + h_perp * 0.95)
    atoms['H'] = n_pos + h_dir * BOND['N-H']

    # === N-terminal methyl cap (CH3-N) - ROTATABLE ===
    n_to_ca = normalize(ca_pos - n_pos)
    n_to_h = normalize(atoms['H'] - n_pos)

    cn_base_dir = normalize(-n_to_ca)
    cn_dir = normalize(cn_base_dir + away_from_lig * 0.5)
    atoms['CN'] = n_pos + cn_dir * BOND['N-C']

    # Build hydrogens with rotation around N-CN axis
    hn_positions = build_tetrahedral_h(
        atoms['CN'], n_pos, BOND['C-H'], 3,
        initial_angle=n_term_rotation
    )
    atoms['HN1'] = hn_positions[0]
    atoms['HN2'] = hn_positions[1]
    atoms['HN3'] = hn_positions[2]

    # === C-terminal methyl cap (C-CH3) - ROTATABLE ===
    c_to_ca = normalize(ca_pos - c_pos)
    c_to_o = normalize(atoms['O'] - c_pos)

    cc_base_dir = normalize(-c_to_ca)
    cc_dir = normalize(cc_base_dir + away_from_lig * 0.5)
    atoms['CC'] = c_pos + cc_dir * BOND['C-C']

    # Build hydrogens with rotation around C-CC axis
    hc_positions = build_tetrahedral_h(
        atoms['CC'], c_pos, BOND['C-H'], 3,
        initial_angle=c_term_rotation
    )
    atoms['HC1'] = hc_positions[0]
    atoms['HC2'] = hc_positions[1]
    atoms['HC3'] = hc_positions[2]

    return atoms


def build_aliphatic_cys_fragment(
    sg_pos: np.ndarray,
    cb_pos: np.ndarray,
    ca_pos: np.ndarray,
    ligand_pos: np.ndarray = None,
    n_term_rotation: float = 0.0,  # Rotation angle for N-terminal CH3 H atoms
    c_term_rotation: float = 0.0,  # Rotation angle for C-terminal CH3 H atoms
) -> Dict[str, np.ndarray]:
    """
    Build aliphatic-capped fragment: CH3-NH-CA(-CB-SG)-CH3

    NO carbonyls - pure aliphatic caps for minimal electronic polarization.

    Structure:
        CH3 - N - CA - CC - H3
              |    |
              H    CB
                   |
                   SG - Ligand

    CA has 4 bonds: N, CB, CC, HA (tetrahedral)
    N has 3 bonds: CH3, CA, H (sp3 amine)
    CC has 4 bonds: CA, HC1, HC2, HC3 (methyl)

    n_term_rotation: rotation angle (radians) for N-cap CH3 H atoms
    c_term_rotation: rotation angle (radians) for C-cap CH3 H atoms
    """
    atoms = {}

    # Keep actual positions from PDB
    atoms['SG'] = sg_pos
    atoms['CB'] = cb_pos
    atoms['CA'] = ca_pos

    # Direction away from ligand (for orienting caps)
    if ligand_pos is not None:
        away_from_lig = normalize(ca_pos - ligand_pos)
    else:
        away_from_lig = normalize(ca_pos - sg_pos)

    # HB2, HB3 on CB (tetrahedral with CA and SG)
    cb_to_sg = normalize(sg_pos - cb_pos)
    cb_to_ca = normalize(ca_pos - cb_pos)
    hb_perp = np.cross(cb_to_sg, cb_to_ca)
    if np.linalg.norm(hb_perp) < 1e-6:
        hb_perp = perpendicular_vector(cb_to_sg)
    else:
        hb_perp = normalize(hb_perp)
    hb_avg = normalize(cb_to_sg + cb_to_ca)

    atoms['HB2'] = cb_pos + normalize(-hb_avg + hb_perp * 0.9) * BOND['C-H']
    atoms['HB3'] = cb_pos + normalize(-hb_avg - hb_perp * 0.9) * BOND['C-H']

    # === Place N and CC tetrahedrally around CA ===
    # CB is already fixed. We need to place N and CC at ~109.47° from CB and each other.

    ca_to_cb = normalize(cb_pos - ca_pos)

    # Find two perpendicular vectors in the plane perpendicular to CA-CB
    perp1 = perpendicular_vector(ca_to_cb)
    perp2 = normalize(np.cross(ca_to_cb, perp1))

    # Tetrahedral angle from the CB-CA axis
    tet_angle = math.radians(109.47)

    # N position: in the plane perpendicular to CB, pointing away from ligand
    # Rotate to be ~109.47° from CB
    n_base = -ca_to_cb * math.cos(math.pi - tet_angle) + perp1 * math.sin(math.pi - tet_angle)

    # Rotate n_base around ca_to_cb to point more toward away_from_lig
    # Find angle to rotate
    proj_away = away_from_lig - np.dot(away_from_lig, ca_to_cb) * ca_to_cb
    if np.linalg.norm(proj_away) > 1e-6:
        proj_away = normalize(proj_away)
        # Angle between perp1 and proj_away
        cos_angle = np.clip(np.dot(perp1, proj_away), -1, 1)
        sin_angle = np.dot(np.cross(perp1, proj_away), ca_to_cb)
        rot_angle = math.atan2(sin_angle, cos_angle)
    else:
        rot_angle = 0.0

    # Apply rotation
    n_dir = -ca_to_cb * math.cos(math.pi - tet_angle)
    n_dir += (perp1 * math.cos(rot_angle) + perp2 * math.sin(rot_angle)) * math.sin(math.pi - tet_angle)
    atoms['N'] = ca_pos + normalize(n_dir) * BOND['CA-N']

    # CC position: ~109.47° from both CB and N
    # Use the third tetrahedral position
    atoms['CC'] = place_tetrahedral_third(
        ca_pos, cb_pos, atoms['N'], BOND['C-C'],
        prefer_dir=away_from_lig
    )

    # HA: fourth tetrahedral position around CA
    atoms['HA'] = place_tetrahedral_fourth(ca_pos, [atoms['N'], cb_pos, atoms['CC']], BOND['C-H'])

    # === N-terminal: CH3-NH ===
    # N has 3 bonds: to CA, to H, and to CN (the cap methyl carbon)
    n_to_ca = normalize(ca_pos - atoms['N'])

    # CN position: opposite to CA
    cn_dir = -n_to_ca
    atoms['CN'] = atoms['N'] + cn_dir * BOND['N-C']

    # H on N: third position in sp3 geometry
    atoms['H'] = place_tetrahedral_third(
        atoms['N'], ca_pos, atoms['CN'], BOND['N-H'],
        prefer_dir=away_from_lig
    )

    # HN1, HN2, HN3: hydrogens on CN (methyl)
    hn_positions = build_tetrahedral_h(
        atoms['CN'], atoms['N'], BOND['C-H'], 3,
        initial_angle=n_term_rotation
    )
    atoms['HN1'] = hn_positions[0]
    atoms['HN2'] = hn_positions[1]
    atoms['HN3'] = hn_positions[2]

    # === C-terminal: CH3 directly on CA ===
    # HC1, HC2, HC3: hydrogens on CC
    hc_positions = build_tetrahedral_h(
        atoms['CC'], ca_pos, BOND['C-H'], 3,
        initial_angle=c_term_rotation
    )
    atoms['HC1'] = hc_positions[0]
    atoms['HC2'] = hc_positions[1]
    atoms['HC3'] = hc_positions[2]

    return atoms


def calculate_min_cap_ligand_distance(
    cap_atoms: Dict[str, np.ndarray],
    ligand_atoms: List[Dict],
    cap_type: str = "methyl"
) -> float:
    """Calculate minimum distance between cap atoms and ligand atoms."""
    # Cap atoms that we want to keep far from ligand
    cap_names = {'CN', 'HN1', 'HN2', 'HN3', 'CC', 'HC1', 'HC2', 'HC3'}

    min_dist = float('inf')

    for cap_name in cap_names:
        if cap_name not in cap_atoms:
            continue
        cap_pos = cap_atoms[cap_name]

        for lig_atom in ligand_atoms:
            lig_pos = np.array([lig_atom["x"], lig_atom["y"], lig_atom["z"]])
            dist = np.linalg.norm(cap_pos - lig_pos)
            if dist < min_dist:
                min_dist = dist

    return min_dist


def optimize_cap_conformations(
    sg_pos: np.ndarray,
    cb_pos: np.ndarray,
    ca_pos: np.ndarray,
    n_pos: Optional[np.ndarray],
    c_pos: Optional[np.ndarray],
    o_pos: Optional[np.ndarray],
    beta_pos: np.ndarray,
    ligand_atoms: List[Dict],
    n_samples: int = 36,
    cap_type: str = "methyl"
) -> Tuple[Dict[str, np.ndarray], float, float, float, float]:
    """
    Sample cap conformations and select the one with maximum ligand distance.

    Args:
        cap_type: "methyl" (with carbonyl) or "aliphatic" (no carbonyl)

    Returns:
        (best_atoms, best_n_rot, best_c_rot, best_min_dist, baseline_dist)
    """
    print(f"Cap type: {cap_type}")
    print(f"Sampling {n_samples} conformations for each cap (total: {n_samples**2})...")

    best_atoms = None
    best_min_dist = 0.0
    best_n_rot = 0.0
    best_c_rot = 0.0

    # Sample rotations for N-terminal and C-terminal caps
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)

    for n_rot in angles:
        for c_rot in angles:
            # Build fragment with these rotations
            if cap_type == "aliphatic":
                atoms = build_aliphatic_cys_fragment(
                    sg_pos, cb_pos, ca_pos,
                    ligand_pos=beta_pos,
                    n_term_rotation=n_rot,
                    c_term_rotation=c_rot
                )
            else:
                atoms = build_methyl_cys_fragment(
                    sg_pos, cb_pos, ca_pos, n_pos, c_pos, o_pos,
                    ligand_pos=beta_pos,
                    n_term_rotation=n_rot,
                    c_term_rotation=c_rot
                )

            # Calculate minimum distance to ligand
            min_dist = calculate_min_cap_ligand_distance(atoms, ligand_atoms, cap_type)

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_atoms = atoms
                best_n_rot = n_rot
                best_c_rot = c_rot

    print(f"\nOptimal conformation found:")
    print(f"  N-cap CH3 rotation: {math.degrees(best_n_rot):.1f}°")
    print(f"  C-cap CH3 rotation: {math.degrees(best_c_rot):.1f}°")
    print(f"  Minimum cap-ligand distance: {best_min_dist:.2f} Å")

    # Calculate baseline (no rotation)
    if cap_type == "aliphatic":
        baseline_atoms = build_aliphatic_cys_fragment(
            sg_pos, cb_pos, ca_pos,
            ligand_pos=beta_pos,
            n_term_rotation=0.0,
            c_term_rotation=0.0
        )
    else:
        baseline_atoms = build_methyl_cys_fragment(
            sg_pos, cb_pos, ca_pos, n_pos, c_pos, o_pos,
            ligand_pos=beta_pos,
            n_term_rotation=0.0,
            c_term_rotation=0.0
        )
    baseline_dist = calculate_min_cap_ligand_distance(baseline_atoms, ligand_atoms, cap_type)
    improvement = best_min_dist - baseline_dist

    print(f"  Baseline distance: {baseline_dist:.2f} Å")
    if baseline_dist > 0:
        print(f"  Improvement: {improvement:+.2f} Å ({100*improvement/baseline_dist:+.1f}%)")
    else:
        print(f"  Improvement: {improvement:+.2f} Å")

    return best_atoms, best_n_rot, best_c_rot, best_min_dist, baseline_dist


# ============================================================================
# MOL2 atom type fixer
# ============================================================================

def fix_mol2_atom_types(mol2_in: str, mol2_out: str, cap_type: str = "methyl") -> None:
    """Ensure cap atoms have correct SYBYL types in MOL2."""
    with open(mol2_in) as f:
        lines = f.readlines()

    if cap_type == "aliphatic":
        # Aliphatic: CH3-NH-CA-CH3 (no carbonyl, N is sp3 amine)
        cap_types = {
            'SG': 'S.3', 'CB': 'C.3', 'CA': 'C.3',
            'N': 'N.3',   # sp3 amine (not amide)
            'H': 'H', 'HA': 'H', 'HB2': 'H', 'HB3': 'H',
            'CN': 'C.3', 'HN1': 'H', 'HN2': 'H', 'HN3': 'H',
            'CC': 'C.3', 'HC1': 'H', 'HC2': 'H', 'HC3': 'H',
        }
    else:
        # Methyl: CH3-N-CA-C(=O)-CH3 (with carbonyl)
        cap_types = {
            'SG': 'S.3', 'CB': 'C.3', 'CA': 'C.3',
            'N': 'N.am', 'C': 'C.2', 'O': 'O.2',
            'H': 'H', 'HA': 'H', 'HB2': 'H', 'HB3': 'H',
            'CN': 'C.3', 'HN1': 'H', 'HN2': 'H', 'HN3': 'H',
            'CC': 'C.3', 'HC1': 'H', 'HC2': 'H', 'HC3': 'H',
        }

    out = []
    in_atoms = False
    for line in lines:
        stripped = line.strip()
        if stripped == "@<TRIPOS>ATOM":
            in_atoms = True
            out.append(line)
            continue
        if stripped.startswith("@<TRIPOS>") and stripped != "@<TRIPOS>ATOM":
            in_atoms = False
            out.append(line)
            continue
        if in_atoms and stripped:
            parts = line.split()
            if len(parts) >= 6:
                name = parts[1]
                if name in cap_types:
                    parts[5] = cap_types[name]
                    if len(parts) >= 9:
                        fixed = (f"{int(parts[0]):7d} {parts[1]:<4s}"
                                 f"{float(parts[2]):11.4f}{float(parts[3]):9.4f}{float(parts[4]):9.4f} "
                                 f"{parts[5]:<6s}{int(parts[6]):5d} {parts[7]:<8s}{float(parts[8]):11.4f}\n")
                    else:
                        fixed = " ".join(parts) + "\n"
                    out.append(fixed)
                    continue
        out.append(line)

    with open(mol2_out, "w") as f:
        f.writelines(out)


# ============================================================================
# Main function
# ============================================================================

def build_optimized_fragment(
    complex_pdb: str,
    cys_resname: str,
    cys_resid: int,
    lig_resname: str,
    lig_resid: int,
    out_prefix: str,
    n_samples: int = 36,
    cap_type: str = "methyl",
    make_mol2: bool = False,
    gen3d: bool = False,
    ph: float = 7.4,
    protonate: bool = False
):
    """Build ligand + optimized capped CYS fragment.

    Args:
        cap_type: "methyl" (with carbonyl) or "aliphatic" (no carbonyl)
    """
    from pathlib import Path

    # Create output directory
    out_prefix_path = Path(out_prefix)
    if out_prefix_path.parent != Path('.'):
        out_prefix_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {out_prefix_path.parent}\n")

    atoms, conects = parse_pdb(complex_pdb)

    # Get ligand atoms
    lig_atoms = select_ligand(atoms, lig_resname, lig_resid)
    if not lig_atoms:
        raise SystemExit(f"ERROR: No ligand atoms found for {lig_resname} {lig_resid}")
    print(f"Found {len(lig_atoms)} ligand atoms")

    # Get cysteine backbone atoms
    sg = find_cys_atom(atoms, cys_resname, cys_resid, "SG")
    cb = find_cys_atom(atoms, cys_resname, cys_resid, "CB")
    ca = find_cys_atom(atoms, cys_resname, cys_resid, "CA")
    n_atom = find_cys_atom(atoms, cys_resname, cys_resid, "N")
    c_atom = find_cys_atom(atoms, cys_resname, cys_resid, "C")
    o_atom = find_cys_atom(atoms, cys_resname, cys_resid, "O")

    if not all([sg, cb, ca]):
        raise SystemExit(f"ERROR: Missing cysteine core atoms (SG, CB, CA)")

    if cap_type == "methyl" and not all([n_atom, c_atom]):
        raise SystemExit(f"ERROR: Missing backbone N/C atoms (required for methyl cap type)")

    found_atoms = "SG, CB, CA"
    if n_atom:
        found_atoms += ", N"
    if c_atom:
        found_atoms += ", C"
    if o_atom:
        found_atoms += ", O"
    print(f"Found CYS atoms: {found_atoms}")

    # Find ligand β-carbon
    lig_serials = {a["serial"] for a in lig_atoms}
    beta_serial = find_beta_from_conect(conects, sg["serial"], lig_serials)
    beta_atom = None

    if beta_serial:
        for a in lig_atoms:
            if a["serial"] == beta_serial:
                beta_atom = a
                break

    if beta_atom is None:
        sg_pos = np.array([sg["x"], sg["y"], sg["z"]])
        best_dist, best = 1e9, None
        for a in lig_atoms:
            if a["elem"].upper() in ("H", "D"):
                continue
            d = np.linalg.norm(np.array([a["x"], a["y"], a["z"]]) - sg_pos)
            if d < best_dist:
                best_dist, best = d, a
        beta_atom = best
        if beta_atom:
            beta_serial = beta_atom["serial"]
            print(f"Using nearest heavy atom: {beta_atom['name'].strip()} (dist={best_dist:.2f} Å)")

    if beta_atom is None:
        raise SystemExit("ERROR: Could not resolve β atom")

    print(f"Ligand attachment: {beta_atom['name'].strip()}\n")

    # Get positions
    sg_pos = np.array([sg["x"], sg["y"], sg["z"]])
    cb_pos = np.array([cb["x"], cb["y"], cb["z"]])
    ca_pos = np.array([ca["x"], ca["y"], ca["z"]])
    n_pos = np.array([n_atom["x"], n_atom["y"], n_atom["z"]]) if n_atom else None
    c_pos = np.array([c_atom["x"], c_atom["y"], c_atom["z"]]) if c_atom else None
    o_pos = np.array([o_atom["x"], o_atom["y"], o_atom["z"]]) if o_atom else None
    beta_pos = np.array([beta_atom["x"], beta_atom["y"], beta_atom["z"]])

    # OPTIMIZE CAP CONFORMATIONS
    cap_atoms, n_rot, c_rot, min_dist, baseline_dist = optimize_cap_conformations(
        sg_pos, cb_pos, ca_pos, n_pos, c_pos, o_pos,
        beta_pos, lig_atoms, n_samples, cap_type
    )

    # Build output
    new_atoms = []
    serial_counter = 1
    old_to_new = {}

    lig_resn = lig_resname[:3].upper()
    chain = "A"
    cap_resi = 1
    lig_resi = 2

    # Atom order depends on cap type
    if cap_type == "aliphatic":
        # CH3-NH-CA(-CB-SG)-CH3 (no C=O)
        cap_order = [
            ('CN', 'C'), ('HN1', 'H'), ('HN2', 'H'), ('HN3', 'H'),
            ('N', 'N'), ('H', 'H'),
            ('CA', 'C'), ('HA', 'H'),
            ('CB', 'C'), ('HB2', 'H'), ('HB3', 'H'),
            ('SG', 'S'),
            ('CC', 'C'), ('HC1', 'H'), ('HC2', 'H'), ('HC3', 'H'),
        ]
    else:
        # CH3-N-CA-C(=O)-CH3 (with carbonyl)
        cap_order = [
            ('CN', 'C'), ('HN1', 'H'), ('HN2', 'H'), ('HN3', 'H'),
            ('N', 'N'), ('H', 'H'),
            ('CA', 'C'), ('HA', 'H'),
            ('CB', 'C'), ('HB2', 'H'), ('HB3', 'H'),
            ('SG', 'S'),
            ('C', 'C'), ('O', 'O'),
            ('CC', 'C'), ('HC1', 'H'), ('HC2', 'H'), ('HC3', 'H'),
        ]

    cap_serials = {}
    for name, elem in cap_order:
        if name not in cap_atoms:
            continue
        pos = cap_atoms[name]
        new_atoms.append({
            "serial": serial_counter,
            "name": name,
            "resn": "CYS",
            "chain": chain,
            "resi": cap_resi,
            "x": pos[0], "y": pos[1], "z": pos[2],
            "elem": elem,
            "rec": "ATOM"
        })
        cap_serials[name] = serial_counter
        serial_counter += 1

    # Add ligand
    for a in lig_atoms:
        name = a["name"].strip()[:4]
        elem = a["elem"].strip().upper()
        if len(elem) > 1:
            elem = elem[0] + elem[1:].lower()
        new_atoms.append({
            "serial": serial_counter,
            "name": name,
            "resn": lig_resn,
            "chain": chain,
            "resi": lig_resi,
            "x": a["x"], "y": a["y"], "z": a["z"],
            "elem": elem,
            "rec": "HETATM"
        })
        old_to_new[a["serial"]] = serial_counter
        serial_counter += 1

    beta_new_serial = old_to_new.get(beta_serial)
    sg_serial = cap_serials['SG']

    # Write PDB
    out_pdb = f"{out_prefix}.pdb"
    lines = []

    for na in new_atoms:
        lines.append(format_pdb_line(
            serial=na["serial"],
            name=na["name"],
            resn=na["resn"],
            chain=na["chain"],
            resi=na["resi"],
            x=na["x"], y=na["y"], z=na["z"],
            elem=na["elem"],
            rec=na["rec"]
        ) + "\n")

    lines.append("TER\n")

    # CONECT records - different for aliphatic vs methyl
    if cap_type == "aliphatic":
        # CH3-NH-CA(-CB-SG)-CH3
        lines.append(f"CONECT{cap_serials['CN']:5d}{cap_serials['N']:5d}"
                     f"{cap_serials['HN1']:5d}{cap_serials['HN2']:5d}{cap_serials['HN3']:5d}\n")
        lines.append(f"CONECT{cap_serials['N']:5d}{cap_serials['CN']:5d}"
                     f"{cap_serials['CA']:5d}{cap_serials['H']:5d}\n")
        lines.append(f"CONECT{cap_serials['CA']:5d}{cap_serials['N']:5d}"
                     f"{cap_serials['CB']:5d}{cap_serials['CC']:5d}{cap_serials['HA']:5d}\n")
        lines.append(f"CONECT{cap_serials['CB']:5d}{cap_serials['CA']:5d}"
                     f"{cap_serials['SG']:5d}{cap_serials['HB2']:5d}{cap_serials['HB3']:5d}\n")
        lines.append(f"CONECT{cap_serials['SG']:5d}{cap_serials['CB']:5d}{beta_new_serial:5d}\n")
        lines.append(f"CONECT{beta_new_serial:5d}{cap_serials['SG']:5d}\n")
        lines.append(f"CONECT{cap_serials['CC']:5d}{cap_serials['CA']:5d}"
                     f"{cap_serials['HC1']:5d}{cap_serials['HC2']:5d}{cap_serials['HC3']:5d}\n")
    else:
        # CH3-N-CA-C(=O)-CH3
        lines.append(f"CONECT{cap_serials['CN']:5d}{cap_serials['N']:5d}"
                     f"{cap_serials['HN1']:5d}{cap_serials['HN2']:5d}{cap_serials['HN3']:5d}\n")
        lines.append(f"CONECT{cap_serials['N']:5d}{cap_serials['CN']:5d}"
                     f"{cap_serials['CA']:5d}{cap_serials['H']:5d}\n")
        lines.append(f"CONECT{cap_serials['CA']:5d}{cap_serials['N']:5d}"
                     f"{cap_serials['CB']:5d}{cap_serials['C']:5d}{cap_serials['HA']:5d}\n")
        lines.append(f"CONECT{cap_serials['CB']:5d}{cap_serials['CA']:5d}"
                     f"{cap_serials['SG']:5d}{cap_serials['HB2']:5d}{cap_serials['HB3']:5d}\n")
        lines.append(f"CONECT{cap_serials['C']:5d}{cap_serials['CA']:5d}"
                     f"{cap_serials['O']:5d}{cap_serials['CC']:5d}\n")
        lines.append(f"CONECT{cap_serials['SG']:5d}{cap_serials['CB']:5d}{beta_new_serial:5d}\n")
        lines.append(f"CONECT{beta_new_serial:5d}{cap_serials['SG']:5d}\n")
        lines.append(f"CONECT{cap_serials['CC']:5d}{cap_serials['C']:5d}"
                     f"{cap_serials['HC1']:5d}{cap_serials['HC2']:5d}{cap_serials['HC3']:5d}\n")

    lines.append("END\n")

    with open(out_pdb, "w") as w:
        w.writelines(lines)

    beta_name = beta_atom["name"].strip()
    cap_desc = "aliphatic (CH3-NH-CA-CH3, no carbonyl)" if cap_type == "aliphatic" else "methyl (CH3-N-CA-C=O-CH3)"
    print(f"\nWrote optimized fragment: {out_pdb}")
    print(f"  Cap type: {cap_desc}")
    print(f"  Cap residue: CYS (residue {cap_resi})")
    print(f"  Ligand residue: {lig_resn} (residue {lig_resi})")

    # Save metadata
    meta = {
        "complex_pdb": complex_pdb,
        "cys_resname": cys_resname,
        "cys_resid": cys_resid,
        "lig_resname": lig_resname,
        "lig_resid": lig_resid,
        "cap_type": f"{'ALIPHATIC' if cap_type == 'aliphatic' else 'METHYL'}-OPTIMIZED",
        "cap_structure": "CH3-NH-CA-CH3 (no carbonyl)" if cap_type == "aliphatic" else "CH3-N-CA-C(=O)-CH3",
        "optimization": {
            "n_samples": n_samples,
            "n_cap_rotation_deg": math.degrees(n_rot),
            "c_cap_rotation_deg": math.degrees(c_rot),
            "min_cap_ligand_distance_angstrom": min_dist,
            "baseline_distance_angstrom": baseline_dist,
            "improvement_angstrom": min_dist - baseline_dist
        },
        "cap_serials": cap_serials,
        "sg_serial": sg_serial,
        "beta_serial_new": beta_new_serial,
        "beta_name": beta_name,
        "fragment_pdb": out_pdb,
    }
    with open(f"{out_prefix}_meta.json", "w") as w:
        json.dump(meta, w, indent=2)
    print(f"Wrote metadata: {out_prefix}_meta.json")

    # MOL2 conversion
    if make_mol2:
        obabel = shutil.which("obabel")
        if obabel:
            out_mol2 = f"{out_prefix}.mol2"
            cmd = ["obabel", "-i", "pdb", out_pdb, "-o", "mol2", "-O", out_mol2, "-h"]
            if gen3d:
                cmd.extend(["--gen3d"])
            subprocess.run(cmd, capture_output=True, text=True)
            if Path(out_mol2).exists():
                print(f"Wrote MOL2: {out_mol2}")

                fixed_mol2 = f"{out_prefix}_fixed.mol2"
                fix_mol2_atom_types(out_mol2, fixed_mol2, cap_type)
                print(f"Fixed atom types: {fixed_mol2}")

                if protonate:
                    prot_mol2 = f"{out_prefix}_pH{ph:.1f}.mol2"
                    cmd2 = ["obabel", "-i", "mol2", fixed_mol2, "-o", "mol2", "-O", prot_mol2,
                            "-p", str(ph), "--partialcharge", "none"]
                    subprocess.run(cmd2, capture_output=True, text=True)
                    if Path(prot_mol2).exists():
                        print(f"Wrote protonated MOL2: {prot_mol2}")


def main():
    ap = argparse.ArgumentParser(
        description="Extract covalent fragment with OPTIMIZED caps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Cap types:
  methyl     CH3-N-CA-C(=O)-CH3  Keeps backbone carbonyl (default)
  aliphatic  CH3-NH-CA-CH3       No carbonyl - pure aliphatic caps
                                 Better for minimizing electronic polarization

Example:
  python b04_extract_adduct_fragment_methyl_optimized.py \\
    --complex complex.pdb \\
    --cys-resid 1039 \\
    --cap-type aliphatic \\
    --out-prefix Outputs/adduct
"""
    )
    ap.add_argument("--complex", required=True, help="Path to complex.pdb")
    ap.add_argument("--cys-resname", default="CYS", help="Cysteine residue name")
    ap.add_argument("--cys-resid", type=int, required=True, help="Cysteine residue number")
    ap.add_argument("--lig-resname", default="LIG", help="Ligand residue name")
    ap.add_argument("--lig-resid", type=int, default=1, help="Ligand residue number")
    ap.add_argument("--out-prefix", default="adduct_opt", help="Output prefix (creates dirs if needed)")
    ap.add_argument("--cap-type", choices=["methyl", "aliphatic"], default="methyl",
                   help="Cap type: 'methyl' (with C=O) or 'aliphatic' (no carbonyl)")
    ap.add_argument("--n-samples", type=int, default=36,
                   help="Number of rotation samples per cap (default: 36, 10° steps)")
    ap.add_argument("--make-mol2", action="store_true", help="Convert to MOL2")
    ap.add_argument("--gen3d", action="store_true", help="Use --gen3d in OpenBabel")
    ap.add_argument("--ph", type=float, default=7.4, help="pH for protonation")
    ap.add_argument("--protonate", action="store_true", help="Protonate MOL2")
    args = ap.parse_args()

    build_optimized_fragment(
        complex_pdb=args.complex,
        cys_resname=args.cys_resname,
        cys_resid=args.cys_resid,
        lig_resname=args.lig_resname,
        lig_resid=args.lig_resid,
        out_prefix=args.out_prefix,
        n_samples=args.n_samples,
        cap_type=args.cap_type,
        make_mol2=args.make_mol2,
        gen3d=args.gen3d,
        ph=args.ph,
        protonate=args.protonate
    )


if __name__ == "__main__":
    main()
