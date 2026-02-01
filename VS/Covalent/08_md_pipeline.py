#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B_covalent_md_pipeline.py

Unified pipeline for covalent ligand MD setup.

This script automates the complete workflow:
1. Extract capped adduct fragment from docked complex (b04)
2. Parameterize with acpype (GAFF2 charges)
3. Create CYL residue and modified force field (b05)
4. Add GAFF2 bonded parameters to force field
5. Assemble complex with CYL residue (b06)
6. GROMACS setup: pdb2gmx, solvate, ions, EM

Usage:
    python B_covalent_md_pipeline.py \\
        --complex Outputs/Covalent/docking/complex.pdb \\
        --cys-resid 1039 \\
        --output-dir Outputs/Covalent/md_prep

Requirements:
    - OpenBabel (obabel) for format conversion
    - acpype for parameterization
    - GROMACS (gmx) for MD setup
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Import functions from component scripts
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Handle imports with numeric prefixes
import importlib
_extract = importlib.import_module("05_extract_adduct")
build_optimized_fragment = _extract.build_optimized_fragment
_assemble = importlib.import_module("07_assemble_complex")
assemble_covalent_complex = _assemble.assemble_covalent_complex


def run_cmd(cmd, cwd=None, check=True, env=None, capture=False):
    """Run a shell command."""
    print(f"  $ {cmd}")
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    if capture:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd,
            check=check, env=run_env,
            executable='/bin/bash',
            capture_output=True, text=True
        )
        return result
    else:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd,
            check=check, env=run_env,
            executable='/bin/bash'
        )
        return result.returncode == 0


def run_python_script(script_path, args, cwd=None):
    """Run a Python script with arguments."""
    python = sys.executable
    cmd = f"{python} {script_path} {args}"
    print(f"\n>>> Running: {script_path.name}")
    return run_cmd(cmd, cwd=cwd)


def run_acpype(mol2_file, output_dir, net_charge=0, charge_method='bcc', timeout=180, 
               gasteiger_fallback=True):
    """Run acpype to parameterize the adduct.
    
    Args:
        mol2_file: Input mol2 file
        output_dir: Output directory for acpype results
        net_charge: Net charge of the molecule
        charge_method: Charge method ('bcc' for AM1-BCC, 'gas' for Gasteiger)
        timeout: Timeout in seconds for AM1-BCC (default: 180s = 3 min)
        gasteiger_fallback: If True, fallback to Gasteiger if BCC times out
    
    Returns:
        Tuple of (acpype_dir, charge_method_used)
    """
    print(f"\n  Running acpype on {mol2_file.name} (charge method: {charge_method})...")

    # acpype needs the file in current directory or absolute path
    mol2_abs = mol2_file.resolve()
    stem = mol2_file.stem
    
    # Clean up any previous acpype output
    old_acpype = output_dir / f"{stem}.acpype"
    if old_acpype.exists():
        shutil.rmtree(old_acpype)

    # Run acpype with specified charge method
    cmd = f"acpype -i {mol2_abs} -c {charge_method} -n {net_charge} -a gaff2"
    
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=output_dir,
            capture_output=True, text=True,
            timeout=timeout if charge_method == 'bcc' else None,
            executable='/bin/bash'
        )
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT: AM1-BCC took longer than {timeout}s")
        success = False
        # Kill any lingering sqm processes
        subprocess.run("pkill -9 -f sqm", shell=True, capture_output=True)
    
    # Find the output directory
    acpype_dir = output_dir / f"{stem}.acpype"
    if not acpype_dir.exists():
        for d in output_dir.iterdir():
            if d.is_dir() and d.name.endswith('.acpype'):
                acpype_dir = d
                break
    
    # Check if parameterization succeeded
    gmx_itp = acpype_dir / f"{stem}_GMX.itp" if acpype_dir.exists() else None
    if gmx_itp and gmx_itp.exists():
        print(f"    ✓ Parameterization succeeded with {charge_method.upper()} charges")
        return acpype_dir, charge_method
    
    # Fallback to Gasteiger if BCC failed and fallback is enabled
    if charge_method == 'bcc' and gasteiger_fallback:
        print(f"    Falling back to Gasteiger charges...")
        
        # Clean up failed BCC attempt
        if acpype_dir.exists():
            shutil.rmtree(acpype_dir)
        # Also clean up any temp files
        for tmp in output_dir.glob(".acpype_tmp*"):
            shutil.rmtree(tmp, ignore_errors=True)
        
        return run_acpype(mol2_file, output_dir, net_charge, 
                         charge_method='gas', timeout=None, gasteiger_fallback=False)
    
    return acpype_dir, charge_method


def get_gaff2_mass_from_typename(typename):
    """
    Infer mass and atomic number from GAFF2 atom type name.
    GAFF2 naming: first letter(s) indicate element.
    """
    # GAFF2 element prefixes and their masses/atomic numbers
    gaff2_elements = {
        'h': (1.008, 1),    # hydrogen
        'c': (12.01, 6),    # carbon
        'n': (14.01, 7),    # nitrogen
        'o': (16.00, 8),    # oxygen
        'f': (19.00, 9),    # fluorine
        's': (32.06, 16),   # sulfur
        'p': (30.97, 15),   # phosphorus
        'cl': (35.45, 17),  # chlorine
        'br': (79.90, 35),  # bromine
        'i': (126.9, 53),   # iodine
    }

    name_lower = typename.lower()

    # Check for two-letter elements first (cl, br)
    if name_lower.startswith('cl'):
        return gaff2_elements['cl']
    elif name_lower.startswith('br'):
        return gaff2_elements['br']

    # Then single letter
    first_char = name_lower[0] if name_lower else ''
    if first_char in gaff2_elements:
        return gaff2_elements[first_char]

    # Default to carbon if unknown
    return (12.01, 6)


def parse_gmx_top_atomtypes(top_file):
    """
    Parse atomtypes from acpype GMX.top file.
    Returns dict of {typename: {'mass': float, 'at_num': int, 'sigma': float, 'epsilon': float}}

    Note: acpype often sets mass=0 in [ atomtypes ] (masses come from [ atoms ] section).
    We infer masses from GAFF2 naming conventions when mass is 0.
    """
    atomtypes = {}
    with open(top_file) as f:
        content = f.read()

    # Find [ atomtypes ] section
    at_match = re.search(r'\[ atomtypes \]\s*\n(.*?)(?:\n\[|\Z)', content, re.DOTALL)
    if at_match:
        for line in at_match.group(1).strip().split('\n'):
            if line.strip() and not line.startswith(';'):
                parts = line.split()
                # Format variations:
                # name bond_type mass charge ptype sigma epsilon  (acpype style)
                # name at_num mass charge ptype sigma epsilon
                # name mass charge ptype sigma epsilon (older)
                if len(parts) >= 6:
                    name = parts[0]
                    sigma = None
                    epsilon = None
                    mass = 0.0
                    at_num = 0

                    try:
                        # Try acpype format: name bond_type mass charge ptype sigma epsilon
                        # bond_type is often same as name, mass is often 0.0
                        if len(parts) >= 7:
                            # Check if parts[1] looks like a bond type (string) or at_num (int)
                            try:
                                at_num = int(parts[1])
                                # It's an integer, so format is: name at_num mass charge ptype sigma epsilon
                                mass = float(parts[2])
                                sigma = float(parts[5])
                                epsilon = float(parts[6])
                            except ValueError:
                                # parts[1] is a string (bond_type), so acpype format
                                mass = float(parts[2])
                                sigma = float(parts[5])
                                epsilon = float(parts[6])
                        else:
                            # 6-column format: name mass charge ptype sigma epsilon
                            mass = float(parts[1])
                            sigma = float(parts[4])
                            epsilon = float(parts[5])
                    except (ValueError, IndexError):
                        continue

                    if sigma is None or epsilon is None:
                        continue

                    # If mass is 0 or very small, infer from atom type name
                    if mass < 0.5:
                        mass, at_num = get_gaff2_mass_from_typename(name)

                    atomtypes[name] = {
                        'mass': mass,
                        'at_num': at_num,
                        'sigma': sigma,
                        'epsilon': epsilon
                    }
    return atomtypes


def parse_lig_gmx_itp(itp_file):
    """Parse acpype GMX.itp to extract atom info and bonded parameters.

    Returns atom_info dict with both type and name for crossterm generation.
    Also returns raw bonded parameters with atom indices for crossterm generation.
    """
    with open(itp_file) as f:
        content = f.read()

    # Parse atoms section: index -> {type, name} mapping
    atom_info = {}
    atoms_match = re.search(r'\[ atoms \]\s*\n(.*?)(?:\n\[|\Z)', content, re.DOTALL)
    if atoms_match:
        for line in atoms_match.group(1).strip().split('\n'):
            if line.strip() and not line.startswith(';'):
                parts = line.split()
                # Format: nr type resi res atom cgnr charge mass
                if len(parts) >= 5:
                    idx = int(parts[0])
                    atype = parts[1]
                    aname = parts[4]
                    atom_info[idx] = {'type': atype, 'name': aname}

    # Build simple index -> type mapping for backward compatibility
    atom_types = {idx: info['type'] for idx, info in atom_info.items()}

    # Parse bonds from [ bonds ] section - keep BOTH type-based and index-based
    bonds = []
    bonds_raw = []  # Raw bonds with indices for crossterm generation
    bonds_match = re.search(r'\[ bonds \]\s*\n(.*?)(?:\n\[|\Z)', content, re.DOTALL)
    if bonds_match:
        for line in bonds_match.group(1).strip().split('\n'):
            if line.strip() and not line.startswith(';'):
                parts = line.split()
                if len(parts) >= 5:
                    i, j = int(parts[0]), int(parts[1])
                    funct = int(parts[2])
                    r = float(parts[3])
                    k = float(parts[4])
                    ti, tj = atom_types.get(i, 'X'), atom_types.get(j, 'X')
                    # Store raw with indices
                    bonds_raw.append((i, j, funct, r, k))
                    # Store type-based (sorted)
                    if ti > tj:
                        ti, tj = tj, ti
                    bonds.append((ti, tj, funct, r, k))

    # Parse angles from [ angles ] section
    angles = []
    angles_raw = []
    angles_match = re.search(r'\[ angles \]\s*\n(.*?)(?:\n\[|\Z)', content, re.DOTALL)
    if angles_match:
        for line in angles_match.group(1).strip().split('\n'):
            if line.strip() and not line.startswith(';'):
                parts = line.split()
                if len(parts) >= 6:
                    i, j, k = int(parts[0]), int(parts[1]), int(parts[2])
                    funct = int(parts[3])
                    theta = float(parts[4])
                    cth = float(parts[5])
                    ti, tj, tk = atom_types.get(i, 'X'), atom_types.get(j, 'X'), atom_types.get(k, 'X')
                    # Store raw
                    angles_raw.append((i, j, k, funct, theta, cth))
                    # Store type-based (sorted endpoints)
                    if ti > tk:
                        ti, tk = tk, ti
                    angles.append((ti, tj, tk, funct, theta, cth))

    # Parse dihedrals - acpype uses "[ dihedrals ] ; propers" format
    propers = []
    propers_raw = []
    # Try both formats: with and without the "; propers" comment
    dih_match = re.search(r'\[ dihedrals \].*?; propers\s*\n(.*?)(?:\n\[ dihedrals \]|\Z)', content, re.DOTALL)
    if not dih_match:
        # Try without the comment - just first dihedrals section
        dih_match = re.search(r'\[ dihedrals \]\s*\n(.*?)(?:\n\[|\Z)', content, re.DOTALL)

    if dih_match:
        for line in dih_match.group(1).strip().split('\n'):
            if line.strip() and not line.startswith(';'):
                parts = line.split()
                if len(parts) >= 8:
                    i, j, k, l = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    funct = int(parts[4])
                    phase = float(parts[5])
                    kd = float(parts[6])
                    pn = int(parts[7])
                    ti = atom_types.get(i, 'X')
                    tj = atom_types.get(j, 'X')
                    tk = atom_types.get(k, 'X')
                    tl = atom_types.get(l, 'X')
                    propers_raw.append((i, j, k, l, funct, phase, kd, pn))
                    propers.append((ti, tj, tk, tl, funct, phase, kd, pn))

    # Parse improper dihedrals - acpype uses "[ dihedrals ] ; impropers" format
    impropers = []
    impropers_raw = []
    imp_match = re.search(r'\[ dihedrals \].*?; impropers\s*\n(.*?)(?:\n\[|\Z)', content, re.DOTALL)
    if imp_match:
        for line in imp_match.group(1).strip().split('\n'):
            if line.strip() and not line.startswith(';'):
                parts = line.split()
                if len(parts) >= 8:
                    i, j, k, l = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    funct = int(parts[4])
                    phase = float(parts[5])
                    kd = float(parts[6])
                    pn = int(parts[7])
                    ti = atom_types.get(i, 'X')
                    tj = atom_types.get(j, 'X')
                    tk = atom_types.get(k, 'X')
                    tl = atom_types.get(l, 'X')
                    impropers_raw.append((i, j, k, l, funct, phase, kd, pn))
                    impropers.append((ti, tj, tk, tl, funct, phase, kd, pn))

    # Return both type-based and raw data
    raw_bonded = {
        'bonds': bonds_raw,
        'angles': angles_raw,
        'propers': propers_raw,
        'impropers': impropers_raw
    }

    return atom_info, bonds, angles, propers, impropers, raw_bonded


# Mapping from GAFF2 backbone atom types to AMBER types
# These are the types used for backbone atoms in the capped adduct mol2
# that need to be converted to AMBER types in the RTP
GAFF2_TO_AMBER_BACKBONE = {
    'n7': 'N',   # sp3 N in capped adduct -> amide N in backbone
    'c3': 'CT',  # sp3 C for CA, CB
    'ss': 'S',   # thioether S for SG
    'h1': 'H1',  # H on C with 1 EW group (HA, HB)
    'hn': 'H',   # H on N
}


def add_crossterm_bonded_params(atom_info, bonds, angles, propers, impropers, raw_bonded):
    """
    Add crossterm bonded parameters for AMBER-GAFF2 interfaces.

    The RTP uses AMBER types for backbone atoms but GAFF2 types for ligand.
    This creates crossterm bonded interactions at the SG-ligand interface.

    IMPORTANT: Only adds TRUE crossterms - parameters that have BOTH AMBER
    and GAFF2 atom types. Pure AMBER-AMBER types (CT-CT, CT-N, etc.) are
    already in the AMBER force field and should NOT be duplicated.

    Uses atom INDICES (not just types) to correctly identify which atoms
    are backbone vs ligand, since the same GAFF2 type (e.g., c3) can be
    used for both backbone (CA, CB) and ligand atoms (C1).
    """
    # Identify which atom INDICES are backbone atoms by name
    backbone_names = {'N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'SG'}
    backbone_indices = set()
    for idx, info in atom_info.items():
        if info['name'] in backbone_names:
            backbone_indices.add(idx)

    # Ligand atom indices = everything not backbone or cap
    ligand_indices = set(atom_info.keys()) - backbone_indices

    # Build mapping: atom index -> AMBER type for backbone atoms
    idx_to_amber_type = {}
    for idx in backbone_indices:
        gaff2_type = atom_info[idx]['type']
        if gaff2_type in GAFF2_TO_AMBER_BACKBONE:
            idx_to_amber_type[idx] = GAFF2_TO_AMBER_BACKBONE[gaff2_type]

    if not idx_to_amber_type:
        print("  No backbone atoms found that need AMBER mapping")
        return bonds, angles, propers, impropers

    # GAFF2 types for all atoms
    idx_to_gaff2_type = {idx: info['type'] for idx, info in atom_info.items()}

    print(f"  Backbone atom indices with AMBER types: {idx_to_amber_type}")

    def get_mixed_type(idx):
        """Get the atom type: AMBER for backbone, GAFF2 for ligand."""
        if idx in idx_to_amber_type:
            return idx_to_amber_type[idx]
        return idx_to_gaff2_type.get(idx, 'X')

    def is_true_crossterm(indices):
        """
        Check if this is a TRUE crossterm (has both backbone AND ligand atoms).

        We only want to add parameters that span the AMBER-GAFF2 boundary.
        Pure backbone (AMBER-AMBER) parameters are already in ffbonded.itp.
        Pure ligand (GAFF2-GAFF2) parameters are in the original bonds list.
        """
        has_backbone = any(idx in backbone_indices for idx in indices)
        has_ligand = any(idx in ligand_indices for idx in indices)
        return has_backbone and has_ligand

    # Generate crossterms from raw bonds - ONLY true crossterms
    crossterm_bonds = []
    for raw in raw_bonded['bonds']:
        i, j, funct, r, k = raw
        if is_true_crossterm([i, j]):
            ti, tj = get_mixed_type(i), get_mixed_type(j)
            if ti > tj:
                ti, tj = tj, ti
            crossterm_bonds.append((ti, tj, funct, r, k))

    # Generate crossterms from raw angles - ONLY true crossterms
    crossterm_angles = []
    for raw in raw_bonded['angles']:
        i, j, k, funct, theta, cth = raw
        if is_true_crossterm([i, j, k]):
            ti, tj, tk = get_mixed_type(i), get_mixed_type(j), get_mixed_type(k)
            if ti > tk:
                ti, tk = tk, ti
            crossterm_angles.append((ti, tj, tk, funct, theta, cth))

    # Generate crossterms from raw proper dihedrals - ONLY true crossterms
    crossterm_propers = []
    for raw in raw_bonded['propers']:
        i, j, k, l, funct, phase, kd, pn = raw
        if is_true_crossterm([i, j, k, l]):
            ti = get_mixed_type(i)
            tj = get_mixed_type(j)
            tk = get_mixed_type(k)
            tl = get_mixed_type(l)
            crossterm_propers.append((ti, tj, tk, tl, funct, phase, kd, pn))

    # Generate crossterms from raw improper dihedrals - ONLY true crossterms
    crossterm_impropers = []
    for raw in raw_bonded['impropers']:
        i, j, k, l, funct, phase, kd, pn = raw
        if is_true_crossterm([i, j, k, l]):
            ti = get_mixed_type(i)
            tj = get_mixed_type(j)
            tk = get_mixed_type(k)
            tl = get_mixed_type(l)
            crossterm_impropers.append((ti, tj, tk, tl, funct, phase, kd, pn))

    # Combine original GAFF2-only and crossterm parameters
    all_bonds = bonds + crossterm_bonds
    all_angles = angles + crossterm_angles
    all_propers = propers + crossterm_propers
    all_impropers = impropers + crossterm_impropers

    print(f"  Added crossterms: {len(crossterm_bonds)} bonds, {len(crossterm_angles)} angles, "
          f"{len(crossterm_propers)} propers, {len(crossterm_impropers)} impropers")

    return all_bonds, all_angles, all_propers, all_impropers


def deduplicate(items, n_key_fields):
    """Remove duplicates based on first n fields."""
    seen = set()
    unique = []
    for item in items:
        key = tuple(item[:n_key_fields])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def deduplicate_dihedrals(dihedrals):
    """Remove duplicate dihedrals - keep only ONE entry per atom type combo.

    GROMACS 2025 doesn't support multiple lines with different periodicities
    for the same atom types in [ dihedraltypes ]. We keep only the term with
    the largest force constant (most important contribution).

    For proper MD, a single dominant Fourier term is usually sufficient.
    """
    # Group by atom types
    dihedral_groups = {}
    for d in dihedrals:
        # d = (ti, tj, tk, tl, funct, phase, kd, pn)
        key = (d[0], d[1], d[2], d[3])  # Just atom types
        if key not in dihedral_groups:
            dihedral_groups[key] = []
        dihedral_groups[key].append(d)

    # For each group, keep the term with the largest force constant
    unique = []
    for key, group in dihedral_groups.items():
        # Sort by absolute force constant (kd at index 6), keep largest
        best = max(group, key=lambda x: abs(x[6]))
        unique.append(best)

    return unique


def write_gaff2_bonded(bonds, angles, propers, impropers, output_file):
    """Write GAFF2 bonded parameters to ITP file.

    All parameters come directly from acpype - no filtering or cross-term additions.
    """
    with open(output_file, 'w') as f:
        f.write("; GAFF2 bonded parameters for covalent ligand\n")
        f.write("; Generated from acpype GMX.itp - all parameters included\n;\n\n")

        # Bond types
        f.write("[ bondtypes ]\n")
        f.write("; i    j  func       b0          kb\n")
        for b in bonds:
            f.write(f"  {b[0]:<4s} {b[1]:<4s}  {b[2]}   {b[3]:.5e}  {b[4]:.5e}\n")

        # Angle types
        f.write("\n[ angletypes ]\n")
        f.write(";  i    j    k  func       th0         cth\n")
        for a in angles:
            f.write(f"  {a[0]:<4s} {a[1]:<4s} {a[2]:<4s}  {a[3]}   {a[4]:.4e}  {a[5]:.5e}\n")

        # Proper dihedral types
        f.write("\n[ dihedraltypes ]\n")
        f.write(";i    j    k    l  func      phase      kd         pn\n")
        for d in propers:
            f.write(f"  {d[0]:<4s} {d[1]:<4s} {d[2]:<4s} {d[3]:<4s}  {d[4]}   {d[5]:>6.2f}  {d[6]:.5e}  {d[7]}\n")

        # Improper dihedral types
        if impropers:
            f.write("\n; Improper dihedrals\n")
            f.write("[ dihedraltypes ]\n")
            for d in impropers:
                f.write(f"  {d[0]:<4s} {d[1]:<4s} {d[2]:<4s} {d[3]:<4s}  {d[4]}   {d[5]:>6.2f}  {d[6]:.5e}  {d[7]}\n")

    print(f"  Wrote {len(bonds)} bond types, {len(angles)} angle types, "
          f"{len(propers)} proper dihedrals, {len(impropers)} improper dihedrals")


def add_include_to_ffbonded(ff_dir):
    """Add include directive for gaff2_bonded.itp to ffbonded.itp."""
    ffbonded = ff_dir / 'ffbonded.itp'
    include_line = '#include "gaff2_bonded.itp"'

    with open(ffbonded) as f:
        content = f.read()

    if include_line in content:
        print("  gaff2_bonded.itp already included in ffbonded.itp")
        return

    lines = content.split('\n')
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('[ bondtypes ]'):
            insert_idx = i
            break

    lines.insert(insert_idx, f'\n; GAFF2 bonded parameters for covalent ligands\n{include_line}\n')

    with open(ffbonded, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  Added include directive to {ffbonded.name}")


def add_all_atomtypes_from_acpype(gmx_top, ff_dir):
    """
    Dynamically add ALL atom types from acpype GMX.top to force field.

    This ensures all GAFF2 atom types used in the ligand are available
    in the force field, without requiring hard-coded lists.
    """
    # Parse atomtypes from acpype output
    acpype_atomtypes = parse_gmx_top_atomtypes(gmx_top)
    print(f"  Found {len(acpype_atomtypes)} atomtypes in acpype output")

    # 1. Update gaff2_atomtypes.itp (LJ parameters)
    gaff2_file = ff_dir / "gaff2_atomtypes.itp"
    if not gaff2_file.exists():
        print(f"  WARNING: {gaff2_file} not found, creating new file")
        existing_gaff2_types = set()
        gaff2_content = "[ atomtypes ]\n; name  at_num  mass      charge  ptype  sigma        epsilon\n"
    else:
        with open(gaff2_file) as f:
            gaff2_content = f.read()
        # Find existing types
        existing_gaff2_types = set()
        for line in gaff2_content.split('\n'):
            if line.strip() and not line.startswith(';') and not line.startswith('['):
                parts = line.split()
                if parts:
                    existing_gaff2_types.add(parts[0])

    # Add missing types to gaff2_atomtypes.itp
    new_gaff2_lines = []
    for atype, params in acpype_atomtypes.items():
        if atype not in existing_gaff2_types:
            # Determine atomic number from mass if not available
            at_num = params.get('at_num', 0)
            if at_num == 0:
                mass = params['mass']
                if mass < 2:
                    at_num = 1  # H
                elif mass < 13:
                    at_num = 6  # C
                elif mass < 15:
                    at_num = 7  # N
                elif mass < 17:
                    at_num = 8  # O
                elif mass < 20:
                    at_num = 9  # F
                elif mass < 33:
                    at_num = 16  # S
                elif mass < 36:
                    at_num = 17  # Cl
                elif mass < 80:
                    at_num = 35  # Br
                else:
                    at_num = 53  # I

            new_gaff2_lines.append(
                f"  {atype:<6s}  {at_num:>3d}  {params['mass']:>10.4f}  0.0000  "
                f"A  {params['sigma']:.6e}  {params['epsilon']:.6e}"
            )

    if new_gaff2_lines:
        with open(gaff2_file, 'a') as f:
            f.write("\n; Additional GAFF2 types from acpype\n")
            for line in new_gaff2_lines:
                f.write(line + '\n')
        print(f"  Added {len(new_gaff2_lines)} atomtypes to gaff2_atomtypes.itp")

    # 2. Update atomtypes.atp (required for pdb2gmx)
    atp_file = ff_dir / "atomtypes.atp"
    if atp_file.exists():
        with open(atp_file) as f:
            atp_content = f.read()
        existing_atp_types = set()
        for line in atp_content.split('\n'):
            if line.strip() and not line.startswith(';'):
                parts = line.split()
                if parts:
                    existing_atp_types.add(parts[0])

        new_atp_lines = []
        for atype, params in acpype_atomtypes.items():
            if atype not in existing_atp_types:
                new_atp_lines.append(f"{atype:<6s}  {params['mass']:>10.5f}")

        if new_atp_lines:
            with open(atp_file, 'a') as f:
                f.write("; Additional GAFF2 types from acpype\n")
                for line in new_atp_lines:
                    f.write(line + '\n')
            print(f"  Added {len(new_atp_lines)} atomtypes to atomtypes.atp")

    if not new_gaff2_lines and not (atp_file.exists() and new_atp_lines):
        print(f"  All atomtypes already present in force field")


def add_gaff2_bonded_params(gmx_itp, ff_dir):
    """
    Add GAFF2 bonded parameters from GMX.itp to force field.

    All parameters from acpype are included without filtering.
    Atom types are dynamically added from the GMX.top file.

    Also adds crossterm parameters for AMBER backbone - GAFF2 ligand interfaces.
    """
    print(f"  Parsing {gmx_itp.name}...")
    atom_info, bonds, angles, propers, impropers, raw_bonded = parse_lig_gmx_itp(gmx_itp)

    print(f"  Found {len(bonds)} bonds, {len(angles)} angles, "
          f"{len(propers)} propers, {len(impropers)} impropers")

    # Add crossterm parameters for AMBER-GAFF2 interfaces
    # This handles the backbone (AMBER types) to ligand (GAFF2 types) connections
    print("  Adding crossterm parameters for AMBER-GAFF2 interface...")
    bonds, angles, propers, impropers = add_crossterm_bonded_params(
        atom_info, bonds, angles, propers, impropers, raw_bonded
    )

    # Deduplicate (no filtering - include all parameters)
    bonds = deduplicate(bonds, 2)
    angles = deduplicate(angles, 3)
    propers = deduplicate_dihedrals(propers)
    impropers = deduplicate_dihedrals(impropers)

    print(f"  After deduplication: {len(bonds)} bonds, {len(angles)} angles, "
          f"{len(propers)} propers, {len(impropers)} impropers")

    # Dynamically add all atom types from acpype GMX.itp
    # Note: acpype puts atomtypes in the .itp file, not .top (which just #includes the .itp)
    add_all_atomtypes_from_acpype(gmx_itp, ff_dir)

    # Write output
    output_file = ff_dir / "gaff2_bonded.itp"
    write_gaff2_bonded(bonds, angles, propers, impropers, output_file)
    print(f"  Wrote {output_file.name}")

    # Add include to ffbonded.itp
    add_include_to_ffbonded(ff_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Unified covalent ligand MD setup pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python B_covalent_md_pipeline.py \\
        --complex Outputs/Covalent/docking/complex.pdb \\
        --cys-resid 1039 \\
        --output-dir Outputs/Covalent/md_prep

This will:
    1. Extract capped CYS+ligand fragment from the complex
    2. Parameterize with acpype (GAFF2)
    3. Create custom CYS-ligand residue and modified force field
    4. Assemble complex with custom residue replacing CYS
    5. Run GROMACS pdb2gmx, solvate, add ions
    6. Generate EM input files

Use --cyl-resname to specify unique names (CX1, CX2, C32, etc.) for RBFE.
"""
    )

    # Input arguments
    parser.add_argument('--complex', required=True,
                        help='Docked complex PDB (protein + ligand)')
    parser.add_argument('--cys-resid', type=int, required=True,
                        help='Covalent cysteine residue number')
    parser.add_argument('--lig-resname', default='LIG',
                        help='Ligand residue name (default: LIG)')
    parser.add_argument('--lig-resid', type=int, default=1,
                        help='Ligand residue number (default: 1)')

    # Output arguments
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for all files')

    # Options
    parser.add_argument('--cyl-resname', default='CYL',
                        help='Residue name for covalent CYS-ligand (3 chars, default: CYL). '
                             'Use unique names for RBFE: CX1, CX2, C32, etc.')
    parser.add_argument('--cap-type', choices=['methyl', 'aliphatic'], default='aliphatic',
                        help='Cap type for fragment (default: aliphatic)')
    parser.add_argument('--net-charge', type=int, default=0,
                        help='Net charge of adduct (default: 0)')
    parser.add_argument('--gmx', default='gmx',
                        help='GROMACS executable (default: gmx)')
    parser.add_argument('--skip-extract', action='store_true',
                        help='Skip extraction (use existing params/adduct.mol2)')
    parser.add_argument('--skip-param', action='store_true',
                        help='Skip parameterization (use existing acpype output)')
    parser.add_argument('--charge-method', choices=['bcc', 'gas'], default='bcc',
                        help='Charge method: bcc (AM1-BCC, slower but more accurate) or '
                             'gas (Gasteiger, fast empirical). Default: bcc')
    parser.add_argument('--bcc-timeout', type=int, default=180,
                        help='Timeout in seconds for AM1-BCC parameterization (default: 180). '
                             'If exceeded, falls back to Gasteiger.')
    parser.add_argument('--no-gasteiger-fallback', action='store_true',
                        help='Disable automatic fallback to Gasteiger if AM1-BCC fails/times out')

    args = parser.parse_args()

    # Validate residue name (must be 3 characters for PDB compatibility)
    cyl_resname = args.cyl_resname.upper()
    if len(cyl_resname) != 3:
        print(f"ERROR: --cyl-resname must be exactly 3 characters (got '{args.cyl_resname}')")
        print("       Examples: CYL, CX1, CX2, C32, C86")
        return 1
    if not cyl_resname.isalnum():
        print(f"ERROR: --cyl-resname must be alphanumeric (got '{args.cyl_resname}')")
        return 1

    # Resolve paths
    complex_pdb = Path(args.complex).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Create subdirectories
    params_dir = output_dir / "params"
    md_dir = output_dir / "md_simulation"
    # b05 creates {resname}_residue/ and amber99sb-ildn-{resname}.ff/ inside its output dir
    cyl_output_dir = md_dir  # Residue and force field created directly in md_simulation

    output_dir.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(exist_ok=True)
    md_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Covalent Ligand MD Setup Pipeline")
    print("=" * 60)
    print(f"Complex PDB:    {complex_pdb}")
    print(f"Cys residue:    {args.cys_resid}")
    print(f"Ligand:         {args.lig_resname}:{args.lig_resid}")
    print(f"CYL resname:    {cyl_resname}")
    print(f"Output dir:     {output_dir}")
    print(f"Cap type:       {args.cap_type}")
    print(f"Net charge:     {args.net_charge}")
    print(f"Charge method:  {args.charge_method} (timeout: {args.bcc_timeout}s)")
    print(f"GROMACS:        {args.gmx}")

    # Verify input exists
    if not complex_pdb.exists():
        print(f"\nERROR: Complex PDB not found: {complex_pdb}")
        return 1

    # =========================================================================
    # Step 1: Extract capped adduct fragment
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 1: Extract capped adduct fragment (b04)")
    print("=" * 60)

    adduct_prefix = params_dir / "adduct"
    adduct_pdb = Path(f"{adduct_prefix}.pdb")
    adduct_mol2 = Path(f"{adduct_prefix}.mol2")

    if args.skip_extract and adduct_mol2.exists():
        print(f"  Skipping extraction, using existing: {adduct_mol2}")
    else:
        build_optimized_fragment(
            complex_pdb=str(complex_pdb),
            cys_resname="CYS",
            cys_resid=args.cys_resid,
            lig_resname=args.lig_resname,
            lig_resid=args.lig_resid,
            out_prefix=str(adduct_prefix),
            n_samples=36,
            cap_type=args.cap_type,
            make_mol2=True,
            gen3d=False,
            ph=7.4,
            protonate=True
        )

    if not adduct_mol2.exists():
        print(f"ERROR: Adduct MOL2 not created: {adduct_mol2}")
        return 1

    print(f"  Adduct fragment: {adduct_mol2}")

    # =========================================================================
    # Step 2: Parameterize with acpype
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Parameterize with acpype (GAFF2)")
    print("=" * 60)

    # Prepare mol2 for acpype (merge residues if needed)
    adduct_merged = params_dir / "adduct_merged.mol2"

    # Read mol2 and merge residues - need to fix residue names in ATOM section
    with open(adduct_mol2) as f:
        mol2_lines = f.readlines()

    # Parse mol2 and unify residue names to "MOL" with resid 1
    merged_lines = []
    in_atom_section = False
    for line in mol2_lines:
        if line.startswith('@<TRIPOS>ATOM'):
            in_atom_section = True
            merged_lines.append(line)
        elif line.startswith('@<TRIPOS>'):
            in_atom_section = False
            merged_lines.append(line)
        elif in_atom_section and line.strip():
            # mol2 ATOM format: atom_id name x y z type resid resname charge
            parts = line.split()
            if len(parts) >= 8:
                # Replace resid (column 6, 0-indexed) with 1 and resname (column 7) with MOL
                parts[6] = '1'
                parts[7] = 'MOL'
                # Reconstruct line with proper spacing
                merged_lines.append(f"{parts[0]:>7s} {parts[1]:<4s} {parts[2]:>10s} {parts[3]:>10s} {parts[4]:>10s} {parts[5]:<5s} {parts[6]:>5s} {parts[7]:<4s} {parts[8]:>10s}\n")
            else:
                merged_lines.append(line)
        else:
            merged_lines.append(line)

    with open(adduct_merged, 'w') as f:
        f.writelines(merged_lines)

    print(f"  Merged residues in {adduct_merged.name} (unified to MOL:1)")

    # Check for existing acpype output
    acpype_dir = params_dir / "adduct_merged.acpype"

    # acpype output files (naming depends on charge method):
    #   - adduct_merged_bcc_gaff2.mol2 or adduct_merged_gas_gaff2.mol2
    #   - adduct_merged_GMX.itp (GROMACS include topology with bonded params)
    #   - adduct_merged_GMX.top (GROMACS master topology)
    gmx_mol2 = None
    gmx_itp = None
    charge_method_used = args.charge_method
    
    if acpype_dir.exists():
        for f in acpype_dir.iterdir():
            if f.name.endswith('_gaff2.mol2'):
                gmx_mol2 = f
            elif f.name.endswith('_GMX.itp'):
                gmx_itp = f

    if args.skip_param and acpype_dir.exists() and gmx_mol2 and gmx_mol2.exists():
        print(f"  Skipping parameterization, using existing: {acpype_dir}")
    else:
        acpype_dir, charge_method_used = run_acpype(
            adduct_merged, params_dir, 
            net_charge=args.net_charge,
            charge_method=args.charge_method,
            timeout=args.bcc_timeout,
            gasteiger_fallback=not args.no_gasteiger_fallback
        )

        if acpype_dir and acpype_dir.exists():
            # Find output files - acpype naming convention
            gmx_mol2 = None
            gmx_itp = None
            for f in acpype_dir.iterdir():
                if f.name.endswith('_gaff2.mol2'):  # Works for both bcc and gas
                    gmx_mol2 = f
                elif f.name.endswith('_GMX.itp'):
                    gmx_itp = f
    
    if charge_method_used != args.charge_method:
        print(f"  Note: Used {charge_method_used.upper()} charges (fallback from {args.charge_method.upper()})")

    if not gmx_mol2 or not gmx_mol2.exists():
        print(f"ERROR: acpype did not create parameterized mol2")
        if acpype_dir and acpype_dir.exists():
            print(f"  Available files: {[f.name for f in acpype_dir.iterdir()]}")
        return 1

    if not gmx_itp or not gmx_itp.exists():
        print(f"ERROR: acpype did not create GMX.itp")
        return 1

    print(f"  Parameterized mol2: {gmx_mol2}")
    print(f"  GROMACS ITP:        {gmx_itp}")

    # =========================================================================
    # Step 3: Create CYL residue and modified force field
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 3: Create CYL residue (b05)")
    print("=" * 60)

    b05_script = script_dir / "b05_create_cyl_residue.py"

    run_python_script(
        b05_script,
        f"--adduct-mol2 {gmx_mol2} "
        f"--residue-name {cyl_resname} "
        f"--output-dir {cyl_output_dir} "
        f"--acpype-itp {gmx_itp}"
    )

    # Check outputs - b05 creates nested structure
    # Note: b05 creates directories named after the residue (lowercase)
    cyl_dir_name = f"{cyl_resname.lower()}_residue"
    ff_name = f"amber99sb-ildn-{cyl_resname.lower()}.ff"
    cyl_meta_file = cyl_output_dir / cyl_dir_name / f"{cyl_resname.lower()}_meta.json"
    ff_dir = cyl_output_dir / ff_name

    if not cyl_meta_file.exists():
        print(f"ERROR: {cyl_resname} meta not created: {cyl_meta_file}")
        return 1

    with open(cyl_meta_file) as f:
        cyl_meta = json.load(f)

    print(f"  {cyl_resname} residue created: {cyl_meta_file}")

    # =========================================================================
    # Step 4: Add GAFF2 bonded parameters
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 4: Add GAFF2 bonded parameters")
    print("=" * 60)

    add_gaff2_bonded_params(gmx_itp, ff_dir)

    # =========================================================================
    # Step 5: Assemble covalent complex
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 5: Assemble complex (b06)")
    print("=" * 60)

    complex_cyl = md_dir / f"complex_{cyl_resname.lower()}.pdb"

    assemble_covalent_complex(
        complex_pdb=complex_pdb,
        mol2_path=gmx_mol2,
        cyl_meta=cyl_meta,
        cys_resid=args.cys_resid,
        output_pdb=complex_cyl,
        cyl_resname=cyl_resname
    )

    if not complex_cyl.exists():
        print(f"ERROR: Complex not created: {complex_cyl}")
        return 1

    print(f"  Complex with {cyl_resname}: {complex_cyl}")

    # =========================================================================
    # Step 6: Run pdb2gmx
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 6: Run pdb2gmx")
    print("=" * 60)

    gmx = args.gmx
    gro_file = md_dir / "protein.gro"
    top_file = md_dir / "topol.top"

    # Set GMXLIB to include custom force field directory
    gmxlib_path = str(cyl_output_dir)
    existing_gmxlib = os.environ.get("GMXLIB", "")
    if existing_gmxlib:
        gmxlib_path = f"{gmxlib_path}:{existing_gmxlib}"
    gmx_env = {"GMXLIB": gmxlib_path}
    print(f"  GMXLIB={gmxlib_path}")

    ff_basename = f"amber99sb-ildn-{cyl_resname.lower()}"
    # Don't use -ignh: it would ignore ALL hydrogens and try to rebuild from .hdb,
    # but ligand hydrogens aren't in .hdb. Instead, b06 now correctly omits only
    # the backbone amide H (which has wrong coords from capped mol2), and pdb2gmx
    # will build it correctly. All other hydrogens are kept from the input PDB.
    run_cmd(
        f"{gmx} pdb2gmx -f {complex_cyl} -o {gro_file} -p {top_file} "
        f"-ff {ff_basename} -water tip3p",
        cwd=md_dir,
        env=gmx_env
    )

    if not gro_file.exists():
        print("ERROR: pdb2gmx failed")
        return 1

    # =========================================================================
    # Step 7: Solvate and add ions
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 7: Solvate and add ions")
    print("=" * 60)

    boxed_gro = md_dir / "boxed.gro"
    solvated_gro = md_dir / "solvated.gro"
    ions_gro = md_dir / "solvated_ions.gro"

    # Create box
    run_cmd(
        f"{gmx} editconf -f {gro_file} -o {boxed_gro} -c -d 1.2 -bt cubic",
        cwd=md_dir,
        env=gmx_env
    )

    # Solvate
    run_cmd(
        f"{gmx} solvate -cp {boxed_gro} -cs spc216.gro -o {solvated_gro} -p {top_file}",
        cwd=md_dir,
        env=gmx_env
    )

    # Create ions.mdp
    ions_mdp = md_dir / "ions.mdp"
    with open(ions_mdp, 'w') as f:
        f.write("""; ions.mdp
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000
nstlist     = 1
cutoff-scheme = Verlet
coulombtype = cutoff
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
""")

    # Prepare for genion
    ions_tpr = md_dir / "ions.tpr"
    run_cmd(
        f"{gmx} grompp -f {ions_mdp} -c {solvated_gro} -p {top_file} -o {ions_tpr} -maxwarn 25",
        cwd=md_dir,
        env=gmx_env
    )

    # Add ions to neutralize and add 150 mM NaCl ionic strength
    run_cmd(
        f"echo 'SOL' | {gmx} genion -s {ions_tpr} -o {ions_gro} -p {top_file} "
        f"-pname NA -nname CL -neutral -conc 0.15",
        cwd=md_dir,
        env=gmx_env
    )

    # =========================================================================
    # Step 8: Generate EM files
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 8: Generate EM files")
    print("=" * 60)

    em_mdp = md_dir / "em.mdp"
    with open(em_mdp, 'w') as f:
        f.write("""; em.mdp - Energy minimization
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000
nstlist     = 1
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
nstxout     = 500
nstvout     = 500
nstenergy   = 500
nstlog      = 500
""")

    em_tpr = md_dir / "em.tpr"
    run_cmd(
        f"{gmx} grompp -f {em_mdp} -c {ions_gro} -p {top_file} -o {em_tpr} -maxwarn 25",
        cwd=md_dir,
        env=gmx_env
    )

    # =========================================================================
    # Done
    # =========================================================================
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nResidue name: {cyl_resname}")
    print(f"Output files in {output_dir}:")
    print(f"  - params/                 : Adduct extraction and parameterization")
    print(f"  - md_simulation/          : GROMACS MD setup files")
    print(f"      - {cyl_dir_name}/          : {cyl_resname} residue definition")
    print(f"      - {ff_name}/: Modified force field with GAFF2")
    print(f"      - complex_{cyl_resname.lower()}.pdb       : {cyl_resname} complex structure")
    print(f"      - protein.gro           : GROMACS coordinates")
    print(f"      - topol.top             : Topology")
    print(f"      - solvated_ions.gro     : Solvated system with ions")
    print(f"      - em.tpr                : EM input")
    print(f"\nTo run energy minimization:")
    print(f"  cd {md_dir}")
    print(f"  {gmx} mdrun -deffnm em -v")

    return 0


if __name__ == '__main__':
    sys.exit(main())
