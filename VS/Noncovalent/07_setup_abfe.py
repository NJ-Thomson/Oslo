#!/usr/bin/env python3
"""
ABFE Setup with Boresch Restraints for Noncovalent Ligands

Sets up Absolute Binding Free Energy (ABFE) calculations using alchemical
free energy perturbation with Boresch-style orientational restraints.

ABFE calculates the binding free energy via a thermodynamic cycle:
    dG_bind = dG_complex - dG_solvent + dG_restraint

Two legs are simulated:
1. Complex leg: Ligand decoupled from protein-ligand complex
2. Solvent leg: Ligand decoupled from pure solvent

Boresch restraints anchor the ligand orientation during decoupling in the
complex leg, with an analytical correction for the restraint contribution.

Restraint Selection:
- Uses MDRestraintsGenerator when trajectory is provided (recommended)
- Falls back to geometric selection if no trajectory available
- MDRestraintsGenerator analyzes MD trajectory to find optimal, stable anchor atoms
- Recommended: ~20 ns production MD before restraint selection to ensure
  stable anchor points are identified

Pre-ABFE Equilibration (recommended protocol):
1. 10,000 step energy minimization
2. 1 ns restrained NVT equilibration
3. 1 ns restrained NPT equilibration (Berendsen)
4. 5 ns unrestrained NPT (Parrinello-Rahman)
5. 20 ns NPT production → use this trajectory for MDRestraintsGenerator

FEP Schedule Options (--fep_schedule):
  staged-chained:  Stage-based with chaining (most rigorous, default)
                   restraints → coul → vdw run sequentially
  staged-parallel: Stage-based without chaining (faster)
                   All stages run from input.gro in parallel
  combined:        Combined lambda vectors (legacy approach)

Lambda Schedule (validated defaults):
- Restraint: Dense sampling (0 → 1) with 21 windows
- Coulomb: 11 windows (0 → 1, charges off)
- VdW: 20 windows with dense endpoint sampling (0 → 1, vdW off)

Soft-core Parameters:
- sc-alpha = 0.5, sc-power = 1, sc-sigma = 0.3

Dependencies:
    - GROMACS (gmx)
    - MDRestraintsGenerator (optional, for robust restraint selection)
    - MDAnalysis (required by MDRestraintsGenerator)
    - Output from 05_setup_complex.py or 06_test_binding_stability.py

Installation:
    pip install MDRestraintsGenerator MDAnalysis

Usage:
    python 07_setup_abfe.py \\
        --complex_gro COMPLEX.gro \\
        --topology TOPOL.top \\
        --ligand_itp LIGAND.itp \\
        --output_dir OUTPUT [options]

Examples:
    # First run stability test with 20 ns production
    python 06_test_binding_stability.py \\
        --complex_gro Outputs/NonCovalent/md_simulation/minimized.gro \\
        --topology Outputs/NonCovalent/md_simulation/topol.top \\
        --output_dir Outputs/NonCovalent/stability \\
        --prod_time 20

    # Then setup ABFE with MDRestraintsGenerator (recommended)
    python 07_setup_abfe.py \\
        --complex_gro Outputs/NonCovalent/stability/prod.gro \\
        --topology Outputs/NonCovalent/stability/topol.top \\
        --ligand_itp Outputs/NonCovalent/params/Inhib_42/LIG.itp \\
        --trajectory Outputs/NonCovalent/stability/prod.xtc \\
        --output_dir Outputs/NonCovalent/ABFE/Inhib_42 \\
        --prod_time 5

    # Without trajectory (uses geometric restraint selection - less robust)
    python 07_setup_abfe.py \\
        --complex_gro Outputs/NonCovalent/stability/prod.gro \\
        --topology Outputs/NonCovalent/stability/topol.top \\
        --ligand_itp Outputs/NonCovalent/params/Inhib_42/LIG.itp \\
        --output_dir Outputs/NonCovalent/ABFE/Inhib_42 \\
        --prod_time 5

    # Custom anchor atoms (bypasses automatic selection)
    python 07_setup_abfe.py \\
        --complex_gro complex.gro \\
        --topology topol.top \\
        --ligand_itp LIG.itp \\
        --output_dir ABFE_custom \\
        --protein_anchors 100,105,110 \\
        --ligand_anchors 1,5,10
"""

import argparse
import os
import sys
import subprocess
import shutil
import math
import numpy as np
from pathlib import Path

# Optional MDRestraintsGenerator import
try:
    from MDRestraintsGenerator import search, restraints
    import MDAnalysis as mda
    HAS_MDRESTRAINTS = True
except ImportError:
    HAS_MDRESTRAINTS = False


def find_gmx():
    """Find GROMACS executable."""
    for cmd in ['gmx', 'gmx_mpi']:
        try:
            result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                return cmd
        except FileNotFoundError:
            continue
    return None


def run_command(cmd, description, cwd=None, input_text=None, verbose=True):
    """Execute a shell command and handle errors."""
    if verbose:
        print(f"  {description}...")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
        input=input_text,
        shell=isinstance(cmd, str)
    )

    if result.returncode != 0:
        if verbose:
            print(f"  WARNING: {description} returned non-zero")
        return False, result

    if verbose:
        print(f"  Done")
    return True, result


def read_gro_coordinates(gro_path):
    """Read coordinates from GRO file."""
    coords = {}
    with open(gro_path) as f:
        lines = f.readlines()

    n_atoms = int(lines[1].strip())
    for i, line in enumerate(lines[2:2+n_atoms]):
        resnum = int(line[0:5])
        resname = line[5:10].strip()
        atomname = line[10:15].strip()
        atomnum = int(line[15:20])
        x = float(line[20:28]) * 10  # nm to Angstrom
        y = float(line[28:36]) * 10
        z = float(line[36:44]) * 10
        coords[atomnum] = {
            'resnum': resnum,
            'resname': resname,
            'atomname': atomname,
            'x': x, 'y': y, 'z': z
        }

    return coords


def find_anchor_atoms(gro_path, ligand_resname='LIG'):
    """
    Automatically select anchor atoms for Boresch restraints.

    Selects:
    - 3 protein backbone atoms (CA) near the ligand
    - 3 ligand heavy atoms that are well-distributed

    Returns:
        protein_atoms: list of 3 atom indices
        ligand_atoms: list of 3 atom indices
    """
    coords = read_gro_coordinates(gro_path)

    # Separate protein and ligand atoms
    protein_atoms = []
    ligand_atoms = []

    for atomnum, atom in coords.items():
        if atom['resname'] == ligand_resname:
            # Only heavy atoms for ligand
            if not atom['atomname'].startswith('H'):
                ligand_atoms.append(atomnum)
        elif atom['atomname'] == 'CA':
            # CA atoms for protein
            protein_atoms.append(atomnum)

    if len(ligand_atoms) < 3:
        print(f"  WARNING: Not enough ligand heavy atoms found")
        return None, None

    if len(protein_atoms) < 3:
        print(f"  WARNING: Not enough protein CA atoms found")
        return None, None

    # Calculate ligand center
    lig_center = np.array([0.0, 0.0, 0.0])
    for atomnum in ligand_atoms:
        atom = coords[atomnum]
        lig_center += np.array([atom['x'], atom['y'], atom['z']])
    lig_center /= len(ligand_atoms)

    # Find closest protein CA atoms to ligand
    protein_distances = []
    for atomnum in protein_atoms:
        atom = coords[atomnum]
        pos = np.array([atom['x'], atom['y'], atom['z']])
        dist = np.linalg.norm(pos - lig_center)
        protein_distances.append((atomnum, dist))

    protein_distances.sort(key=lambda x: x[1])

    # Select 3 protein atoms that are reasonably spaced
    selected_protein = [protein_distances[0][0]]
    for atomnum, dist in protein_distances[1:]:
        atom = coords[atomnum]
        pos = np.array([atom['x'], atom['y'], atom['z']])

        # Check distance from already selected atoms
        min_sep = min(
            np.linalg.norm(pos - np.array([coords[a]['x'], coords[a]['y'], coords[a]['z']]))
            for a in selected_protein
        )

        if min_sep > 3.0:  # At least 3 A apart
            selected_protein.append(atomnum)
            if len(selected_protein) == 3:
                break

    if len(selected_protein) < 3:
        selected_protein = [d[0] for d in protein_distances[:3]]

    # Select 3 ligand atoms that are well-distributed
    # Use first, middle, and last heavy atom indices as approximation
    n_lig = len(ligand_atoms)
    selected_ligand = [
        ligand_atoms[0],
        ligand_atoms[n_lig // 2],
        ligand_atoms[-1]
    ]

    return selected_protein, selected_ligand


def find_anchor_atoms_mdrestraints(topology_path, trajectory_path, ligand_resname='LIG',
                                    output_dir=None, temperature=298.15):
    """
    Use MDRestraintsGenerator to find optimal Boresch anchor atoms.

    This analyzes MD trajectory to find stable anchor points with minimal
    fluctuation, which is crucial for accurate restraint corrections.

    Args:
        topology_path: Path to TPR file (contains topology with angles/bonds) or GRO file
        trajectory_path: Path to trajectory (XTC/TRR)
        ligand_resname: Residue name of ligand
        output_dir: Directory for output plots/files
        temperature: Simulation temperature in K

    Returns:
        protein_atoms: list of 3 atom indices (1-indexed for GROMACS)
        ligand_atoms: list of 3 atom indices (1-indexed for GROMACS)
        boresch_params: dict with restraint parameters from trajectory analysis
    """
    if not HAS_MDRESTRAINTS:
        print("  ERROR: MDRestraintsGenerator not installed")
        print("  Install with: pip install MDRestraintsGenerator MDAnalysis")
        return None, None, None

    print(f"  Loading trajectory: {trajectory_path}")
    print(f"  Using topology: {topology_path}")
    u = mda.Universe(str(topology_path), str(trajectory_path))

    # Select ligand atoms
    ligand_sel = u.select_atoms(f"resname {ligand_resname}")
    if len(ligand_sel) == 0:
        print(f"  ERROR: No atoms found with resname {ligand_resname}")
        return None, None, None

    print(f"  Found {len(ligand_sel)} ligand atoms")

    # Find stable ligand anchor atoms using MDRestraintsGenerator
    print("  Analyzing ligand for stable anchor points...")
    ligand_atoms_analysis = search.find_ligand_atoms(u, f"resname {ligand_resname}")

    # Find protein anchor atoms near the ligand
    print("  Finding protein anchor atoms near binding site...")

    # Use the first set of ligand atoms from analysis
    l_atoms = ligand_atoms_analysis[0]
    print(f"  Using ligand anchor atoms: {l_atoms}")

    # Use FindHostAtoms to identify suitable protein anchors
    # Pass first ligand atom index for proximity search
    host_finder = search.FindHostAtoms(u, l_atoms[0], p_selection="protein and name CA")
    host_finder.run()

    # Now find optimal Boresch restraint using trajectory
    print("  Analyzing trajectory for optimal Boresch restraints...")
    print("  (This may take a while for long trajectories)")

    # Create atom sets: each is (ligand_atoms, protein_atoms) tuple
    atom_set = [(l_atoms, p) for p in host_finder.host_atoms]

    boresch_finder = restraints.FindBoreschRestraint(u, atom_set)
    boresch_finder.run()

    # Get the best restraint
    best_restraint = boresch_finder.restraint

    # Save diagnostic plots if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Plot restraint distributions
            print(f"  Saving diagnostic plots to {output_dir}")
            best_restraint.plot(str(output_dir / 'boresch_restraint_analysis.png'))
        except Exception as e:
            print(f"  Warning: Could not save plots: {e}")
            print(f"           (This is usually a numpy/matplotlib version compatibility issue with MDRestraintsGenerator)")
            print(f"           The restraint parameters are still valid - plotting is optional.")

    # Extract atom indices from restraint objects
    # Bond contains (l_atoms[0], p_atoms[0])
    # Angles[0] contains (l_atoms[1], l_atoms[0], p_atoms[0])
    # Angles[1] contains (l_atoms[0], p_atoms[0], p_atoms[1])
    # Dihedrals[0] contains (l_atoms[2], l_atoms[1], l_atoms[0], p_atoms[0])
    # Dihedrals[2] contains (l_atoms[0], p_atoms[0], p_atoms[1], p_atoms[2])
    ligand_atoms_0idx = [
        best_restraint.bond.atomgroup.atoms[0].ix,        # L1
        best_restraint.angles[0].atomgroup.atoms[0].ix,   # L2
        best_restraint.dihedrals[0].atomgroup.atoms[0].ix # L3
    ]
    protein_atoms_0idx = [
        best_restraint.bond.atomgroup.atoms[1].ix,        # P1
        best_restraint.angles[1].atomgroup.atoms[2].ix,   # P2
        best_restraint.dihedrals[2].atomgroup.atoms[3].ix # P3
    ]

    # Convert to 1-indexed for GROMACS
    protein_atoms = [a + 1 for a in protein_atoms_0idx]
    ligand_atoms = [a + 1 for a in ligand_atoms_0idx]

    # Extract equilibrium values from trajectory analysis
    # These are averaged over the trajectory for stability
    boresch_params = {
        'r0': best_restraint.bond.mean / 10.0,  # A to nm
        'theta_A0': best_restraint.angles[0].mean,  # degrees
        'theta_B0': best_restraint.angles[1].mean,  # degrees
        'phi_A0': best_restraint.dihedrals[0].mean,  # degrees
        'phi_B0': best_restraint.dihedrals[1].mean,  # degrees
        'phi_C0': best_restraint.dihedrals[2].mean,  # degrees
        # Standard force constants (can be adjusted)
        'kr': 4184.0,  # kJ/mol/nm^2 (10 kcal/mol/A^2)
        'ktheta': 41.84,  # kJ/mol/rad^2
        'kphi': 41.84,  # kJ/mol/rad^2
        'protein_atoms': protein_atoms,
        'ligand_atoms': ligand_atoms,
        # Store fluctuations for diagnostics (attribute is 'stdev' not 'std')
        'r_std': best_restraint.bond.stdev / 10.0,
        'theta_A_std': best_restraint.angles[0].stdev,
        'theta_B_std': best_restraint.angles[1].stdev,
        'phi_A_std': best_restraint.dihedrals[0].stdev,
        'phi_B_std': best_restraint.dihedrals[1].stdev,
        'phi_C_std': best_restraint.dihedrals[2].stdev,
    }

    # Calculate analytical correction using standard_state method
    try:
        dG_restr = best_restraint.standard_state(temperature=temperature)
        boresch_params['dG_restr_mdrestraints'] = dG_restr
        print(f"  MDRestraintsGenerator dG_restraint: {dG_restr:.2f} kJ/mol")
    except Exception as e:
        print(f"  Warning: Could not compute standard state correction: {e}")

    print(f"\n  Optimal anchor atoms found:")
    print(f"    Protein (P1-P2-P3): {protein_atoms[0]}-{protein_atoms[1]}-{protein_atoms[2]}")
    print(f"    Ligand (L1-L2-L3):  {ligand_atoms[0]}-{ligand_atoms[1]}-{ligand_atoms[2]}")

    return protein_atoms, ligand_atoms, boresch_params


def calculate_boresch_parameters(gro_path, protein_atoms, ligand_atoms):
    """
    Calculate Boresch restraint parameters from equilibrium structure.

    Boresch restraints define:
    - r: distance P1-L1
    - theta_A: angle P2-P1-L1
    - theta_B: angle P1-L1-L2
    - phi_A: dihedral P3-P2-P1-L1
    - phi_B: dihedral P2-P1-L1-L2
    - phi_C: dihedral P1-L1-L2-L3

    Returns:
        dict with restraint parameters
    """
    coords = read_gro_coordinates(gro_path)

    # Get positions
    P1 = np.array([coords[protein_atoms[0]]['x'],
                   coords[protein_atoms[0]]['y'],
                   coords[protein_atoms[0]]['z']])
    P2 = np.array([coords[protein_atoms[1]]['x'],
                   coords[protein_atoms[1]]['y'],
                   coords[protein_atoms[1]]['z']])
    P3 = np.array([coords[protein_atoms[2]]['x'],
                   coords[protein_atoms[2]]['y'],
                   coords[protein_atoms[2]]['z']])

    L1 = np.array([coords[ligand_atoms[0]]['x'],
                   coords[ligand_atoms[0]]['y'],
                   coords[ligand_atoms[0]]['z']])
    L2 = np.array([coords[ligand_atoms[1]]['x'],
                   coords[ligand_atoms[1]]['y'],
                   coords[ligand_atoms[1]]['z']])
    L3 = np.array([coords[ligand_atoms[2]]['x'],
                   coords[ligand_atoms[2]]['y'],
                   coords[ligand_atoms[2]]['z']])

    # Calculate r (P1-L1 distance)
    r = np.linalg.norm(P1 - L1)

    # Calculate angles
    def angle(a, b, c):
        """Angle ABC in degrees."""
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.degrees(np.arccos(cos_angle))

    theta_A = angle(P2, P1, L1)
    theta_B = angle(P1, L1, L2)

    # Calculate dihedrals
    def dihedral(a, b, c, d):
        """Dihedral ABCD in degrees."""
        b1 = b - a
        b2 = c - b
        b3 = d - c

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        m1 = np.cross(n1, b2 / np.linalg.norm(b2))

        x = np.dot(n1, n2)
        y = np.dot(m1, n2)

        return np.degrees(np.arctan2(y, x))

    phi_A = dihedral(P3, P2, P1, L1)
    phi_B = dihedral(P2, P1, L1, L2)
    phi_C = dihedral(P1, L1, L2, L3)

    return {
        'r0': r / 10.0,  # Convert to nm
        'theta_A0': theta_A,
        'theta_B0': theta_B,
        'phi_A0': phi_A,
        'phi_B0': phi_B,
        'phi_C0': phi_C,
        'kr': 4184.0,  # kJ/mol/nm^2 (10 kcal/mol/A^2)
        'ktheta': 41.84,  # kJ/mol/rad^2
        'kphi': 41.84,  # kJ/mol/rad^2
        'protein_atoms': protein_atoms,
        'ligand_atoms': ligand_atoms
    }


def calculate_restraint_correction(params, temperature=298.15):
    """
    Calculate analytical correction for Boresch restraints.

    dG_restr = -RT * ln(V0 * 8*pi^2 / (r0^2 * sin(theta_A) * sin(theta_B)) *
                        sqrt(kr*ktheta^2*kphi^3 / (2*pi*RT)^6))

    Returns dG in kJ/mol
    """
    R = 8.314e-3  # kJ/mol/K
    V0 = 1.66054  # Standard volume in nm^3 (1 M = 1/1660.54 nm^3)

    r0 = params['r0']
    theta_A = math.radians(params['theta_A0'])
    theta_B = math.radians(params['theta_B0'])

    kr = params['kr']
    ktheta = params['ktheta']
    kphi = params['kphi']

    RT = R * temperature

    # Force constant prefactor
    k_factor = math.sqrt(kr * ktheta**2 * kphi**3 / (2 * math.pi * RT)**6)

    # Geometric factor
    geom_factor = V0 * 8 * math.pi**2 / (r0**2 * math.sin(theta_A) * math.sin(theta_B))

    dG_restr = -RT * math.log(geom_factor * k_factor)

    return dG_restr


def write_restraint_itp(output_path, params):
    """
    Write Boresch restraint topology.

    Uses intermolecular_interactions section for GROMACS.
    """
    P1, P2, P3 = params['protein_atoms']
    L1, L2, L3 = params['ligand_atoms']

    content = f"""; Boresch orientational restraints for ABFE
; Atoms: P3({P3})-P2({P2})-P1({P1})---L1({L1})-L2({L2})-L3({L3})
;
; Equilibrium values from input structure:
;   r0 = {params['r0']:.4f} nm
;   theta_A0 = {params['theta_A0']:.2f} deg
;   theta_B0 = {params['theta_B0']:.2f} deg
;   phi_A0 = {params['phi_A0']:.2f} deg
;   phi_B0 = {params['phi_B0']:.2f} deg
;   phi_C0 = {params['phi_C0']:.2f} deg

[ intermolecular_interactions ]

[ bonds ]
; P1   L1   type   r0 (nm)    kr (kJ/mol/nm^2)   r0_B   kr_B
{P1}  {L1}   6      {params['r0']:.4f}    0.0               {params['r0']:.4f}  {params['kr']:.1f}

[ angles ]
; P2   P1   L1   type  theta0 (deg)  ktheta (kJ/mol/rad^2)  theta0_B  ktheta_B
{P2}  {P1}  {L1}   1     {params['theta_A0']:.2f}        0.0                      {params['theta_A0']:.2f}   {params['ktheta']:.2f}
; P1   L1   L2
{P1}  {L1}  {L2}   1     {params['theta_B0']:.2f}        0.0                      {params['theta_B0']:.2f}   {params['ktheta']:.2f}

[ dihedrals ]
; P3   P2   P1   L1   type  phi0 (deg)  kphi (kJ/mol/rad^2)  mult  phi0_B  kphi_B  mult_B
{P3}  {P2}  {P1}  {L1}   2     {params['phi_A0']:.2f}       0.0                    {params['phi_A0']:.2f}  {params['kphi']:.2f}
; P2   P1   L1   L2
{P2}  {P1}  {L1}  {L2}   2     {params['phi_B0']:.2f}       0.0                    {params['phi_B0']:.2f}  {params['kphi']:.2f}
; P1   L1   L2   L3
{P1}  {L1}  {L2}  {L3}   2     {params['phi_C0']:.2f}       0.0                    {params['phi_C0']:.2f}  {params['kphi']:.2f}
"""

    with open(output_path, 'w') as f:
        f.write(content)

    return output_path


def generate_stage_lambdas():
    """
    Generate validated lambda schedules as separate stages.

    Each transformation stage (restraints, coulomb, vdw) is run as separate
    simulations with stage-specific coupling settings.

    Returns:
        dict with keys: 'restraints', 'coul', 'vdw'
        Each value is a list of lambda values for that stage.
    """
    # Restraint lambdas (0=off, 1=on) - 12 windows
    # Only used for complex leg
    restraint_lambdas = [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 0.95, 1.0
    ]

    # Coulomb lambdas (0=on, 1=off in couple-lambda sense) - 11 windows
    coul_lambdas = [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1.0
    ]

    # VdW lambdas (0=on, 1=off) - 21 windows
    # Dense near lambda=1 (decoupled) where soft-core is critical
    vdw_lambdas = [
        0.0, 0.05, 0.1, 0.15, 0.2, 0.25,
        0.3, 0.35, 0.4, 0.45, 0.5,
        0.55, 0.6, 0.65, 0.7, 0.75,
        0.8, 0.85, 0.9, 0.95, 1.0
    ]

    return {
        'restraints': restraint_lambdas,
        'coul': coul_lambdas,
        'vdw': vdw_lambdas
    }


def generate_combined_lambdas():
    """
    Generate lambda schedule in combined format (legacy approach).

    This is kept for backwards compatibility but the separate stages approach
    (generate_stage_lambdas) is now preferred.
    """
    stages = generate_stage_lambdas()

    # Complex leg: restraint on, then decouple
    complex_lambdas = []

    # Phase 1: Turn on restraints (coul=1, vdw=1)
    for r in stages['restraints']:
        complex_lambdas.append((r, 1.0, 1.0))

    # Phase 2: Turn off electrostatics (restr=1, vdw=1)
    for c in stages['coul'][1:]:  # Skip first
        complex_lambdas.append((1.0, 1.0 - c, 1.0))

    # Phase 3: Turn off vdW (restr=1, coul=0)
    for v in stages['vdw'][1:]:  # Skip first
        complex_lambdas.append((1.0, 0.0, 1.0 - v))

    # Solvent leg: just decouple (no restraint)
    solvent_lambdas = []

    # Phase 1: Turn off electrostatics
    for c in stages['coul']:
        solvent_lambdas.append((1.0 - c, 1.0))

    # Phase 2: Turn off vdW
    for v in stages['vdw'][1:]:
        solvent_lambdas.append((0.0, 1.0 - v))

    return complex_lambdas, solvent_lambdas


def generate_lambda_schedule(n_windows_restr=5, n_windows_coul=10, n_windows_vdw=15,
                              use_validated_schedule=True):
    """
    Generate lambda schedule for ABFE (combined format for legacy mode).

    Args:
        n_windows_restr: Number of restraint windows (ignored if use_validated_schedule=True)
        n_windows_coul: Number of coulomb windows (ignored if use_validated_schedule=True)
        n_windows_vdw: Number of vdW windows (ignored if use_validated_schedule=True)
        use_validated_schedule: Use validated schedule (recommended)

    Complex leg: restraint -> coul -> vdw
    Solvent leg: coul -> vdw (no restraint)

    Returns:
        complex_lambdas: list of (restr, coul, vdw) tuples
        solvent_lambdas: list of (coul, vdw) tuples
    """
    if use_validated_schedule:
        return generate_combined_lambdas()

    # Original schedule (for custom window counts)
    # Restraint lambda (0 = off, 1 = on)
    restr_lambdas = np.linspace(0, 1, n_windows_restr)

    # Coulomb lambda (1 = on, 0 = off)
    coul_lambdas = np.linspace(1, 0, n_windows_coul)

    # VdW lambda (1 = on, 0 = off) - use more windows at endpoints
    vdw_lambdas = np.concatenate([
        np.linspace(1.0, 0.5, n_windows_vdw // 2),
        np.linspace(0.5, 0.0, n_windows_vdw - n_windows_vdw // 2 + 1)[1:]
    ])

    # Complex leg: restraint on, then decouple
    complex_lambdas = []

    # Phase 1: Turn on restraints (coul=1, vdw=1)
    for r in restr_lambdas:
        complex_lambdas.append((r, 1.0, 1.0))

    # Phase 2: Turn off electrostatics (restr=1, vdw=1)
    for c in coul_lambdas[1:]:  # Skip first (already at 1)
        complex_lambdas.append((1.0, c, 1.0))

    # Phase 3: Turn off vdW (restr=1, coul=0)
    for v in vdw_lambdas[1:]:  # Skip first (already at 1)
        complex_lambdas.append((1.0, 0.0, v))

    # Solvent leg: just decouple (no restraint)
    solvent_lambdas = []

    # Phase 1: Turn off electrostatics
    for c in coul_lambdas:
        solvent_lambdas.append((c, 1.0))

    # Phase 2: Turn off vdW
    for v in vdw_lambdas[1:]:
        solvent_lambdas.append((0.0, v))

    return complex_lambdas, solvent_lambdas


# =============================================================================
# Stage-Based MDP Generation
# =============================================================================

def get_stage_coupling(stage):
    """
    Get the couple-lambda0/1 settings for each stage.

    Returns:
        tuple: (couple_lambda0, couple_lambda1)
    """
    if stage == 'restraints':
        # Restraints only - no change in intermolecular coupling
        return 'vdw-q', 'vdw-q'
    elif stage == 'coul':
        # Turn off electrostatics, keep VdW
        return 'vdw-q', 'vdw'
    elif stage == 'vdw':
        # Turn off VdW (electrostatics already off)
        return 'vdw', 'none'
    else:
        raise ValueError(f"Unknown stage: {stage}")


def get_lambda_vectors(stage, lambdas):
    """
    Generate lambda vector strings for MDP file.

    Args:
        stage: 'restraints', 'coul', or 'vdw'
        lambdas: List of lambda values for this stage

    Returns:
        str: Lambda vector lines for MDP file
    """
    n = len(lambdas)
    lambda_str = ' '.join(f'{l:.2f}' for l in lambdas)
    bonded_ones = ' '.join(['1.0'] * n)

    if stage == 'restraints':
        # Only bonded-lambdas varies
        return f"bonded-lambdas           = {lambda_str}"
    elif stage == 'coul':
        # coul-lambdas varies, bonded stays at 1.0
        return f"bonded-lambdas           = {bonded_ones}\ncoul-lambdas             = {lambda_str}"
    elif stage == 'vdw':
        # vdw-lambdas varies, bonded stays at 1.0
        return f"bonded-lambdas           = {bonded_ones}\nvdw-lambdas              = {lambda_str}"
    else:
        raise ValueError(f"Unknown stage: {stage}")


def write_stage_mdp(output_path, template_name, stage, state, lambdas,
                    ligand_moltype='LIG', temperature=298.15, nsteps=None):
    """
    Write MDP file for a specific stage using templates.

    Args:
        output_path: Output file path
        template_name: Template file name (em.mdp, nvt.mdp, npt_posres.mdp, npt.mdp, prod.mdp)
        stage: 'restraints', 'coul', or 'vdw'
        state: Lambda state index
        lambdas: List of lambda values for this stage
        ligand_moltype: Molecule type name for ligand
        temperature: Reference temperature
        nsteps: Number of steps (for prod.mdp)
    """
    script_dir = Path(__file__).parent
    template_path = script_dir / 'templates' / 'mdp' / template_name

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, 'r') as f:
        content = f.read()

    # Get stage-specific settings
    couple_lambda0, couple_lambda1 = get_stage_coupling(stage)
    lambda_vectors = get_lambda_vectors(stage, lambdas)

    # Replace placeholders
    content = content.replace('{{STAGE}}', stage)
    content = content.replace('{{STATE}}', str(state))
    content = content.replace('{{LIGAND_MOLTYPE}}', ligand_moltype)
    content = content.replace('{{COUPLE_LAMBDA0}}', couple_lambda0)
    content = content.replace('{{COUPLE_LAMBDA1}}', couple_lambda1)
    content = content.replace('{{LAMBDA_VECTORS}}', lambda_vectors)
    content = content.replace('{{TEMPERATURE}}', str(temperature))

    if nsteps is not None:
        content = content.replace('{{NSTEPS}}', str(nsteps))

    with open(output_path, 'w') as f:
        f.write(content)

    return output_path


def setup_stage_directory(base_dir, stage, state, lambdas, ligand_moltype='LIG',
                          temperature=298.15, prod_nsteps=2500000):
    """
    Set up a single lambda window directory for a stage.

    Uses branched workflow:
    - Window 0: NPT equilibration (500 ps) -> production
    - Windows 1-N: Branch from window 0, short equilibration (50 ps) -> production

    Args:
        base_dir: Base directory for this leg (e.g., complex/fep/simulation)
        stage: 'restraints', 'coul', or 'vdw'
        state: Lambda state index
        lambdas: Full list of lambda values for this stage
        ligand_moltype: Molecule type name
        temperature: Reference temperature
        prod_nsteps: Production steps (default 2.5M = 10 ns at 4 fs)

    Returns:
        Path to created directory
    """
    window_dir = base_dir / f'{stage}.{state}'
    window_dir.mkdir(parents=True, exist_ok=True)

    # Determine phases based on window index
    # Window 0: Full equilibration sequence (EM -> NVT -> NPT_posres -> NPT -> Prod)
    # Windows 1-N: Branch from window 0, short equilibration only
    if state == 0:
        phases = ['em', 'nvt', 'npt_posres', 'npt', 'prod']
    else:
        phases = ['equil', 'prod']

    for phase in phases:
        phase_dir = window_dir / phase
        phase_dir.mkdir(exist_ok=True)

        # Determine template and nsteps
        if phase == 'prod':
            template = 'prod.mdp'
            nsteps = prod_nsteps
        elif phase == 'equil':
            template = 'equil.mdp'
            nsteps = None  # Uses default 50 ps from template
        elif phase == 'npt_posres':
            template = 'npt_posres.mdp'
            nsteps = None  # Uses default 100 ps from template
        else:
            template = f'{phase}.mdp'
            nsteps = None

        # Write MDP file
        write_stage_mdp(
            phase_dir / f'{phase}.mdp',
            template,
            stage,
            state,
            lambdas,
            ligand_moltype=ligand_moltype,
            temperature=temperature,
            nsteps=nsteps
        )

    return window_dir


# =============================================================================
# Legacy Combined MDP Generation (for backwards compatibility)
# =============================================================================

def write_fep_mdp(output_path, lambda_state, lambdas, leg='complex',
                  nsteps=2500000, dt=0.002, ref_temp=298.15, with_restraint=True,
                  ligand_moltype='LIG'):
    """
    Write MDP file for FEP simulation (legacy combined approach).

    Args:
        lambda_state: Index of current lambda window
        lambdas: Full list of lambda values
        leg: 'complex' or 'solvent'
        with_restraint: Include restraint lambda (complex leg only)
        ligand_moltype: Molecule type name for ligand (from topology [ moleculetype ])
    """
    ns = nsteps * dt / 1000

    # Build lambda vectors
    if leg == 'complex':
        restr_vec = ' '.join(f'{l[0]:.4f}' for l in lambdas)
        coul_vec = ' '.join(f'{l[1]:.4f}' for l in lambdas)
        vdw_vec = ' '.join(f'{l[2]:.4f}' for l in lambdas)

        lambda_section = f"""
; Lambda vectors (restraint, coul, vdw)
restraint_lambdas   = {restr_vec}
coul_lambdas        = {coul_vec}
vdw_lambdas         = {vdw_vec}
"""
    else:
        coul_vec = ' '.join(f'{l[0]:.4f}' for l in lambdas)
        vdw_vec = ' '.join(f'{l[1]:.4f}' for l in lambdas)

        lambda_section = f"""
; Lambda vectors (coul, vdw)
coul_lambdas        = {coul_vec}
vdw_lambdas         = {vdw_vec}
"""

    content = f"""; Production FEP - {leg} leg, lambda state {lambda_state}
integrator          = md
dt                  = {dt}
nsteps              = {nsteps}  ; {ns:.1f} ns

nstlog              = 5000
nstenergy           = 5000
nstxout-compressed  = 5000

tcoupl              = V-rescale
tc-grps             = System
tau_t               = 0.1
ref_t               = {ref_temp}

pcoupl              = C-rescale
pcoupltype          = isotropic
tau_p               = 2.0
ref_p               = 1.0
compressibility     = 4.5e-5
refcoord_scaling    = com

gen_vel             = no
continuation        = yes

constraints         = h-bonds
constraint_algorithm = LINCS

; Neighbor searching
cutoff-scheme       = Verlet
nstlist             = 100
pbc                 = xyz

; Electrostatics
coulombtype         = PME
rcoulomb            = 1.0
fourierspacing      = 0.12

; VdW
vdwtype             = Cut-off
rvdw                = 1.0
DispCorr            = EnerPres

; Free energy settings
free_energy         = yes
init_lambda_state   = {lambda_state}
{lambda_section}

; Molecule coupling - THIS IS ESSENTIAL
; couple-moltype specifies which molecule to decouple
; couple-lambda0 = vdw-q means fully coupled (VdW + Coulomb) at lambda=0
; couple-lambda1 = none means fully decoupled at lambda=1
; couple-intramol = no means don't scale intramolecular interactions
couple-moltype      = {ligand_moltype}
couple-lambda0      = vdw-q
couple-lambda1      = none
couple-intramol     = no

; Soft-core parameters
; These settings prevent singularities when vdW interactions are turned off
; sc-alpha = 0.5 provides smooth transitions at lambda endpoints
; sc-power = 1 (linear soft-core)
; sc-sigma = 0.3 nm is the soft-core interaction radius
; sc-coul = yes applies soft-core to Coulomb as well (more stable)
sc-alpha            = 0.5
sc-power            = 1
sc-sigma            = 0.3
sc-coul             = yes

; FEP output
; nstdhdl = 100 outputs dH/dlambda every 0.2 ps (with dt=0.002)
; calc-lambda-neighbors = -1 calculates dH/dlambda for all lambda states (for MBAR)
nstdhdl             = 100
calc-lambda-neighbors = -1
"""

    with open(output_path, 'w') as f:
        f.write(content)

    return output_path


def write_equil_mdp(output_path, stage, lambda_state, lambdas, leg='complex',
                    nsteps=50000, dt=0.002, ref_temp=298.15, ligand_moltype='LIG'):
    """Write equilibration MDP for FEP window."""
    ns = nsteps * dt / 1000

    if leg == 'complex':
        restr_vec = ' '.join(f'{l[0]:.4f}' for l in lambdas)
        coul_vec = ' '.join(f'{l[1]:.4f}' for l in lambdas)
        vdw_vec = ' '.join(f'{l[2]:.4f}' for l in lambdas)
        lambda_section = f"""restraint_lambdas   = {restr_vec}
coul_lambdas        = {coul_vec}
vdw_lambdas         = {vdw_vec}"""
    else:
        coul_vec = ' '.join(f'{l[0]:.4f}' for l in lambdas)
        vdw_vec = ' '.join(f'{l[1]:.4f}' for l in lambdas)
        lambda_section = f"""coul_lambdas        = {coul_vec}
vdw_lambdas         = {vdw_vec}"""

    if stage == 'nvt':
        pcoupl_section = "pcoupl = no"
        gen_vel = f"gen_vel = yes\ngen_temp = {ref_temp}\ngen_seed = -1"
        continuation = ""
        posres = "define = -DPOSRES"
    else:  # npt
        pcoupl_section = """pcoupl = C-rescale
pcoupltype = isotropic
tau_p = 2.0
ref_p = 1.0
compressibility = 4.5e-5
refcoord_scaling = com"""
        gen_vel = "gen_vel = no"
        continuation = "continuation = yes"
        posres = "define = -DPOSRES"

    content = f"""; {stage.upper()} equilibration - {leg} leg, lambda {lambda_state}
integrator          = md
dt                  = {dt}
nsteps              = {nsteps}  ; {ns:.3f} ns

nstlog              = 1000
nstenergy           = 1000
nstxout-compressed  = 1000

tcoupl              = V-rescale
tc-grps             = System
tau_t               = 0.1
ref_t               = {ref_temp}

{pcoupl_section}

{gen_vel}
{continuation}

constraints         = h-bonds
constraint_algorithm = LINCS

{posres}

cutoff-scheme       = Verlet
nstlist             = 100
pbc                 = xyz

coulombtype         = PME
rcoulomb            = 1.0
fourierspacing      = 0.12

vdwtype             = Cut-off
rvdw                = 1.0
DispCorr            = EnerPres

; Free energy
free_energy         = yes
init_lambda_state   = {lambda_state}
{lambda_section}

; Molecule coupling
couple-moltype      = {ligand_moltype}
couple-lambda0      = vdw-q
couple-lambda1      = none
couple-intramol     = no

sc-alpha            = 0.5
sc-power            = 1
sc-sigma            = 0.3
sc-coul             = yes

nstdhdl             = 100
calc-lambda-neighbors = -1
"""

    with open(output_path, 'w') as f:
        f.write(content)

    return output_path


def setup_solvent_leg(ligand_gro, ligand_itp, output_dir, gmx, box_size=3.0):
    """
    Set up the solvent leg (ligand alone in water).
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert input paths to absolute
    ligand_gro = Path(ligand_gro).resolve()
    ligand_itp = Path(ligand_itp).resolve()

    print("\n  Setting up solvent leg...")

    # Copy ligand files
    shutil.copy(ligand_itp, output_dir / 'LIG.itp')

    # Check for atomtypes file (often required for GAFF/custom ligands)
    ligand_atomtypes = ligand_itp.parent / 'LIG_atomtypes.itp'
    has_atomtypes = ligand_atomtypes.exists()
    if has_atomtypes:
        shutil.copy(ligand_atomtypes, output_dir / 'LIG_atomtypes.itp')

    # Create topology for ligand in water
    atomtypes_include = '#include "LIG_atomtypes.itp"\n\n' if has_atomtypes else ''
    top_content = f"""; Ligand in solvent topology
#include "amber99sb-ildn.ff/forcefield.itp"

; Include ligand atomtypes (if present)
{atomtypes_include}; Include ligand parameters
#include "LIG.itp"

; Include water topology
#include "amber99sb-ildn.ff/tip3p.itp"

; Include ions
#include "amber99sb-ildn.ff/ions.itp"

[ system ]
Ligand in water

[ molecules ]
LIG     1
"""

    with open(output_dir / 'topol.top', 'w') as f:
        f.write(top_content)

    # Create box around ligand (use absolute paths)
    boxed_gro = output_dir / 'ligand_box.gro'
    cmd = [gmx, 'editconf', '-f', str(ligand_gro), '-o', str(boxed_gro),
           '-bt', 'dodecahedron', '-d', str(box_size), '-c']
    success, _ = run_command(cmd, "Creating box", cwd=str(output_dir))
    if not success:
        return False

    # Solvate (use absolute paths)
    solv_gro = output_dir / 'ligand_solv.gro'
    top_file = output_dir / 'topol.top'
    cmd = [gmx, 'solvate', '-cp', str(boxed_gro), '-cs', 'spc216',
           '-o', str(solv_gro), '-p', str(top_file)]
    success, _ = run_command(cmd, "Solvating", cwd=str(output_dir))
    if not success:
        return False

    # Add ions
    ions_mdp = output_dir / 'ions.mdp'
    with open(ions_mdp, 'w') as f:
        f.write("integrator = steep\nnsteps = 0\n")

    tpr = output_dir / 'ions.tpr'
    mdout = output_dir / 'mdout_ions.mdp'
    cmd = [gmx, 'grompp', '-f', str(ions_mdp), '-c', str(solv_gro),
           '-p', str(top_file), '-o', str(tpr),
           '-po', str(mdout), '-maxwarn', '1']
    success, _ = run_command(cmd, "Preparing for ions", cwd=str(output_dir))

    ions_gro = output_dir / 'ligand_ions.gro'
    cmd = [gmx, 'genion', '-s', str(tpr), '-o', str(ions_gro),
           '-p', str(top_file), '-pname', 'NA', '-nname', 'CL',
           '-neutral', '-conc', '0.15']
    success, _ = run_command(cmd, "Adding ions", input_text='SOL\n', cwd=str(output_dir))

    # Energy minimization
    em_mdp = output_dir / 'em.mdp'
    with open(em_mdp, 'w') as f:
        f.write("""integrator = steep
emtol = 1000.0
emstep = 0.01
nsteps = 5000
nstlist = 10
cutoff-scheme = Verlet
pbc = xyz
coulombtype = PME
rcoulomb = 1.0
vdwtype = Cut-off
rvdw = 1.0
""")

    em_tpr = output_dir / 'em.tpr'
    mdout_em = output_dir / 'mdout_em.mdp'
    cmd = [gmx, 'grompp', '-f', str(em_mdp), '-c', str(ions_gro),
           '-p', str(top_file), '-o', str(em_tpr),
           '-po', str(mdout_em), '-maxwarn', '2']
    success, _ = run_command(cmd, "Preparing EM", cwd=str(output_dir))

    cmd = [gmx, 'mdrun', '-deffnm', str(output_dir / 'em')]
    success, _ = run_command(cmd, "Running EM", cwd=str(output_dir))

    final_gro = output_dir / 'em.gro'
    if final_gro.exists():
        shutil.copy(final_gro, output_dir / 'input.gro')
        return True

    return False


def write_run_script(output_dir, leg, n_windows, gmx='gmx', gpu=False):
    """Write run script for one leg of ABFE."""
    gpu_flag = '-nb gpu -pme gpu -bonded gpu' if gpu else ''

    # Sequential bash script (for local runs)
    script = f"""#!/bin/bash
# ABFE {leg} leg - run all lambda windows sequentially
# Run this script from within the {leg} directory
# For HPC/cluster: use ./submit.sh instead

set -e

SCRIPT_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
cd "$SCRIPT_DIR"

GMX="{gmx}"
GPU_FLAG="{gpu_flag}"

echo "=============================================="
echo "ABFE {leg.upper()} LEG"
echo "=============================================="
echo "Running {n_windows} lambda windows"
echo ""

for i in $(seq 0 {n_windows - 1}); do
    LAMBDA_DIR=$(printf "lambda%02d" $i)
    echo "Processing $LAMBDA_DIR..."

    cd "$SCRIPT_DIR/$LAMBDA_DIR"

    # NVT equilibration
    if [ ! -f nvt.gro ]; then
        $GMX grompp -f nvt.mdp -c ../input.gro -r ../input.gro -p ../topol.top -o nvt.tpr -maxwarn 2
        $GMX mdrun -deffnm nvt $GPU_FLAG
    fi

    # NPT equilibration
    if [ ! -f npt.gro ]; then
        $GMX grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p ../topol.top -o npt.tpr -maxwarn 2
        $GMX mdrun -deffnm npt $GPU_FLAG
    fi

    # Production
    if [ ! -f prod.gro ]; then
        $GMX grompp -f prod.mdp -c npt.gro -t npt.cpt -p ../topol.top -o prod.tpr -maxwarn 2
        $GMX mdrun -deffnm prod $GPU_FLAG
    fi

    echo "$LAMBDA_DIR complete"
    cd "$SCRIPT_DIR"
done

echo ""
echo "=============================================="
echo "{leg.upper()} LEG COMPLETE"
echo "=============================================="
"""

    script_path = output_dir / 'run_all.sh'
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)

    # Also write SLURM job array scripts (SLURM job arrays)
    write_slurm_scripts(output_dir, leg, n_windows, gmx, gpu)

    return script_path


def write_slurm_scripts(output_dir, leg, n_windows, gmx='gmx', gpu=False):
    """
    Write validated-style SLURM submission scripts for ABFE.

    Creates:
    - submit_slurm.sh: Main submission script
    - run_window.sh: Script run by each array task
    """
    # GPU flags for mdrun
    if gpu:
        gpu_flag = "-nb gpu -pme gpu -bonded gpu"
    else:
        gpu_flag = ""

    # Main SLURM submission script (LUMI settings)
    submit_script = f"""#!/bin/bash
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --partition=small-g
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --job-name=abfe_{leg}
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --array=0-{n_windows - 1}

# =============================================================================
# ABFE {leg.upper()} LEG - SLURM Job Array Submission
# ABFE SLURM submission script
# =============================================================================
#
# Usage:
#   sbatch submit_slurm.sh
#
# This submits {n_windows} independent jobs (one per lambda window) as a job array.
# Each job runs: NVT equilibration -> NPT equilibration -> Production MD
#
# Monitor progress:
#   squeue -u $USER
#   sacct -j <jobid>
#
# After completion, analyze with:
#   cd .. && ./analyze.sh
#
# =============================================================================

# Use SLURM_SUBMIT_DIR (directory where sbatch was called)
SCRIPT_DIR="${{SLURM_SUBMIT_DIR}}"
cd "$SCRIPT_DIR"

# Lambda window index from SLURM array task ID
LAMBDA_IDX=$SLURM_ARRAY_TASK_ID
LAMBDA_DIR=$(printf "lambda%02d" $LAMBDA_IDX)

echo "=============================================="
echo "ABFE {leg.upper()} LEG - Window $LAMBDA_IDX"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Working directory: $SCRIPT_DIR/$LAMBDA_DIR"
echo "Start time: $(date)"
echo "=============================================="

# Load GROMACS module (LUMI)
module use /appl/local/csc/modulefiles
module load gromacs/2025.4-gpu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

GMX="gmx_mpi"
GPU_FLAG="{gpu_flag}"

cd "$SCRIPT_DIR/$LAMBDA_DIR"

# =============================================================================
# NVT Equilibration
# =============================================================================
if [ ! -f nvt.gro ]; then
    echo ""
    echo "Running NVT equilibration..."
    $GMX grompp -f nvt.mdp -c ../input.gro -r ../input.gro -p ../topol.top -o nvt.tpr -maxwarn 2
    $GMX mdrun -deffnm nvt $GPU_FLAG -v
    echo "NVT complete"
else
    echo "NVT already complete, skipping..."
fi

# =============================================================================
# NPT Equilibration
# =============================================================================
if [ ! -f npt.gro ]; then
    echo ""
    echo "Running NPT equilibration..."
    $GMX grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p ../topol.top -o npt.tpr -maxwarn 2
    $GMX mdrun -deffnm npt $GPU_FLAG -v
    echo "NPT complete"
else
    echo "NPT already complete, skipping..."
fi

# =============================================================================
# Production MD
# =============================================================================
if [ ! -f prod.gro ]; then
    echo ""
    echo "Running Production MD..."
    $GMX grompp -f prod.mdp -c npt.gro -t npt.cpt -p ../topol.top -o prod.tpr -maxwarn 2
    $GMX mdrun -deffnm prod $GPU_FLAG -v
    echo "Production complete"
else
    echo "Production already complete, skipping..."
fi

echo ""
echo "=============================================="
echo "Window $LAMBDA_IDX COMPLETE"
echo "End time: $(date)"
echo "=============================================="
"""

    submit_path = output_dir / 'submit_slurm.sh'
    with open(submit_path, 'w') as f:
        f.write(submit_script)
    os.chmod(submit_path, 0o755)

    # Also create a helper script to check job status
    status_script = f"""#!/bin/bash
# Check status of ABFE {leg} simulations

SCRIPT_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "ABFE {leg.upper()} LEG - Status Check"
echo "=============================================="

completed=0
running=0
pending=0

for i in $(seq 0 {n_windows - 1}); do
    LAMBDA_DIR=$(printf "lambda%02d" $i)
    if [ -f "$LAMBDA_DIR/prod.gro" ]; then
        status="COMPLETE"
        ((completed++))
    elif [ -f "$LAMBDA_DIR/npt.gro" ]; then
        status="NPT done, prod running/pending"
        ((running++))
    elif [ -f "$LAMBDA_DIR/nvt.gro" ]; then
        status="NVT done, npt running/pending"
        ((running++))
    elif [ -f "$LAMBDA_DIR/nvt.tpr" ]; then
        status="NVT running"
        ((running++))
    else
        status="Not started"
        ((pending++))
    fi
    echo "$LAMBDA_DIR: $status"
done

echo ""
echo "=============================================="
echo "Summary: $completed/{n_windows} complete, $running running, $pending pending"
echo "=============================================="
"""

    status_path = output_dir / 'check_status.sh'
    with open(status_path, 'w') as f:
        f.write(status_script)
    os.chmod(status_path, 0o755)

    return submit_path


def write_stage_run_script(output_dir, leg, stages, gmx='gmx', gpu=False, stage_chaining=True):
    """
    Write run scripts for stage-based ABFE workflow with branching and optional chaining.

    Stage chaining (if enabled - sequential between stages):
    - restraints.0 starts from input.gro (pre-equilibrated from stability test)
    - coul.0 waits for restraints to complete, starts from restraints.N/prod/prod.gro
    - vdw.0 waits for coul to complete, starts from coul.N/prod/prod.gro

    Without stage chaining (faster, all stages parallel):
    - All stages start from input.gro simultaneously

    Branched within each stage (parallel):
    - Window 0: NPT equilibration (500 ps) -> production
    - Windows 1-N: Branch from window 0's NPT output -> short equil (50 ps) -> production

    Stage order:
    - Complex: restraints -> coul -> vdw
    - Solvent: coul -> vdw

    Args:
        output_dir: Base directory (complex/ or solvent/)
        leg: 'complex' or 'solvent'
        stages: Dict from generate_stage_lambdas()
        gmx: GROMACS executable
        gpu: Whether to use GPU
        stage_chaining: If True, stages wait for previous stage completion (default True)
    """
    gpu_flag = '-nb gpu -pme gpu -bonded gpu -v' if gpu else '-v'

    # Determine which stages to include
    if leg == 'complex':
        stage_order = ['restraints', 'coul', 'vdw']
    else:
        stage_order = ['coul', 'vdw']

    # Calculate total windows for job array
    total_windows = sum(len(stages[s]) for s in stage_order)

    # Generate header based on chaining mode
    if stage_chaining:
        workflow_desc = """# Workflow (STAGE CHAINING ENABLED):
#   Stage chaining: restraints -> coul -> vdw
#     - restraints.0 starts from input.gro (pre-equilibrated)
#     - coul.0 waits for restraints.N, starts from its prod.gro
#     - vdw.0 waits for coul.N, starts from its prod.gro
#
#   Branched within each stage:
#     - Window 0: npt/ (500 ps), prod/     <- equilibrate, then production
#     - Windows 1-N: equil/ (50 ps), prod/ <- branch from window 0"""
        mode_label = "Branched + Chained"
    else:
        workflow_desc = """# Workflow (NO STAGE CHAINING - faster):
#   All stages start from input.gro in parallel
#
#   Branched within each stage:
#     - Window 0: npt/ (500 ps), prod/     <- equilibrate, then production
#     - Windows 1-N: equil/ (50 ps), prod/ <- branch from window 0"""
        mode_label = "Branched, No Chaining"

    # Create SLURM submission script for stage-based workflow
    submit_script = f"""#!/bin/bash
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --partition=small-g
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --job-name=abfe_{leg}
#SBATCH --output=fep/simulation/slurm_%A_%a.out
#SBATCH --error=fep/simulation/slurm_%A_%a.err
#SBATCH --array=0-{total_windows - 1}

# =============================================================================
# ABFE {leg.upper()} LEG - Stage-Based SLURM Job Array ({mode_label})
# =============================================================================
#
# Directory structure:
#   fep/simulation/restraints.0/, restraints.1/, ..., restraints.N/
#   fep/simulation/coul.0/, coul.1/, ..., coul.N/
#   fep/simulation/vdw.0/, vdw.1/, ..., vdw.N/
#
{workflow_desc}
#
# =============================================================================

SCRIPT_DIR="${{SLURM_SUBMIT_DIR}}"
cd "$SCRIPT_DIR"

# Load GROMACS module (LUMI)
module use /appl/local/csc/modulefiles
module load gromacs/2025.4-gpu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

GMX="gmx_mpi"
GPU_FLAG="{gpu_flag}"

# Map array task ID to stage and window
TASK_ID=$SLURM_ARRAY_TASK_ID
"""

    # Add stage mapping logic and store stage sizes
    offset = 0
    for stage_name in stage_order:
        n = len(stages[stage_name])
        submit_script += f"""
# {stage_name}: windows {offset} to {offset + n - 1}
if [ $TASK_ID -ge {offset} ] && [ $TASK_ID -lt {offset + n} ]; then
    STAGE="{stage_name}"
    WINDOW_ID=$((TASK_ID - {offset}))
fi
"""
        offset += n

    # Add stage size variables for chaining logic (only if chaining enabled)
    if stage_chaining:
        if leg == 'complex':
            n_restr = len(stages['restraints'])
            n_coul = len(stages['coul'])
            submit_script += f"""
# Stage sizes for chaining
N_RESTRAINTS={n_restr}
N_COUL={n_coul}
"""
        else:
            n_coul = len(stages['coul'])
            submit_script += f"""
# Stage sizes for chaining
N_COUL={n_coul}
"""

    chaining_mode = "WITH STAGE CHAINING" if stage_chaining else "(NO CHAINING - all from input.gro)"
    submit_script += f"""
WINDOW_DIR="fep/simulation/$STAGE.$WINDOW_ID"

echo "=============================================="
echo "ABFE {leg.upper()} LEG - $STAGE.$WINDOW_ID"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Stage: $STAGE, Window: $WINDOW_ID"
echo "Node: $SLURMD_NODENAME"
echo "Working directory: $SCRIPT_DIR/$WINDOW_DIR"
echo "Start time: $(date)"
echo "=============================================="

cd "$SCRIPT_DIR/$WINDOW_DIR"

# =============================================================================
# BRANCHED WORKFLOW {chaining_mode}
# =============================================================================
# Window 0: EM -> NVT (100ps) -> NPT_posres (100ps) -> NPT (500ps) -> Prod
# Windows 1-N: Branch from window 0's NPT output -> short equil (50 ps) -> Prod
#
# NOTE: Windows 1-N are submitted with SLURM --dependency on window 0.
#       They will not start until window 0 completes (no wasted resources).
# =============================================================================

if [ $WINDOW_ID -eq 0 ]; then
    # =============================================================================
    # WINDOW 0: Full NPT equilibration
    # =============================================================================

    # Determine input structure
    # Path from inside npt/prod/equil subdirs to leg root (where input.gro and topol.top are)
    # Structure: leg_dir/fep/simulation/stage.X/subdir/ → need ../../../../ to reach leg_dir
    INPUT_GRO="../../../../input.gro"
    TOPOL="../../../../topol.top"
"""

    # Add stage chaining logic only if enabled
    if stage_chaining:
        if leg == 'complex':
            submit_script += """
    if [ "$STAGE" == "coul" ]; then
        # Coulomb stage: wait for and use restraints stage final output
        PREV_STAGE_OUTPUT="../restraints.$((N_RESTRAINTS-1))/prod/prod.gro"
        echo "Stage chaining: waiting for restraints stage to complete..."
        WAIT_COUNT=0
        while [ ! -f "$PREV_STAGE_OUTPUT" ]; do
            sleep 60
            ((WAIT_COUNT++))
            if [ $WAIT_COUNT -ge 360 ]; then
                echo "ERROR: Timeout waiting for restraints stage (6 hours). Exiting."
                exit 1
            fi
        done
        # Path from npt/ subdir: ../ to stage.X, then to prev stage's prod.gro
        INPUT_GRO="../$PREV_STAGE_OUTPUT"
        echo "Using restraints stage output: $INPUT_GRO"
    elif [ "$STAGE" == "vdw" ]; then
        # VdW stage: wait for and use coul stage final output
        PREV_STAGE_OUTPUT="../coul.$((N_COUL-1))/prod/prod.gro"
        echo "Stage chaining: waiting for coul stage to complete..."
        WAIT_COUNT=0
        while [ ! -f "$PREV_STAGE_OUTPUT" ]; do
            sleep 60
            ((WAIT_COUNT++))
            if [ $WAIT_COUNT -ge 360 ]; then
                echo "ERROR: Timeout waiting for coul stage (6 hours). Exiting."
                exit 1
            fi
        done
        INPUT_GRO="../$PREV_STAGE_OUTPUT"
        echo "Using coul stage output: $INPUT_GRO"
    else
        echo "First stage (restraints): using input.gro"
    fi
"""
        else:  # solvent leg
            submit_script += """
    if [ "$STAGE" == "vdw" ]; then
        # VdW stage: wait for and use coul stage final output
        PREV_STAGE_OUTPUT="../coul.$((N_COUL-1))/prod/prod.gro"
        echo "Stage chaining: waiting for coul stage to complete..."
        WAIT_COUNT=0
        while [ ! -f "$PREV_STAGE_OUTPUT" ]; do
            sleep 60
            ((WAIT_COUNT++))
            if [ $WAIT_COUNT -ge 360 ]; then
                echo "ERROR: Timeout waiting for coul stage (6 hours). Exiting."
                exit 1
            fi
        done
        INPUT_GRO="../$PREV_STAGE_OUTPUT"
        echo "Using coul stage output: $INPUT_GRO"
    else
        echo "First stage (coul): using input.gro"
    fi
"""
    else:
        # No chaining - all stages start from input.gro
        submit_script += """
    echo "No stage chaining: using input.gro for all stages"
"""

    submit_script += f"""
    # =============================================================================
    # EQUILIBRATION SEQUENCE: EM -> NVT -> NPT_posres -> NPT -> Prod
    # =============================================================================

    # Energy Minimization
    if [ ! -f em/em.gro ]; then
        echo ""
        echo "Running Energy Minimization..."
        cd em
        $GMX grompp -f em.mdp -c "$INPUT_GRO" -p "$TOPOL" -o em.tpr -maxwarn 2
        $GMX mdrun -deffnm em $GPU_FLAG
        cd ..
        echo "Energy minimization complete"
    else
        echo "Energy minimization already complete, skipping..."
    fi

    # NVT equilibration with position restraints (100 ps)
    if [ ! -f nvt/nvt.gro ]; then
        echo ""
        echo "Running NVT equilibration with position restraints (100 ps)..."
        cd nvt
        $GMX grompp -f nvt.mdp -c ../em/em.gro -r ../em/em.gro -p "$TOPOL" -o nvt.tpr -maxwarn 2
        $GMX mdrun -deffnm nvt $GPU_FLAG
        cd ..
        echo "NVT equilibration complete"
    else
        echo "NVT equilibration already complete, skipping..."
    fi

    # NPT equilibration with position restraints (100 ps, Berendsen)
    if [ ! -f npt_posres/npt_posres.gro ]; then
        echo ""
        echo "Running NPT equilibration with position restraints (100 ps, Berendsen)..."
        cd npt_posres
        $GMX grompp -f npt_posres.mdp -c ../nvt/nvt.gro -r ../nvt/nvt.gro -t ../nvt/nvt.cpt -p "$TOPOL" -o npt_posres.tpr -maxwarn 2
        $GMX mdrun -deffnm npt_posres $GPU_FLAG
        cd ..
        echo "NPT (Berendsen) equilibration complete"
    else
        echo "NPT (Berendsen) equilibration already complete, skipping..."
    fi

    # NPT equilibration without restraints (500 ps, Parrinello-Rahman)
    if [ ! -f npt/npt.gro ]; then
        echo ""
        echo "Running NPT equilibration without restraints (500 ps, Parrinello-Rahman)..."
        cd npt
        $GMX grompp -f npt.mdp -c ../npt_posres/npt_posres.gro -t ../npt_posres/npt_posres.cpt -p "$TOPOL" -o npt.tpr -maxwarn 2
        $GMX mdrun -deffnm npt $GPU_FLAG
        cd ..
        echo "NPT (Parrinello-Rahman) equilibration complete"
    else
        echo "NPT (Parrinello-Rahman) equilibration already complete, skipping..."
    fi

    # Production MD
    if [ ! -f prod/prod.gro ]; then
        echo ""
        echo "Running Production MD..."
        cd prod
        $GMX grompp -f prod.mdp -c ../npt/npt.gro -t ../npt/npt.cpt -p "$TOPOL" -o prod.tpr -maxwarn 2
        $GMX mdrun -deffnm prod $GPU_FLAG
        cd ..
        echo "Production complete"
    else
        echo "Production already complete, skipping..."
    fi

else
    # =============================================================================
    # WINDOWS 1-N: Branch from window 0's equilibrated structure
    # =============================================================================

    WINDOW0_NPT="../$STAGE.0/npt/npt.gro"
    TOPOL="../../../../topol.top"

    # Verify window 0's NPT is complete (should be, due to SLURM dependency)
    if [ ! -f "$WINDOW0_NPT" ]; then
        echo "ERROR: Window 0 NPT not found at $WINDOW0_NPT"
        echo "       This job should have SLURM dependency on window 0."
        exit 1
    fi
    echo "Window 0 NPT complete, branching..."

    # Short equilibration at this lambda (50 ps)
    # Path from equil/: ../ to stage.X, ../ to stage.0, then npt/npt.gro
    if [ ! -f equil/equil.gro ]; then
        echo ""
        echo "Running short equilibration (50 ps)..."
        cd equil
        $GMX grompp -f equil.mdp -c "../$WINDOW0_NPT" -p "$TOPOL" -o equil.tpr -maxwarn 2
        $GMX mdrun -deffnm equil $GPU_FLAG
        cd ..
        echo "Short equilibration complete"
    else
        echo "Short equilibration already complete, skipping..."
    fi

    # Production MD
    if [ ! -f prod/prod.gro ]; then
        echo ""
        echo "Running Production MD..."
        cd prod
        $GMX grompp -f prod.mdp -c ../equil/equil.gro -t ../equil/equil.cpt -p "$TOPOL" -o prod.tpr -maxwarn 2
        $GMX mdrun -deffnm prod $GPU_FLAG
        cd ..
        echo "Production complete"
    else
        echo "Production already complete, skipping..."
    fi
fi

echo ""
echo "=============================================="
echo "Window $STAGE.$WINDOW_ID COMPLETE"
echo "End time: $(date)"
echo "=============================================="
"""

    # Write the job script (used by both window 0 and branch submissions)
    job_script_path = output_dir / 'job_script.sh'
    with open(job_script_path, 'w') as f:
        f.write(submit_script)
    os.chmod(job_script_path, 0o755)

    # Generate master submission script that handles SLURM dependencies
    # This ensures windows 1-N only start AFTER window 0 finishes equilibration
    master_submit = f"""#!/bin/bash
# =============================================================================
# ABFE {leg.upper()} LEG - Master Submission Script with SLURM Dependencies
# =============================================================================
#
# This script submits jobs in the correct order with proper dependencies:
#   1. Window 0 jobs are submitted first (full equilibration)
#   2. Branch jobs (windows 1-N) are submitted with --dependency on window 0
#
# This ensures no resources are wasted on waiting jobs.
#
# Usage: ./submit.sh
# =============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "ABFE {leg.upper()} LEG - Submitting jobs with dependencies"
echo "=============================================="
echo ""

"""

    # Build the submission commands for each stage
    offset = 0
    for stage_name in stage_order:
        n = len(stages[stage_name])
        window0_idx = offset
        branch_start = offset + 1
        branch_end = offset + n - 1

        master_submit += f"""
# -----------------------------------------------------------------------------
# Stage: {stage_name} ({n} windows)
# -----------------------------------------------------------------------------
echo "Submitting {stage_name} stage..."

# Submit window 0 (full equilibration)
JOBID_{stage_name.upper()}_0=$(sbatch --parsable --array={window0_idx} job_script.sh)
echo "  {stage_name}.0 (window 0): Job $JOBID_{stage_name.upper()}_0"

"""
        if n > 1:
            master_submit += f"""# Submit windows 1-{n-1} with dependency on window 0
JOBID_{stage_name.upper()}_BRANCH=$(sbatch --parsable --dependency=afterok:$JOBID_{stage_name.upper()}_0 --array={branch_start}-{branch_end} job_script.sh)
echo "  {stage_name}.1-{n-1} (branch): Job $JOBID_{stage_name.upper()}_BRANCH (depends on $JOBID_{stage_name.upper()}_0)"

"""

        offset += n

    master_submit += """
echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo ""
echo "Monitor with: squeue -u $USER"
echo "Check status: ./check_status.sh"
"""

    submit_path = output_dir / 'submit.sh'
    with open(submit_path, 'w') as f:
        f.write(master_submit)
    os.chmod(submit_path, 0o755)

    # Also create status check script
    write_stage_status_script(output_dir, leg, stages, stage_order)

    return submit_path


def write_stage_status_script(output_dir, leg, stages, stage_order):
    """Write status check script for stage-based workflow."""
    script = f"""#!/bin/bash
# Check status of ABFE {leg} simulations (stage-based)

SCRIPT_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "ABFE {leg.upper()} LEG - Status Check"
echo "=============================================="

completed=0
running=0
pending=0
total=0

"""
    for stage_name in stage_order:
        n = len(stages[stage_name])
        script += f"""
echo ""
echo "Stage: {stage_name} ({n} windows)"
echo "-------------------------------------------"
for i in $(seq 0 {n - 1}); do
    WINDOW_DIR="fep/simulation/{stage_name}.$i"
    ((total++))
    if [ -f "$WINDOW_DIR/prod/prod.gro" ]; then
        status="COMPLETE"
        ((completed++))
    elif [ $i -eq 0 ]; then
        # Window 0: check full equilibration sequence (em -> nvt -> npt_posres -> npt -> prod)
        if [ -f "$WINDOW_DIR/npt/npt.gro" ]; then
            status="npt done, prod running"
            ((running++))
        elif [ -f "$WINDOW_DIR/npt_posres/npt_posres.gro" ]; then
            status="npt_posres done, npt running"
            ((running++))
        elif [ -f "$WINDOW_DIR/nvt/nvt.gro" ]; then
            status="nvt done, npt_posres running"
            ((running++))
        elif [ -f "$WINDOW_DIR/em/em.gro" ]; then
            status="em done, nvt running"
            ((running++))
        else
            status="em running/pending"
            ((pending++))
        fi
    else
        # Windows 1-N: check short equilibration
        if [ -f "$WINDOW_DIR/equil/equil.gro" ]; then
            status="equil done, prod running"
            ((running++))
        elif [ -f "fep/simulation/{stage_name}.0/npt/npt.gro" ]; then
            status="waiting/equil pending"
            ((pending++))
        else
            status="waiting for window 0"
            ((pending++))
        fi
    fi
    echo "  {stage_name}.$i: $status"
done
"""

    script += """
echo ""
echo "=============================================="
echo "Summary: $completed/$total complete, $running running, $pending pending"
echo "=============================================="
"""

    status_path = output_dir / 'check_status.sh'
    with open(status_path, 'w') as f:
        f.write(script)
    os.chmod(status_path, 0o755)


def write_analysis_script(output_dir, complex_windows, solvent_windows, dG_restr):
    """Write analysis scripts by copying from templates and substituting values."""
    # Get the templates directory (relative to this script)
    script_dir = Path(__file__).parent
    templates_dir = script_dir / 'templates'

    # Copy and modify analyze.sh
    analyze_sh_template = templates_dir / 'analyze.sh'
    if analyze_sh_template.exists():
        with open(analyze_sh_template, 'r') as f:
            script = f.read()
        script = script.replace('{{DG_RESTR}}', f'{dG_restr:.2f}')
    else:
        print(f"  WARNING: Template not found: {analyze_sh_template}")
        script = "#!/bin/bash\necho 'Template not found'\n"

    script_path = output_dir / 'analyze.sh'
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)

    # Copy and modify analyze_alchemlyb.py
    python_template = templates_dir / 'analyze_alchemlyb.py'
    if python_template.exists():
        with open(python_template, 'r') as f:
            python_script = f.read()
        python_script = python_script.replace('{{DG_RESTR}}', f'{dG_restr:.4f}')
    else:
        print(f"  WARNING: Template not found: {python_template}")
        python_script = "#!/usr/bin/env python3\nprint('Template not found')\n"

    python_script_path = output_dir / 'analyze_alchemlyb.py'
    with open(python_script_path, 'w') as f:
        f.write(python_script)
    os.chmod(python_script_path, 0o755)

    return script_path


def main():
    parser = argparse.ArgumentParser(
        description="Setup ABFE calculations with Boresch restraints",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument('--complex_gro', '-c', required=True,
                        help='Equilibrated complex GRO file')
    parser.add_argument('--topology', '-p', required=True,
                        help='Complex topology file')
    parser.add_argument('--ligand_itp', required=True,
                        help='Ligand ITP file')
    parser.add_argument('--output_dir', '-o', required=True,
                        help='Output directory for ABFE setup')

    # Optional - ligand identification
    parser.add_argument('--ligand_resname', default='LIG',
                        help='Ligand residue name (default: LIG)')
    parser.add_argument('--ligand_gro', '-l', required=True,
                        help='Ligand-only GRO file for solvent leg (required for ABFE)')

    # Trajectory for MDRestraintsGenerator (recommended for robust restraints)
    parser.add_argument('--trajectory', '-t', default=None,
                        help='Trajectory file (XTC/TRR) for MDRestraintsGenerator analysis (recommended)')
    parser.add_argument('--tpr', default=None,
                        help='TPR file for topology (angles/bonds). Required with --trajectory for MDRestraintsGenerator')

    # Anchor atoms (optional - auto-detected if not provided)
    parser.add_argument('--protein_anchors', default=None,
                        help='Protein anchor atom indices (comma-separated, e.g., 100,105,110)')
    parser.add_argument('--ligand_anchors', default=None,
                        help='Ligand anchor atom indices (comma-separated, e.g., 1,5,10)')

    # Simulation parameters
    parser.add_argument('--prod_time', type=float, default=5.0,
                        help='Production time per window in ns (default: 5)')
    parser.add_argument('--temperature', type=float, default=298.15,
                        help='Temperature in K (default: 298.15, standard thermodynamic temperature)')

    # FEP schedule option
    parser.add_argument('--fep_schedule', type=str, default='staged-chained',
                        choices=['staged-chained', 'staged-parallel', 'combined'],
                        help='''FEP workflow schedule (default: staged-chained):
  staged-chained:  Separate stages (restraints/coul/vdw) with chaining.
                   Most rigorous - each stage starts from previous stage output.
                   Stages run sequentially. (~2-4 hours with full parallelism)
  staged-parallel: Separate stages without chaining.
                   Faster - all stages start from input.gro in parallel.
                   (~35-70 min with full parallelism)
  combined:        Legacy combined lambda vectors in single simulation.
                   Not recommended for production use.''')

    # Execution options
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU acceleration')
    parser.add_argument('--gmx', default=None,
                        help='GROMACS executable')

    args = parser.parse_args()

    # Validate inputs
    complex_gro = Path(args.complex_gro)
    topology = Path(args.topology)
    ligand_itp = Path(args.ligand_itp)

    for path, name in [(complex_gro, 'Complex GRO'), (topology, 'Topology'),
                       (ligand_itp, 'Ligand ITP')]:
        if not path.exists():
            print(f"ERROR: {name} not found: {path}")
            sys.exit(1)

    # Find GROMACS
    gmx = args.gmx or find_gmx()
    if gmx is None:
        print("ERROR: GROMACS not found")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for trajectory file (for MDRestraintsGenerator)
    trajectory = None
    tpr_file = None
    if args.trajectory:
        trajectory = Path(args.trajectory)
        if not trajectory.exists():
            print(f"WARNING: Trajectory file not found: {args.trajectory}")
            trajectory = None
        else:
            # Check for TPR file (required for MDRestraintsGenerator - contains angles/bonds)
            if args.tpr:
                tpr_file = Path(args.tpr)
                if not tpr_file.exists():
                    print(f"WARNING: TPR file not found: {args.tpr}")
                    tpr_file = None
            else:
                # Try to auto-detect TPR in same directory as trajectory
                auto_tpr = trajectory.parent / "prod.tpr"
                if auto_tpr.exists():
                    tpr_file = auto_tpr
                    print(f"  Auto-detected TPR file: {tpr_file}")
                else:
                    print("WARNING: No TPR file specified or found. MDRestraintsGenerator requires a TPR file")
                    print("         for topology information (angles/bonds). Use --tpr to specify one.")

    # Determine FEP schedule type
    use_staged = args.fep_schedule in ['staged-chained', 'staged-parallel']
    use_chaining = args.fep_schedule == 'staged-chained'

    # Generate lambda schedules (for combined mode compatibility)
    complex_lambdas, solvent_lambdas = generate_lambda_schedule(16, 11, 20, use_validated_schedule=True)

    # Count windows for display
    if use_staged:
        stages = generate_stage_lambdas()
        n_complex = len(stages['restraints']) + len(stages['coul']) + len(stages['vdw'])
        n_solvent = len(stages['coul']) + len(stages['vdw'])
    else:
        n_complex = len(complex_lambdas)
        n_solvent = len(solvent_lambdas)

    print("\n" + "="*60)
    print("ABFE SETUP WITH BORESCH RESTRAINTS")
    print("="*60)
    print(f"\n[FEP Schedule: {args.fep_schedule}]")
    print(f"\nInputs:")
    print(f"  Complex:     {complex_gro}")
    print(f"  Topology:    {topology}")
    print(f"  Ligand ITP:  {ligand_itp}")
    if trajectory:
        print(f"  Trajectory:  {trajectory} (for MDRestraintsGenerator)")
        if tpr_file:
            print(f"  TPR file:    {tpr_file} (topology with angles/bonds)")
    print(f"\nSimulation:")
    print(f"  Production:  {args.prod_time} ns per window")
    print(f"  Temperature: {args.temperature} K")
    print(f"  Complex leg: {n_complex} windows")
    print(f"  Solvent leg: {n_solvent} windows")
    print(f"\nOutput:")
    print(f"  Directory:   {output_dir}")

    # Step 1: Find or parse anchor atoms
    print("\n" + "-"*60)
    print("Step 1: Determine Boresch anchor atoms")
    print("-"*60)

    boresch_params = None
    use_mdrestraints = False

    if args.protein_anchors and args.ligand_anchors:
        # User-specified anchor atoms
        protein_atoms = [int(x) for x in args.protein_anchors.split(',')]
        ligand_atoms = [int(x) for x in args.ligand_anchors.split(',')]
        print("  Using user-specified anchor atoms")

    elif trajectory and tpr_file and HAS_MDRESTRAINTS:
        # Use MDRestraintsGenerator for robust restraint selection
        print("  Using MDRestraintsGenerator for trajectory-based restraint selection")
        print("  (This is the recommended approach for robust ABFE)")
        protein_atoms, ligand_atoms, boresch_params = find_anchor_atoms_mdrestraints(
            tpr_file, trajectory, args.ligand_resname,
            output_dir=output_dir, temperature=args.temperature
        )
        if protein_atoms is None:
            print("  WARNING: MDRestraintsGenerator failed, falling back to geometric selection")
            protein_atoms, ligand_atoms = find_anchor_atoms(complex_gro, args.ligand_resname)
        else:
            use_mdrestraints = True

    elif trajectory and not tpr_file:
        print("  WARNING: Trajectory provided but no TPR file found")
        print("  MDRestraintsGenerator requires a TPR file for topology (angles/bonds)")
        print("  Use --tpr to specify one, or place prod.tpr in the same directory as the trajectory")
        print("  Falling back to geometric anchor selection")
        protein_atoms, ligand_atoms = find_anchor_atoms(complex_gro, args.ligand_resname)

    elif trajectory and not HAS_MDRESTRAINTS:
        print("  WARNING: Trajectory provided but MDRestraintsGenerator not installed")
        print("  Install with: pip install MDRestraintsGenerator MDAnalysis")
        print("  Falling back to geometric anchor selection")
        protein_atoms, ligand_atoms = find_anchor_atoms(complex_gro, args.ligand_resname)

    else:
        # Fall back to geometric selection
        print("  Using geometric anchor selection")
        print("  NOTE: For robust ABFE, run ~20 ns equilibration MD")
        print("        before restraint selection. Provide --trajectory for better results.")
        protein_atoms, ligand_atoms = find_anchor_atoms(complex_gro, args.ligand_resname)

    if protein_atoms is None:
        print("ERROR: Could not find anchor atoms automatically")
        print("Please specify --protein_anchors and --ligand_anchors")
        sys.exit(1)

    # Only print anchor atoms if not using MDRestraintsGenerator (which prints them already)
    if not use_mdrestraints:
        print(f"\n  Anchor atoms selected:")
        print(f"    Protein (P1, P2, P3): {protein_atoms[0]}, {protein_atoms[1]}, {protein_atoms[2]}")
        print(f"    Ligand (L1, L2, L3):  {ligand_atoms[0]}, {ligand_atoms[1]}, {ligand_atoms[2]}")

    # Step 2: Calculate Boresch parameters
    print("\n" + "-"*60)
    print("Step 2: Calculate Boresch restraint parameters")
    print("-"*60)

    if boresch_params is None:
        # Calculate from structure if not already computed by MDRestraintsGenerator
        boresch_params = calculate_boresch_parameters(complex_gro, protein_atoms, ligand_atoms)
        print("  Parameters calculated from equilibrium structure")
    else:
        print("  Parameters from MDRestraintsGenerator trajectory analysis")

    print(f"\n  r0 = {boresch_params['r0']:.3f} nm")
    print(f"  theta_A = {boresch_params['theta_A0']:.1f} deg")
    print(f"  theta_B = {boresch_params['theta_B0']:.1f} deg")
    print(f"  phi_A = {boresch_params['phi_A0']:.1f} deg")
    print(f"  phi_B = {boresch_params['phi_B0']:.1f} deg")
    print(f"  phi_C = {boresch_params['phi_C0']:.1f} deg")

    # Show fluctuations if available (from MDRestraintsGenerator)
    if 'r_std' in boresch_params:
        print(f"\n  Fluctuations (from trajectory):")
        print(f"    r: ±{boresch_params['r_std']:.3f} nm")
        print(f"    theta_A: ±{boresch_params['theta_A_std']:.1f} deg")
        print(f"    theta_B: ±{boresch_params['theta_B_std']:.1f} deg")

    # Calculate analytical correction
    dG_restr = calculate_restraint_correction(boresch_params, args.temperature)
    print(f"\n  Restraint correction (analytical): {dG_restr:.2f} kJ/mol ({dG_restr/4.184:.2f} kcal/mol)")

    # Show MDRestraintsGenerator correction if available
    if 'dG_restr_mdrestraints' in boresch_params:
        dG_mdr = boresch_params['dG_restr_mdrestraints']
        print(f"  Restraint correction (MDRestraintsGenerator): {dG_mdr:.2f} kJ/mol ({dG_mdr/4.184:.2f} kcal/mol)")

    # Step 3: Setup complex leg
    print("\n" + "-"*60)
    print("Step 3: Setup complex leg")
    print("-"*60)

    complex_dir = output_dir / 'complex'
    complex_dir.mkdir(exist_ok=True)

    # Copy input files
    # Copy input files (skip if same file)
    input_gro_dst = complex_dir / 'input.gro'
    if complex_gro.resolve() != input_gro_dst.resolve():
        shutil.copy(complex_gro, input_gro_dst)
    topol_dst = complex_dir / 'topol.top'
    if topology.resolve() != topol_dst.resolve():
        shutil.copy(topology, topol_dst)

    # Copy ITP files (skip if same file)
    top_dir = topology.parent
    with open(topology) as f:
        for line in f:
            if '#include' in line and '"' in line:
                itp_name = line.split('"')[1]
                itp_path = top_dir / itp_name
                dst_path = complex_dir / itp_name
                if itp_path.exists() and itp_path.resolve() != dst_path.resolve():
                    shutil.copy(itp_path, dst_path)

    # Write restraint ITP
    restr_itp = complex_dir / 'boresch_restraints.itp'
    write_restraint_itp(restr_itp, boresch_params)
    print(f"  Created: boresch_restraints.itp")

    # Add restraint include to topology
    with open(complex_dir / 'topol.top', 'a') as f:
        f.write('\n; Include Boresch restraints\n')
        f.write('#include "boresch_restraints.itp"\n')

    # Create lambda directories and MDP files
    # With 4 fs timestep (HMR), nsteps = time_ns * 1000 / 0.004
    prod_nsteps = int(args.prod_time * 1000 / 0.004)

    if use_staged:
        # Stage-based approach (recommended)
        stages = generate_stage_lambdas()
        fep_dir = complex_dir / 'fep' / 'simulation'
        fep_dir.mkdir(parents=True, exist_ok=True)

        total_windows = 0
        for stage_name in ['restraints', 'coul', 'vdw']:
            stage_lambdas = stages[stage_name]
            for i in range(len(stage_lambdas)):
                setup_stage_directory(
                    fep_dir, stage_name, i, stage_lambdas,
                    ligand_moltype=args.ligand_resname,
                    temperature=args.temperature,
                    prod_nsteps=prod_nsteps
                )
                total_windows += 1

        write_stage_run_script(complex_dir, 'complex', stages, gmx, args.gpu,
                               stage_chaining=use_chaining)
        print(f"  Created {total_windows} windows across 3 stages:")
        print(f"    - restraints: {len(stages['restraints'])} windows")
        print(f"    - coul: {len(stages['coul'])} windows")
        print(f"    - vdw: {len(stages['vdw'])} windows")
    else:
        # Legacy combined lambda approach
        for i, lambdas in enumerate(complex_lambdas):
            lambda_dir = complex_dir / f'lambda{i:02d}'
            lambda_dir.mkdir(exist_ok=True)

            write_equil_mdp(lambda_dir / 'nvt.mdp', 'nvt', i, complex_lambdas, 'complex',
                            ligand_moltype=args.ligand_resname)
            write_equil_mdp(lambda_dir / 'npt.mdp', 'npt', i, complex_lambdas, 'complex',
                            ligand_moltype=args.ligand_resname)
            write_fep_mdp(lambda_dir / 'prod.mdp', i, complex_lambdas, 'complex',
                          nsteps=prod_nsteps, ref_temp=args.temperature,
                          ligand_moltype=args.ligand_resname)

        write_run_script(complex_dir, 'complex', len(complex_lambdas), gmx, args.gpu)
        print(f"  Created {len(complex_lambdas)} lambda windows")

    # Step 4: Setup solvent leg
    print("\n" + "-"*60)
    print("Step 4: Setup solvent leg")
    print("-"*60)

    solvent_dir = output_dir / 'solvent'

    # Validate ligand GRO (required argument)
    ligand_gro = Path(args.ligand_gro)
    if not ligand_gro.exists():
        print(f"ERROR: Ligand GRO file not found: {ligand_gro}")
        sys.exit(1)

    success = setup_solvent_leg(ligand_gro, ligand_itp, solvent_dir, gmx)
    if success:
        if use_staged:
            # Stage-based approach (no restraints for solvent)
            stages = generate_stage_lambdas()
            fep_dir = solvent_dir / 'fep' / 'simulation'
            fep_dir.mkdir(parents=True, exist_ok=True)

            total_windows = 0
            for stage_name in ['coul', 'vdw']:  # No restraints for solvent
                stage_lambdas = stages[stage_name]
                for i in range(len(stage_lambdas)):
                    setup_stage_directory(
                        fep_dir, stage_name, i, stage_lambdas,
                        ligand_moltype=args.ligand_resname,
                        temperature=args.temperature,
                        prod_nsteps=prod_nsteps
                    )
                    total_windows += 1

            write_stage_run_script(solvent_dir, 'solvent', stages, gmx, args.gpu,
                                   stage_chaining=use_chaining)
            print(f"  Created {total_windows} windows across 2 stages:")
            print(f"    - coul: {len(stages['coul'])} windows")
            print(f"    - vdw: {len(stages['vdw'])} windows")
        else:
            # Legacy combined lambda approach
            for i, lambdas in enumerate(solvent_lambdas):
                lambda_dir = solvent_dir / f'lambda{i:02d}'
                lambda_dir.mkdir(exist_ok=True)

                write_equil_mdp(lambda_dir / 'nvt.mdp', 'nvt', i, solvent_lambdas, 'solvent',
                                ligand_moltype=args.ligand_resname)
                write_equil_mdp(lambda_dir / 'npt.mdp', 'npt', i, solvent_lambdas, 'solvent',
                                ligand_moltype=args.ligand_resname)
                write_fep_mdp(lambda_dir / 'prod.mdp', i, solvent_lambdas, 'solvent',
                              nsteps=prod_nsteps, ref_temp=args.temperature,
                              ligand_moltype=args.ligand_resname)

            write_run_script(solvent_dir, 'solvent', len(solvent_lambdas), gmx, args.gpu)
            print(f"  Created {len(solvent_lambdas)} lambda windows")
    else:
        print("ERROR: Solvent leg setup failed")
        sys.exit(1)

    # Step 5: Write analysis scripts
    print("\n" + "-"*60)
    print("Step 5: Write analysis scripts")
    print("-"*60)

    # Use MDRestraintsGenerator correction if available (more accurate), otherwise analytical
    dG_restr_final = boresch_params.get('dG_restr_mdrestraints', dG_restr)
    write_analysis_script(output_dir, len(complex_lambdas), len(solvent_lambdas), dG_restr_final)
    print(f"  Created: analyze.sh (run this)")
    print(f"  Created: analyze_alchemlyb.py (MBAR analysis, recommended)")

    # Save parameters
    params_file = output_dir / 'abfe_parameters.txt'
    with open(params_file, 'w') as f:
        f.write("ABFE Parameters\n")
        f.write("="*40 + "\n\n")
        f.write("Protocol:\n")
        f.write(f"  FEP schedule: {args.fep_schedule}\n")
        f.write(f"  Restraint selection: {'MDRestraintsGenerator' if use_mdrestraints else 'Geometric'}\n")
        f.write(f"  Soft-core: sc-alpha=0.5, sc-power=1, sc-sigma=0.3\n\n")
        f.write("Input files:\n")
        f.write(f"  Complex GRO: {complex_gro}\n")
        f.write(f"  Topology: {topology}\n")
        f.write(f"  Ligand ITP: {ligand_itp}\n")
        if trajectory:
            f.write(f"  Trajectory: {trajectory}\n")
        f.write(f"\nSimulation parameters:\n")
        f.write(f"  Production time: {args.prod_time} ns/window\n")
        f.write(f"  Temperature: {args.temperature} K\n\n")
        f.write("Boresch Restraints:\n")
        f.write(f"  Protein atoms (P3-P2-P1): {protein_atoms}\n")
        f.write(f"  Ligand atoms (L1-L2-L3): {ligand_atoms}\n")
        f.write(f"  r0 = {boresch_params['r0']:.4f} nm\n")
        f.write(f"  theta_A = {boresch_params['theta_A0']:.2f} deg\n")
        f.write(f"  theta_B = {boresch_params['theta_B0']:.2f} deg\n")
        f.write(f"  phi_A = {boresch_params['phi_A0']:.2f} deg\n")
        f.write(f"  phi_B = {boresch_params['phi_B0']:.2f} deg\n")
        f.write(f"  phi_C = {boresch_params['phi_C0']:.2f} deg\n")
        if 'r_std' in boresch_params:
            f.write(f"\nFluctuations (from trajectory):\n")
            f.write(f"  r: +/-{boresch_params['r_std']:.4f} nm\n")
            f.write(f"  theta_A: +/-{boresch_params['theta_A_std']:.2f} deg\n")
            f.write(f"  theta_B: +/-{boresch_params['theta_B_std']:.2f} deg\n")
        f.write(f"\nRestraint correction (analytical): {dG_restr:.2f} kJ/mol ({dG_restr/4.184:.2f} kcal/mol)\n")
        if 'dG_restr_mdrestraints' in boresch_params:
            dG_mdr = boresch_params['dG_restr_mdrestraints']
            f.write(f"Restraint correction (MDRestraintsGenerator): {dG_mdr:.2f} kJ/mol ({dG_mdr/4.184:.2f} kcal/mol)\n")
        f.write(f"\nLambda windows:\n")
        f.write(f"  Complex leg: {n_complex} windows\n")
        f.write(f"  Solvent leg: {n_solvent} windows\n")
        f.write(f"  Total: {n_complex + n_solvent} windows\n")
        if use_staged:
            stages = generate_stage_lambdas()
            f.write(f"\nStage-based schedule:\n")
            f.write(f"  Restraint windows: {len(stages['restraints'])}\n")
            f.write(f"  Coulomb windows: {len(stages['coul'])}\n")
            f.write(f"  VdW windows: {len(stages['vdw'])}\n")
            f.write(f"  Stage chaining: {'enabled' if use_chaining else 'disabled'}\n")

    # Summary
    print("\n" + "="*60)
    print("ABFE SETUP COMPLETE")
    print("="*60)
    print(f"\nProtocol:")
    print(f"  FEP schedule: {args.fep_schedule}")
    print(f"  Restraint selection: {'MDRestraintsGenerator' if use_mdrestraints else 'Geometric'}")
    print(f"  Total windows: {n_complex + n_solvent}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nTo run simulations:")
    print(f"\n  HPC/SLURM (recommended):")
    print(f"    cd {complex_dir} && ./submit.sh")
    print(f"    cd {solvent_dir} && ./submit.sh")
    print(f"\n  This uses SLURM dependencies to ensure efficient resource usage:")
    print(f"    1. Window 0 jobs submit first (full equilibration)")
    print(f"    2. Branch jobs (windows 1-N) submit with --dependency on window 0")
    print(f"    3. Branch jobs won't start until window 0 completes")
    print(f"\n  Check progress:")
    print(f"    cd {complex_dir} && ./check_status.sh")
    print(f"    cd {solvent_dir} && ./check_status.sh")
    print(f"\nAfter completion, analyze with:")
    print(f"  cd {output_dir} && ./analyze.sh")
    print(f"\n  (Uses alchemlyb/MBAR if installed, otherwise falls back to gmx bar)")
    print(f"  Install alchemlyb: pip install alchemlyb")
    print(f"\nBinding free energy:")
    print(f"  dG_bind = dG_complex - dG_solvent + dG_restraint")
    print(f"  dG_restraint = {dG_restr:.2f} kJ/mol (analytical)")
    if 'dG_restr_mdrestraints' in boresch_params:
        print(f"  dG_restraint = {boresch_params['dG_restr_mdrestraints']:.2f} kJ/mol (MDRestraintsGenerator)")
    print()


if __name__ == "__main__":
    main()
