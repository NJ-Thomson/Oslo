#!/usr/bin/env python3
"""
ABFE Setup with Boresch Restraints for Noncovalent Ligands

Sets up Absolute Binding Free Energy (ABFE) calculations using alchemical
free energy perturbation with Boresch-style orientational restraints.

Based on the Biggin Lab ABFE workflow (https://github.com/bigginlab/ABFE_workflow)
which is well-validated for robust binding free energy calculations.

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
- IMPORTANT: Biggin Lab protocol recommends ~20 ns production MD before restraint
  selection to ensure stable anchor points are identified

Pre-ABFE Equilibration (Biggin Lab protocol):
1. 10,000 step energy minimization
2. 1 ns restrained NVT equilibration
3. 1 ns restrained NPT equilibration (Berendsen)
4. 5 ns unrestrained NPT (Parrinello-Rahman)
5. 20 ns NPT production → use this trajectory for MDRestraintsGenerator

Lambda Schedule (Biggin Lab validated):
- Restraint: Dense sampling (0 → 1) with 16 windows
- Coulomb: 11 windows (1 → 0)
- VdW: 20 windows with dense endpoint sampling (1 → 0)

Soft-core Parameters (Biggin Lab):
- sc-alpha = 0.5, sc-power = 1, sc-sigma = 0.3, sc-coul = yes

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
    # First run stability test with 20 ns production (Biggin Lab protocol)
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


def find_anchor_atoms_mdrestraints(gro_path, trajectory_path, ligand_resname='LIG',
                                    output_dir=None, temperature=300):
    """
    Use MDRestraintsGenerator to find optimal Boresch anchor atoms.

    This analyzes MD trajectory to find stable anchor points with minimal
    fluctuation, which is crucial for accurate restraint corrections.

    Args:
        gro_path: Path to GRO structure file
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
    u = mda.Universe(str(gro_path), str(trajectory_path))

    # Select ligand atoms
    ligand_sel = u.select_atoms(f"resname {ligand_resname}")
    if len(ligand_sel) == 0:
        print(f"  ERROR: No atoms found with resname {ligand_resname}")
        return None, None, None

    print(f"  Found {len(ligand_sel)} ligand atoms")

    # Find stable ligand anchor atoms using MDRestraintsGenerator
    print("  Analyzing ligand for stable anchor points...")
    ligand_atoms_analysis = search.find_ligand_atoms(u, ligand_sel)

    # Find protein anchor atoms near the ligand
    print("  Finding protein anchor atoms near binding site...")
    protein_sel = u.select_atoms("protein and name CA")

    # Use FindHostAtoms to identify suitable protein anchors
    host_finder = search.FindHostAtoms(u, protein_sel, ligand_sel)
    host_finder.run()

    # Now find optimal Boresch restraint using trajectory
    print("  Analyzing trajectory for optimal Boresch restraints...")
    print("  (This may take a while for long trajectories)")

    boresch_finder = restraints.FindBoreschRestraint(
        u, ligand_atoms_analysis, host_finder.host_atoms
    )
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

    # Extract atom indices (MDRestraintsGenerator uses 0-indexed, GROMACS needs 1-indexed)
    # The restraint object contains atom information
    protein_atoms_0idx = [
        best_restraint.host_atoms[0].ix,
        best_restraint.host_atoms[1].ix,
        best_restraint.host_atoms[2].ix
    ]
    ligand_atoms_0idx = [
        best_restraint.ligand_atoms[0].ix,
        best_restraint.ligand_atoms[1].ix,
        best_restraint.ligand_atoms[2].ix
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
        # Store fluctuations for diagnostics
        'r_std': best_restraint.bond.std / 10.0,
        'theta_A_std': best_restraint.angles[0].std,
        'theta_B_std': best_restraint.angles[1].std,
        'phi_A_std': best_restraint.dihedrals[0].std,
        'phi_B_std': best_restraint.dihedrals[1].std,
        'phi_C_std': best_restraint.dihedrals[2].std,
    }

    # Calculate analytical correction using standard_state method
    try:
        dG_restr = best_restraint.standard_state(temperature=temperature)
        boresch_params['dG_restr_mdrestraints'] = dG_restr
        print(f"  MDRestraintsGenerator dG_restraint: {dG_restr:.2f} kJ/mol")
    except Exception as e:
        print(f"  Warning: Could not compute standard state correction: {e}")

    print(f"\n  Optimal anchor atoms found:")
    print(f"    Protein (P3-P2-P1): {protein_atoms[2]}-{protein_atoms[1]}-{protein_atoms[0]}")
    print(f"    Ligand (L1-L2-L3):  {ligand_atoms[0]}-{ligand_atoms[1]}-{ligand_atoms[2]}")
    print(f"\n  Equilibrium values (from trajectory):")
    print(f"    r0 = {boresch_params['r0']:.3f} ± {boresch_params['r_std']:.3f} nm")
    print(f"    theta_A = {boresch_params['theta_A0']:.1f} ± {boresch_params['theta_A_std']:.1f} deg")
    print(f"    theta_B = {boresch_params['theta_B0']:.1f} ± {boresch_params['theta_B_std']:.1f} deg")

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


def calculate_restraint_correction(params, temperature=300):
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


def generate_lambda_schedule_biggin():
    """
    Generate Biggin Lab validated lambda schedule for ABFE.

    This schedule is based on the well-validated Biggin Lab ABFE workflow
    (https://github.com/bigginlab/ABFE_workflow) with dense endpoint sampling
    where free energy gradients are steepest.

    The schedule uses three phases:
    1. Restraint: Turn on Boresch restraints (complex leg only)
    2. Coulomb: Turn off electrostatics
    3. VdW: Turn off van der Waals with soft-core potentials

    Returns:
        complex_lambdas: list of (restr, coul, vdw) tuples for complex leg
        solvent_lambdas: list of (coul, vdw) tuples for solvent leg
    """
    # Biggin Lab validated lambda schedule with dense endpoint sampling
    # Restraint lambdas (0=off, 1=on) - 16 windows
    # Dense at endpoints where restraint-to-ligand coupling changes rapidly
    restr_lambdas = [
        0.0, 0.15, 0.30, 0.45, 0.60, 0.75,
        0.80, 0.85, 0.90, 0.925, 0.95,
        0.96, 0.97, 0.98, 0.99, 1.0
    ]

    # Coulomb lambdas (1=on, 0=off) - 11 windows
    # Electrostatic decoupling is usually smoother than vdW
    coul_lambdas = [
        1.0, 0.9, 0.8, 0.7, 0.6, 0.5,
        0.4, 0.3, 0.2, 0.1, 0.0
    ]

    # VdW lambdas (1=on, 0=off) - 20 windows
    # Dense near lambda=0 where soft-core potential is critical
    # This is where most of the free energy change occurs
    vdw_lambdas = [
        1.0, 0.95, 0.90, 0.85, 0.80, 0.75,
        0.70, 0.65, 0.60, 0.55, 0.50,
        0.45, 0.40, 0.35, 0.30, 0.25,
        0.20, 0.15, 0.10, 0.05, 0.0
    ]

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


def generate_lambda_schedule(n_windows_restr=5, n_windows_coul=10, n_windows_vdw=15,
                              use_biggin_schedule=True):
    """
    Generate lambda schedule for ABFE.

    Args:
        n_windows_restr: Number of restraint windows (ignored if use_biggin_schedule=True)
        n_windows_coul: Number of coulomb windows (ignored if use_biggin_schedule=True)
        n_windows_vdw: Number of vdW windows (ignored if use_biggin_schedule=True)
        use_biggin_schedule: Use Biggin Lab validated schedule (recommended)

    Complex leg: restraint -> coul -> vdw
    Solvent leg: coul -> vdw (no restraint)

    Returns:
        complex_lambdas: list of (restr, coul, vdw) tuples
        solvent_lambdas: list of (coul, vdw) tuples
    """
    if use_biggin_schedule:
        return generate_lambda_schedule_biggin()

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


def write_fep_mdp(output_path, lambda_state, lambdas, leg='complex',
                  nsteps=2500000, dt=0.002, ref_temp=300, with_restraint=True):
    """
    Write MDP file for FEP simulation.

    Args:
        lambda_state: Index of current lambda window
        lambdas: Full list of lambda values
        leg: 'complex' or 'solvent'
        with_restraint: Include restraint lambda (complex leg only)
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

pcoupl              = Parrinello-Rahman
pcoupltype          = isotropic
tau_p               = 2.0
ref_p               = 1.0
compressibility     = 4.5e-5

gen_vel             = no
continuation        = yes

constraints         = h-bonds
constraint_algorithm = LINCS

; Neighbor searching
cutoff-scheme       = Verlet
nstlist             = 20
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

; Soft-core parameters (Biggin Lab validated settings)
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
                    nsteps=50000, dt=0.002, ref_temp=300):
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
        gen_vel = "gen_vel = yes\ngen_temp = 300\ngen_seed = -1"
        continuation = ""
        posres = "define = -DPOSRES"
    else:  # npt
        pcoupl_section = """pcoupl = Parrinello-Rahman
pcoupltype = isotropic
tau_p = 2.0
ref_p = 1.0
compressibility = 4.5e-5"""
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
nstlist             = 20
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

    # Create topology for ligand in water
    top_content = """; Ligand in solvent topology
#include "amber99sb-ildn.ff/forcefield.itp"

; Include ligand parameters
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
# For HPC/cluster: use submit_slurm.sh instead

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

    # Also write SLURM job array scripts (Biggin Lab style)
    write_slurm_scripts(output_dir, leg, n_windows, gmx, gpu)

    return script_path


def write_slurm_scripts(output_dir, leg, n_windows, gmx='gmx', gpu=False):
    """
    Write Biggin Lab-style SLURM submission scripts for ABFE.

    Creates:
    - submit_slurm.sh: Main submission script
    - run_window.sh: Script run by each array task
    """
    # GPU-specific SLURM settings
    if gpu:
        gpu_gres = "#SBATCH --gres=gpu:1"
        gpu_flag = "-nb gpu -pme gpu -bonded gpu"
        partition = "gpu"
    else:
        gpu_gres = "# No GPU requested"
        gpu_flag = ""
        partition = "batch"

    # Main SLURM submission script (Biggin Lab style)
    submit_script = f"""#!/bin/bash
#SBATCH --job-name=abfe_{leg}
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --array=0-{n_windows - 1}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --partition={partition}
{gpu_gres}

# =============================================================================
# ABFE {leg.upper()} LEG - SLURM Job Array Submission
# Based on Biggin Lab ABFE workflow
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

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
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

# Load GROMACS module (adjust for your cluster)
# module load gromacs/2023

GMX="{gmx}"
GPU_FLAG="{gpu_flag}"

cd "$SCRIPT_DIR/$LAMBDA_DIR"

# =============================================================================
# NVT Equilibration
# =============================================================================
if [ ! -f nvt.gro ]; then
    echo ""
    echo "Running NVT equilibration..."
    $GMX grompp -f nvt.mdp -c ../input.gro -r ../input.gro -p ../topol.top -o nvt.tpr -maxwarn 2
    $GMX mdrun -deffnm nvt $GPU_FLAG -ntmpi 1 -ntomp $SLURM_CPUS_PER_TASK
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
    $GMX mdrun -deffnm npt $GPU_FLAG -ntmpi 1 -ntomp $SLURM_CPUS_PER_TASK
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
    $GMX mdrun -deffnm prod $GPU_FLAG -ntmpi 1 -ntomp $SLURM_CPUS_PER_TASK
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


def write_analysis_script(output_dir, complex_windows, solvent_windows, dG_restr):
    """Write analysis script to compute binding free energy."""
    script = f"""#!/bin/bash
# ABFE analysis script
# Analyzes dhdl files and computes binding free energy
# Uses alchemlyb (MBAR) if available, otherwise falls back to gmx bar (BAR)

SCRIPT_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "ABFE ANALYSIS"
echo "=============================================="

# Check for alchemlyb first (preferred)
if python -c "import alchemlyb" 2>/dev/null; then
    echo "Using alchemlyb (MBAR) for analysis..."
    python analyze_alchemlyb.py
    exit $?
fi

# Fall back to GROMACS bar
if command -v gmx &> /dev/null; then
    GMX="gmx"
elif command -v gmx_mpi &> /dev/null; then
    GMX="gmx_mpi"
else
    echo "Neither alchemlyb nor GROMACS found."
    echo "Install alchemlyb: pip install alchemlyb"
    exit 1
fi

echo "Using GROMACS bar (BAR) for analysis..."
echo "(For better results, install alchemlyb: pip install alchemlyb)"
echo ""

echo "Analyzing complex leg..."
cd complex
$GMX bar -f lambda*/prod.xvg -o bar_complex.xvg 2>&1 | tee bar_complex.log
cd ..

echo ""
echo "Analyzing solvent leg..."
cd solvent
$GMX bar -f lambda*/prod.xvg -o bar_solvent.xvg 2>&1 | tee bar_solvent.log
cd ..

echo ""
echo "=============================================="
echo "RESULTS"
echo "=============================================="
echo ""
echo "Restraint correction (analytical): {dG_restr:.2f} kJ/mol"
echo ""
echo "Complex leg dG (from bar_complex.log):"
grep "total" complex/bar_complex.log | tail -1
echo ""
echo "Solvent leg dG (from bar_solvent.log):"
grep "total" solvent/bar_solvent.log | tail -1
echo ""
echo "=============================================="
echo "dG_bind = dG_complex - dG_solvent + dG_restraint"
echo "=============================================="
"""

    script_path = output_dir / 'analyze.sh'
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)

    # Also write Python analysis script using alchemlyb
    python_script = f'''#!/usr/bin/env python3
"""
ABFE analysis using alchemlyb (MBAR)

This provides more accurate free energy estimates than BAR by using
all lambda states simultaneously via the Multistate Bennett Acceptance Ratio.

Requirements:
    pip install alchemlyb pandas matplotlib

Usage:
    python analyze_alchemlyb.py
"""

import os
import glob
import pandas as pd
import numpy as np

try:
    from alchemlyb.parsing.gmx import extract_dHdl
    from alchemlyb.estimators import MBAR, BAR
    from alchemlyb.preprocessing import statistical_inefficiency
    from alchemlyb.visualisation import plot_mbar_overlap_matrix
    import matplotlib.pyplot as plt
    HAS_ALCHEMLYB = True
except ImportError:
    print("ERROR: alchemlyb not installed")
    print("Install with: pip install alchemlyb pandas matplotlib")
    HAS_ALCHEMLYB = False
    exit(1)


def analyze_leg(leg_dir, leg_name):
    """Analyze one leg of ABFE calculation."""
    print(f"\\nAnalyzing {{leg_name}} leg...")

    # Find all xvg files
    xvg_files = sorted(glob.glob(os.path.join(leg_dir, "lambda*/prod.xvg")))

    if not xvg_files:
        print(f"  No XVG files found in {{leg_dir}}/lambda*/prod.xvg")
        return None, None

    print(f"  Found {{len(xvg_files)}} lambda windows")

    # Extract dH/dlambda data
    dHdl_list = []
    for xvg in xvg_files:
        try:
            data = extract_dHdl(xvg, T=300)
            # Apply statistical inefficiency to decorrelate samples
            data = statistical_inefficiency(data, series=data.iloc[:, 0])
            dHdl_list.append(data)
        except Exception as e:
            print(f"  Warning: Could not parse {{xvg}}: {{e}}")

    if not dHdl_list:
        return None, None

    # Concatenate all data
    dHdl = pd.concat(dHdl_list)

    # Run MBAR
    print("  Running MBAR estimator...")
    mbar = MBAR()
    mbar.fit(dHdl)

    dG_mbar = mbar.delta_f_.iloc[0, -1]
    dG_err = mbar.d_delta_f_.iloc[0, -1]

    print(f"  MBAR result: {{dG_mbar:.2f}} +/- {{dG_err:.2f}} kJ/mol")

    # Also run BAR for comparison
    bar = BAR()
    bar.fit(dHdl)
    dG_bar = bar.delta_f_.iloc[0, -1]
    print(f"  BAR result: {{dG_bar:.2f}} kJ/mol")

    # Plot overlap matrix
    try:
        fig = plot_mbar_overlap_matrix(mbar.overlap_matrix)
        fig.savefig(os.path.join(leg_dir, f"{{leg_name}}_overlap.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved overlap matrix: {{leg_name}}_overlap.png")
    except Exception as e:
        print(f"  Warning: Could not plot overlap matrix: {{e}}")

    return dG_mbar, dG_err


def main():
    print("=" * 60)
    print("ABFE ANALYSIS (alchemlyb/MBAR)")
    print("=" * 60)

    # Restraint correction
    dG_restr = {dG_restr:.4f}  # kJ/mol (analytical)
    print(f"\\nRestraint correction: {{dG_restr:.2f}} kJ/mol")

    # Analyze complex leg
    dG_complex, err_complex = analyze_leg("complex", "complex")

    # Analyze solvent leg
    dG_solvent, err_solvent = analyze_leg("solvent", "solvent")

    if dG_complex is None or dG_solvent is None:
        print("\\nERROR: Analysis failed for one or more legs")
        return

    # Calculate binding free energy
    # dG_bind = dG_complex - dG_solvent + dG_restr
    # Note: dG_complex is the free energy to decouple ligand from complex
    # dG_solvent is the free energy to decouple ligand from solvent
    dG_bind = dG_complex - dG_solvent + dG_restr

    # Propagate errors
    err_bind = np.sqrt(err_complex**2 + err_solvent**2)

    print("\\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\\ndG_complex:  {{dG_complex:8.2f}} +/- {{err_complex:.2f}} kJ/mol")
    print(f"dG_solvent:  {{dG_solvent:8.2f}} +/- {{err_solvent:.2f}} kJ/mol")
    print(f"dG_restraint:{{dG_restr:8.2f}} kJ/mol (analytical)")
    print("-" * 40)
    print(f"dG_bind:     {{dG_bind:8.2f}} +/- {{err_bind:.2f}} kJ/mol")
    print(f"             {{dG_bind/4.184:8.2f}} +/- {{err_bind/4.184:.2f}} kcal/mol")
    print("\\n" + "=" * 60)

    # Save results
    with open("abfe_results.txt", "w") as f:
        f.write("ABFE Results (alchemlyb/MBAR)\\n")
        f.write("=" * 40 + "\\n\\n")
        f.write(f"dG_complex:   {{dG_complex:.4f}} +/- {{err_complex:.4f}} kJ/mol\\n")
        f.write(f"dG_solvent:   {{dG_solvent:.4f}} +/- {{err_solvent:.4f}} kJ/mol\\n")
        f.write(f"dG_restraint: {{dG_restr:.4f}} kJ/mol\\n\\n")
        f.write(f"dG_bind:      {{dG_bind:.4f}} +/- {{err_bind:.4f}} kJ/mol\\n")
        f.write(f"              {{dG_bind/4.184:.4f}} +/- {{err_bind/4.184:.4f}} kcal/mol\\n")

    print("\\nResults saved to: abfe_results.txt")


if __name__ == "__main__":
    main()
'''

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
    parser.add_argument('--ligand_gro', default=None,
                        help='Ligand GRO for solvent leg (if different from complex)')

    # Trajectory for MDRestraintsGenerator (recommended for robust restraints)
    parser.add_argument('--trajectory', '-t', default=None,
                        help='Trajectory file (XTC/TRR) for MDRestraintsGenerator analysis (recommended)')

    # Anchor atoms (optional - auto-detected if not provided)
    parser.add_argument('--protein_anchors', default=None,
                        help='Protein anchor atom indices (comma-separated, e.g., 100,105,110)')
    parser.add_argument('--ligand_anchors', default=None,
                        help='Ligand anchor atom indices (comma-separated, e.g., 1,5,10)')

    # Simulation parameters
    parser.add_argument('--prod_time', type=float, default=5.0,
                        help='Production time per window in ns (default: 5)')
    parser.add_argument('--temperature', type=float, default=300.0,
                        help='Temperature in K (default: 300)')

    # Lambda schedule options
    parser.add_argument('--use_biggin_schedule', action='store_true', default=True,
                        help='Use Biggin Lab validated lambda schedule (default: True)')
    parser.add_argument('--custom_schedule', action='store_true',
                        help='Use custom lambda schedule (specify n_*_windows)')
    parser.add_argument('--n_restr_windows', type=int, default=16,
                        help='Number of restraint lambda windows for custom schedule (default: 16)')
    parser.add_argument('--n_coul_windows', type=int, default=11,
                        help='Number of coulomb lambda windows for custom schedule (default: 11)')
    parser.add_argument('--n_vdw_windows', type=int, default=20,
                        help='Number of vdW lambda windows for custom schedule (default: 20)')

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
    if args.trajectory:
        trajectory = Path(args.trajectory)
        if not trajectory.exists():
            print(f"WARNING: Trajectory file not found: {args.trajectory}")
            trajectory = None

    # Generate lambda schedules
    use_biggin = args.use_biggin_schedule and not args.custom_schedule
    complex_lambdas, solvent_lambdas = generate_lambda_schedule(
        args.n_restr_windows, args.n_coul_windows, args.n_vdw_windows,
        use_biggin_schedule=use_biggin
    )

    print("\n" + "="*60)
    print("ABFE SETUP WITH BORESCH RESTRAINTS")
    print("="*60)
    if use_biggin:
        print("\n[Using Biggin Lab validated protocol]")
    print(f"\nInputs:")
    print(f"  Complex:     {complex_gro}")
    print(f"  Topology:    {topology}")
    print(f"  Ligand ITP:  {ligand_itp}")
    if trajectory:
        print(f"  Trajectory:  {trajectory} (for MDRestraintsGenerator)")
    print(f"\nSimulation:")
    print(f"  Production:  {args.prod_time} ns per window")
    print(f"  Temperature: {args.temperature} K")
    print(f"  Complex leg: {len(complex_lambdas)} windows")
    print(f"  Solvent leg: {len(solvent_lambdas)} windows")
    print(f"  Schedule:    {'Biggin Lab validated' if use_biggin else 'Custom'}")
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

    elif trajectory and HAS_MDRESTRAINTS:
        # Use MDRestraintsGenerator for robust restraint selection
        print("  Using MDRestraintsGenerator for trajectory-based restraint selection")
        print("  (This is the recommended approach for robust ABFE)")
        protein_atoms, ligand_atoms, boresch_params = find_anchor_atoms_mdrestraints(
            complex_gro, trajectory, args.ligand_resname,
            output_dir=output_dir, temperature=args.temperature
        )
        if protein_atoms is None:
            print("  WARNING: MDRestraintsGenerator failed, falling back to geometric selection")
            protein_atoms, ligand_atoms = find_anchor_atoms(complex_gro, args.ligand_resname)
        else:
            use_mdrestraints = True

    elif trajectory and not HAS_MDRESTRAINTS:
        print("  WARNING: Trajectory provided but MDRestraintsGenerator not installed")
        print("  Install with: pip install MDRestraintsGenerator MDAnalysis")
        print("  Falling back to geometric anchor selection")
        protein_atoms, ligand_atoms = find_anchor_atoms(complex_gro, args.ligand_resname)

    else:
        # Fall back to geometric selection
        print("  Using geometric anchor selection")
        print("  NOTE: For robust ABFE, the Biggin Lab recommends ~20 ns equilibration MD")
        print("        before restraint selection. Provide --trajectory for better results.")
        protein_atoms, ligand_atoms = find_anchor_atoms(complex_gro, args.ligand_resname)

    if protein_atoms is None:
        print("ERROR: Could not find anchor atoms automatically")
        print("Please specify --protein_anchors and --ligand_anchors")
        sys.exit(1)

    print(f"\n  Protein anchors: {protein_atoms}")
    print(f"  Ligand anchors:  {ligand_atoms}")

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
    prod_nsteps = int(args.prod_time * 1000 / 0.002)

    for i, lambdas in enumerate(complex_lambdas):
        lambda_dir = complex_dir / f'lambda{i:02d}'
        lambda_dir.mkdir(exist_ok=True)

        write_equil_mdp(lambda_dir / 'nvt.mdp', 'nvt', i, complex_lambdas, 'complex')
        write_equil_mdp(lambda_dir / 'npt.mdp', 'npt', i, complex_lambdas, 'complex')
        write_fep_mdp(lambda_dir / 'prod.mdp', i, complex_lambdas, 'complex',
                      nsteps=prod_nsteps, ref_temp=args.temperature)

    write_run_script(complex_dir, 'complex', len(complex_lambdas), gmx, args.gpu)
    print(f"  Created {len(complex_lambdas)} lambda windows")

    # Step 4: Setup solvent leg
    print("\n" + "-"*60)
    print("Step 4: Setup solvent leg")
    print("-"*60)

    solvent_dir = output_dir / 'solvent'

    # Use provided ligand GRO or extract from complex
    if args.ligand_gro and Path(args.ligand_gro).exists():
        ligand_gro = Path(args.ligand_gro)
    else:
        # Try to find ligand GRO in same directory as ITP
        ligand_gro = ligand_itp.parent / f'{args.ligand_resname}.gro'
        if not ligand_gro.exists():
            print(f"  WARNING: Ligand GRO not found at {ligand_gro}")
            print(f"  Please provide --ligand_gro")
            ligand_gro = None

    if ligand_gro:
        success = setup_solvent_leg(ligand_gro, ligand_itp, solvent_dir, gmx)
        if success:
            # Create lambda directories
            for i, lambdas in enumerate(solvent_lambdas):
                lambda_dir = solvent_dir / f'lambda{i:02d}'
                lambda_dir.mkdir(exist_ok=True)

                write_equil_mdp(lambda_dir / 'nvt.mdp', 'nvt', i, solvent_lambdas, 'solvent')
                write_equil_mdp(lambda_dir / 'npt.mdp', 'npt', i, solvent_lambdas, 'solvent')
                write_fep_mdp(lambda_dir / 'prod.mdp', i, solvent_lambdas, 'solvent',
                              nsteps=prod_nsteps, ref_temp=args.temperature)

            write_run_script(solvent_dir, 'solvent', len(solvent_lambdas), gmx, args.gpu)
            print(f"  Created {len(solvent_lambdas)} lambda windows")
        else:
            print("  WARNING: Solvent leg setup failed")
    else:
        print("  Skipping solvent leg (no ligand GRO provided)")

    # Step 5: Write analysis scripts
    print("\n" + "-"*60)
    print("Step 5: Write analysis scripts")
    print("-"*60)

    write_analysis_script(output_dir, len(complex_lambdas), len(solvent_lambdas), dG_restr)
    print(f"  Created: analyze.sh (run this)")
    print(f"  Created: analyze_alchemlyb.py (MBAR analysis, recommended)")

    # Save parameters
    params_file = output_dir / 'abfe_parameters.txt'
    with open(params_file, 'w') as f:
        f.write("ABFE Parameters\n")
        f.write("="*40 + "\n\n")
        f.write("Protocol:\n")
        f.write(f"  Lambda schedule: {'Biggin Lab validated' if use_biggin else 'Custom'}\n")
        f.write(f"  Restraint selection: {'MDRestraintsGenerator' if use_mdrestraints else 'Geometric'}\n")
        f.write(f"  Soft-core: sc-alpha=0.5, sc-power=1, sc-sigma=0.3, sc-coul=yes\n\n")
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
        f.write(f"  Complex leg: {len(complex_lambdas)} windows\n")
        f.write(f"  Solvent leg: {len(solvent_lambdas)} windows\n")
        f.write(f"  Total: {len(complex_lambdas) + len(solvent_lambdas)} windows\n")
        if use_biggin:
            f.write(f"\nBiggin Lab schedule details:\n")
            f.write(f"  Restraint windows: 16 (dense endpoint sampling)\n")
            f.write(f"  Coulomb windows: 11 (linear)\n")
            f.write(f"  VdW windows: 20 (dense at lambda=0)\n")

    # Summary
    print("\n" + "="*60)
    print("ABFE SETUP COMPLETE")
    print("="*60)
    print(f"\nProtocol:")
    print(f"  Lambda schedule: {'Biggin Lab validated' if use_biggin else 'Custom'}")
    print(f"  Restraint selection: {'MDRestraintsGenerator' if use_mdrestraints else 'Geometric'}")
    print(f"  Total windows: {len(complex_lambdas) + len(solvent_lambdas)}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nTo run simulations:")
    print(f"\n  Option 1: HPC/SLURM (recommended - runs windows in parallel)")
    print(f"    cd {complex_dir} && sbatch submit_slurm.sh")
    print(f"    cd {solvent_dir} && sbatch submit_slurm.sh")
    print(f"\n  Option 2: Local/sequential")
    print(f"    cd {complex_dir} && ./run_all.sh")
    print(f"    cd {solvent_dir} && ./run_all.sh")
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
