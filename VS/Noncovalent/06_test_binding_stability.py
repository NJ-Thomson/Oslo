#!/usr/bin/env python3
"""
Binding Pose Stability Test for Noncovalent Complexes

Runs a short MD simulation to assess if the docked ligand pose is stable
in the binding site before proceeding to ABFE calculations.

Workflow:
1. NVT equilibration (100 ps, position restraints on protein)
2. NPT equilibration (1 ns, position restraints on protein)
3. Production MD (5-20 ns, no restraints)
4. Analysis: RMSD, ligand-protein distance, hydrogen bonds

A stable binding pose should show:
- Ligand RMSD < 2-3 A from initial position
- Consistent protein-ligand contacts
- No ligand dissociation from binding site

Dependencies:
    - GROMACS (gmx)
    - Output from 05_setup_complex.py

Usage:
    python 06_test_binding_stability.py \\
        --complex_gro COMPLEX.gro \\
        --topology TOPOL.top \\
        --output_dir OUTPUT [options]

Examples:
    # Default 10 ns production
    python 06_test_binding_stability.py \\
        --complex_gro Outputs/NonCovalent/complex/complex_em.gro \\
        --topology Outputs/NonCovalent/complex/topol.top \\
        --output_dir Outputs/NonCovalent/stability_test

    # Quick 5 ns test
    python 06_test_binding_stability.py \\
        --complex_gro complex_em.gro \\
        --topology topol.top \\
        --output_dir stability_5ns \\
        --prod_time 5

    # Generate run scripts only (for cluster submission)
    python 06_test_binding_stability.py \\
        --complex_gro complex_em.gro \\
        --topology topol.top \\
        --output_dir stability_test \\
        --scripts_only
"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path


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


def write_nvt_mdp(output_path, nsteps=50000, dt=0.002, ref_temp=298.15):
    """Write NVT equilibration MDP file (100ps with position restraints)."""
    content = f"""; NVT equilibration with position restraints
integrator          = md
dt                  = {dt}
nsteps              = {nsteps}  ; {nsteps * dt / 1000:.1f} ns

; Output control
nstlog              = 1000
nstenergy           = 1000
nstxout-compressed  = 5000

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

; Temperature coupling
tcoupl              = V-rescale
tc-grps             = Protein Non-Protein
tau_t               = 0.1 0.1
ref_t               = {ref_temp} {ref_temp}

; Pressure coupling (off for NVT)
pcoupl              = no

; Velocity generation
gen_vel             = yes
gen_temp            = {ref_temp}
gen_seed            = -1

; Constraints
constraints         = h-bonds
constraint_algorithm = LINCS
lincs_iter          = 1
lincs_order         = 4

; Position restraints (if present in topology)
define              = -DPOSRES
"""
    with open(output_path, 'w') as f:
        f.write(content)
    return output_path


def write_npt_mdp(output_path, nsteps=500000, dt=0.002, ref_temp=298.15, ref_p=1.0):
    """Write NPT equilibration MDP file (1ns with position restraints)."""
    content = f"""; NPT equilibration with position restraints
integrator          = md
dt                  = {dt}
nsteps              = {nsteps}  ; {nsteps * dt / 1000:.1f} ns

; Output control
nstlog              = 1000
nstenergy           = 1000
nstxout-compressed  = 5000

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

; Temperature coupling
tcoupl              = V-rescale
tc-grps             = Protein Non-Protein
tau_t               = 0.1 0.1
ref_t               = {ref_temp} {ref_temp}

; Pressure coupling (C-rescale for stability with position restraints)
pcoupl              = C-rescale
pcoupltype          = isotropic
tau_p               = 5.0
ref_p               = {ref_p}
compressibility     = 4.5e-5
refcoord_scaling    = com

; Velocity generation (continue from NVT)
gen_vel             = no
continuation        = yes

; Constraints
constraints         = h-bonds
constraint_algorithm = LINCS
lincs_iter          = 1
lincs_order         = 4

; Position restraints
define              = -DPOSRES
"""
    with open(output_path, 'w') as f:
        f.write(content)
    return output_path


def write_prod_mdp(output_path, nsteps=5000000, dt=0.002, ref_temp=298.15, ref_p=1.0):
    """Write production MD MDP file (no restraints)."""
    content = f"""; Production MD (no restraints)
integrator          = md
dt                  = {dt}
nsteps              = {nsteps}  ; {nsteps * dt / 1000:.1f} ns

; Output control
nstlog              = 5000
nstenergy           = 5000
nstxout-compressed  = 5000   ; 10 ps between frames
compressed-x-grps   = System

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

; Temperature coupling
tcoupl              = V-rescale
tc-grps             = Protein Non-Protein
tau_t               = 0.1 0.1
ref_t               = {ref_temp} {ref_temp}

; Pressure coupling
pcoupl              = Parrinello-Rahman
pcoupltype          = isotropic
tau_p               = 2.0
ref_p               = {ref_p}
compressibility     = 4.5e-5

; Velocity generation (continue from NPT)
gen_vel             = no
continuation        = yes

; Constraints
constraints         = h-bonds
constraint_algorithm = LINCS
lincs_iter          = 1
lincs_order         = 4
"""
    with open(output_path, 'w') as f:
        f.write(content)
    return output_path


def write_run_script(output_dir, gmx='gmx', gpu=False):
    """Write bash script to run the full stability workflow."""
    gpu_flag = '-nb gpu -pme gpu -bonded gpu' if gpu else ''

    # Get the templates directory (relative to this script)
    script_dir = Path(__file__).parent
    templates_dir = script_dir / 'templates'

    template_path = templates_dir / 'stability_run.sh'
    if template_path.exists():
        with open(template_path, 'r') as f:
            script = f.read()
        script = script.replace('{{GMX}}', gmx)
        script = script.replace('{{GPU_FLAG}}', gpu_flag)
    else:
        print(f"  WARNING: Template not found: {template_path}")
        script = "#!/bin/bash\necho 'Template not found'\n"

    script_path = output_dir / 'run.sh'
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    return script_path


def write_analysis_script(output_dir, gmx='gmx', ligand_resname='LIG'):
    """Write bash script for post-simulation analysis."""
    # Get the templates directory (relative to this script)
    script_dir = Path(__file__).parent
    templates_dir = script_dir / 'templates'

    template_path = templates_dir / 'stability_analyze.sh'
    if template_path.exists():
        with open(template_path, 'r') as f:
            script = f.read()
        script = script.replace('{{GMX}}', gmx)
        script = script.replace('{{LIGAND_RESNAME}}', ligand_resname)
    else:
        print(f"  WARNING: Template not found: {template_path}")
        script = "#!/bin/bash\necho 'Template not found'\n"

    script_path = output_dir / 'analyze.sh'
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    return script_path


def copy_input_files(complex_gro, topology, output_dir):
    """Copy and prepare input files for stability test."""
    # Copy main files
    shutil.copy(complex_gro, output_dir / 'input.gro')

    top_dir = Path(topology).parent.resolve()
    output_dir = Path(output_dir).resolve()

    # Copy ALL .itp files from source directory
    for itp_file in top_dir.glob('*.itp'):
        dst_path = output_dir / itp_file.name
        if itp_file.resolve() != dst_path.resolve():
            shutil.copy(itp_file, dst_path)
            print(f"  Copied: {itp_file.name}")

    # Copy topology and fix any absolute paths to relative
    new_topology_lines = []
    with open(topology) as f:
        for line in f:
            if line.strip().startswith('#include'):
                match = line.strip().split('"')
                if len(match) >= 2:
                    itp_ref = match[1]
                    # Fix absolute paths to just filename
                    if itp_ref.startswith('/'):
                        itp_name = Path(itp_ref).name
                        new_topology_lines.append(f'#include "{itp_name}"\n')
                        continue
            new_topology_lines.append(line)

    with open(output_dir / 'topol.top', 'w') as f:
        f.writelines(new_topology_lines)

    return True


def run_command(cmd, description, cwd=None, input_text=None, verbose=True):
    """Execute a shell command and handle errors."""
    if verbose:
        print(f"  {description}...")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
        input=input_text
    )

    if result.returncode != 0:
        print(f"  ERROR: {description} failed!")
        print(f"  STDERR: {result.stderr}")
        return False, result

    if verbose:
        print(f"  Done")
    return True, result


def main():
    parser = argparse.ArgumentParser(
        description="Test binding pose stability with MD simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument('--complex_gro', '-c', required=True,
                        help='Input complex GRO file (from 05_setup_complex.py)')
    parser.add_argument('--topology', '-p', required=True,
                        help='Topology file (topol.top)')
    parser.add_argument('--output_dir', '-o', required=True,
                        help='Output directory for stability test')

    # Simulation parameters
    parser.add_argument('--prod_time', type=float, default=10.0,
                        help='Production time in ns (default: 10)')
    parser.add_argument('--temperature', type=float, default=298.15,
                        help='Temperature in K (default: 298.15)')
    parser.add_argument('--ligand_resname', default='LIG',
                        help='Ligand residue name (default: LIG)')

    # Execution options
    parser.add_argument('--scripts_only', action='store_true',
                        help='Generate scripts only, do not run simulation')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU acceleration')
    parser.add_argument('--gmx', default=None,
                        help='GROMACS executable (auto-detected if not specified)')

    args = parser.parse_args()

    # Validate inputs
    complex_gro = Path(args.complex_gro)
    topology = Path(args.topology)

    if not complex_gro.exists():
        print(f"ERROR: Complex GRO not found: {args.complex_gro}")
        sys.exit(1)
    if not topology.exists():
        print(f"ERROR: Topology not found: {args.topology}")
        sys.exit(1)

    # Find GROMACS
    gmx = args.gmx or find_gmx()
    if gmx is None:
        print("ERROR: GROMACS not found. Please install GROMACS or specify --gmx")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate nsteps
    dt = 0.002  # ps
    prod_nsteps = int(args.prod_time * 1000 / dt)  # ns to steps

    print("\n" + "="*60)
    print("BINDING STABILITY TEST SETUP")
    print("="*60)
    print(f"\nInputs:")
    print(f"  Complex:     {complex_gro}")
    print(f"  Topology:    {topology}")
    print(f"\nSimulation:")
    print(f"  NVT:         100 ps (position restraints)")
    print(f"  NPT:         1 ns (position restraints)")
    print(f"  Production:  {args.prod_time} ns (no restraints)")
    print(f"  Temperature: {args.temperature} K")
    print(f"\nOutput:")
    print(f"  Directory:   {output_dir}")

    # Step 1: Copy input files
    print("\n" + "-"*60)
    print("Step 1: Preparing input files")
    print("-"*60)

    copy_input_files(complex_gro, topology, output_dir)

    # Step 2: Write MDP files
    print("\n" + "-"*60)
    print("Step 2: Writing MDP files")
    print("-"*60)

    write_nvt_mdp(output_dir / 'nvt.mdp', nsteps=50000, ref_temp=args.temperature)
    print(f"  Created: nvt.mdp")

    write_npt_mdp(output_dir / 'npt.mdp', nsteps=500000, ref_temp=args.temperature)
    print(f"  Created: npt.mdp")

    write_prod_mdp(output_dir / 'prod.mdp', nsteps=prod_nsteps, ref_temp=args.temperature)
    print(f"  Created: prod.mdp")

    # Step 3: Write run scripts
    print("\n" + "-"*60)
    print("Step 3: Writing run scripts")
    print("-"*60)

    write_run_script(output_dir, gmx=gmx, gpu=args.gpu)
    print(f"  Created: run.sh")

    write_analysis_script(output_dir, gmx=gmx, ligand_resname=args.ligand_resname)
    print(f"  Created: analyze.sh")

    # Summary
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)

    if args.scripts_only:
        print(f"\nScripts generated in {output_dir}")
        print(f"\nTo run the simulation:")
        print(f"  cd {output_dir}")
        print(f"  ./run.sh")
        print(f"\nAfter simulation completes:")
        print(f"  ./analyze.sh")
    else:
        print(f"\nTo run the stability test:")
        print(f"  cd {output_dir}")
        print(f"  ./run.sh")
        print(f"\nOr submit to cluster queue as appropriate.")

    print(f"\nAfter confirming stability, proceed to ABFE:")
    print(f"  python 07_setup_abfe.py \\")
    print(f"      --complex_gro {output_dir}/prod.gro \\")
    print(f"      --topology {output_dir}/topol.top \\")
    print(f"      --output_dir OUTPUT_DIR")
    print()


if __name__ == "__main__":
    main()
