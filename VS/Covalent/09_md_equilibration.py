#!/usr/bin/env python3
"""
System Stability Assessment for Covalent Complexes

Runs a 20ns equilibration simulation to assess stability of the covalent
complex before proceeding to RBFE calculations. This is essential to ensure
the system is well-equilibrated and the covalent bond is stable.

Workflow:
1. Energy minimization (if not already done)
2. NVT equilibration (100ps, position restraints)
3. NPT equilibration (1ns, position restraints)
4. Production MD (20ns, no restraints)
5. Stability analysis (RMSD, RMSF, energy)

Usage:
    python test_system_stability.py --md_dir <path> --output_dir <path> [options]

Example:
    python test_system_stability.py \\
        --md_dir Outputs/Covalent/Inhib_32_acry/md_simulation \\
        --output_dir Outputs/Covalent/Inhib_32_acry/stability_test \\
        --prod_time 20  # 20 ns
"""

import argparse
import os
import sys
import shutil
import subprocess
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


def write_nvt_mdp(output_path, nsteps=50000, dt=0.002, ref_temp=300):
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


def write_npt_mdp(output_path, nsteps=500000, dt=0.002, ref_temp=300, ref_p=1.0):
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


def write_prod_mdp(output_path, nsteps=10000000, dt=0.002, ref_temp=300, ref_p=1.0):
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

; No position restraints
; define            = -DPOSRES
"""
    with open(output_path, 'w') as f:
        f.write(content)
    return output_path


def write_analysis_script(output_dir, gmx='gmx'):
    """Write bash script for post-simulation analysis."""
    script = f"""#!/bin/bash
# Stability analysis script
# Run after production MD completes
# Run this script from within the stability test directory

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Stability Analysis for Covalent Complex"
echo "=========================================="

# Check if production completed
if [ ! -f prod.gro ]; then
    echo "ERROR: prod.gro not found. Run production MD first."
    exit 1
fi

# 1. RMSD of protein backbone
echo ""
echo "[1] Calculating backbone RMSD..."
echo "Backbone Backbone" | {gmx} rms -s prod.tpr -f prod.xtc -o rmsd_backbone.xvg -tu ns

# 2. RMSD of ligand (covalent residue)
echo ""
echo "[2] Calculating ligand RMSD..."
# First create index for the covalent residue
echo "q" | {gmx} make_ndx -f prod.tpr -o ligand.ndx 2>/dev/null
# Use group 1 (Protein) as reference, try to select ligand manually
echo "Protein Protein" | {gmx} rms -s prod.tpr -f prod.xtc -o rmsd_protein.xvg -tu ns

# 3. RMSF (per-residue fluctuations)
echo ""
echo "[3] Calculating per-residue RMSF..."
echo "Backbone" | {gmx} rmsf -s prod.tpr -f prod.xtc -o rmsf_backbone.xvg -res

# 4. Radius of gyration
echo ""
echo "[4] Calculating radius of gyration..."
echo "Protein" | {gmx} gyrate -s prod.tpr -f prod.xtc -o gyrate.xvg

# 5. Potential energy
echo ""
echo "[5] Extracting potential energy..."
echo "Potential" | {gmx} energy -f prod.edr -o energy_potential.xvg

# 6. Temperature
echo ""
echo "[6] Extracting temperature..."
echo "Temperature" | {gmx} energy -f prod.edr -o energy_temperature.xvg

# 7. Density (box volume)
echo ""
echo "[7] Extracting density..."
echo "Density" | {gmx} energy -f prod.edr -o energy_density.xvg

# 8. Check covalent bond distance (SG-C1)
echo ""
echo "[8] Measuring covalent bond distance..."
# Create index for SG and C1 atoms
cat > bond_index.txt << 'EOF'
[ SG_C1 ]
; Add atom indices for SG and C1 here
EOF
echo "Note: Edit bond_index.txt with correct atom indices for SG-C1 bond measurement"

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - rmsd_backbone.xvg  : Backbone RMSD vs time"
echo "  - rmsd_protein.xvg   : Protein RMSD vs time"
echo "  - rmsf_backbone.xvg  : Per-residue RMSF"
echo "  - gyrate.xvg         : Radius of gyration"
echo "  - energy_*.xvg       : Energy components"
echo ""
echo "Stability criteria (typical values):"
echo "  - Backbone RMSD: should plateau < 3 Å"
echo "  - RMSF: flexible loops may show higher values"
echo "  - Rg: should be stable (no unfolding)"
echo "  - Energy: should fluctuate around stable mean"
"""
    script_path = output_dir / 'analyze_stability.sh'
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    return script_path


def write_run_script(output_dir, gmx='gmx', gpu=False):
    """Write bash script to run the full stability workflow."""
    gpu_flag = '-nb gpu -pme gpu -bonded gpu' if gpu else ''

    script = f"""#!/bin/bash
# Stability test workflow for covalent complex
# Runs: EM -> NVT (100ps) -> NPT (1ns) -> Production (20ns)
# Run this script from within the stability test directory

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
cd "$SCRIPT_DIR"

GMX="{gmx}"

echo "=========================================="
echo "Covalent Complex Stability Assessment"
echo "=========================================="
echo "Working directory: $(pwd)"
echo ""

# Check for required input files
if [ ! -f em.gro ]; then
    echo "ERROR: em.gro not found. Run energy minimization first."
    exit 1
fi

if [ ! -f topol.top ]; then
    echo "ERROR: topol.top not found."
    exit 1
fi

# Step 1: NVT equilibration (100ps)
echo ""
echo "[Step 1/3] NVT Equilibration (100ps)..."
echo "----------------------------------------"
$GMX grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 2
$GMX mdrun -deffnm nvt -v {gpu_flag}

# Step 2: NPT equilibration (1ns)
echo ""
echo "[Step 2/3] NPT Equilibration (1ns)..."
echo "----------------------------------------"
$GMX grompp -f npt.mdp -c nvt.gro -r nvt.gro -p topol.top -t nvt.cpt -o npt.tpr -maxwarn 2
$GMX mdrun -deffnm npt -v {gpu_flag}

# Step 3: Production MD (20ns)
echo ""
echo "[Step 3/3] Production MD (20ns)..."
echo "----------------------------------------"
$GMX grompp -f prod.mdp -c npt.gro -p topol.top -t npt.cpt -o prod.tpr -maxwarn 2
$GMX mdrun -deffnm prod -v {gpu_flag}

echo ""
echo "=========================================="
echo "Stability simulation complete!"
echo "=========================================="
echo ""
echo "Run analysis with: ./analyze_stability.sh"
"""
    script_path = output_dir / 'run_stability.sh'
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)

    # Also write SLURM submission script
    write_slurm_script(output_dir, gmx, gpu)

    return script_path


def write_slurm_script(output_dir, gmx='gmx', gpu=False):
    """
    Write SLURM submission script for stability test.

    Creates:
    - submit_slurm.sh: SLURM job submission script
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

    submit_script = f"""#!/bin/bash
#SBATCH --job-name=stability
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --partition={partition}
{gpu_gres}

# =============================================================================
# COVALENT COMPLEX STABILITY TEST - SLURM Job Submission
# =============================================================================
#
# Usage:
#   sbatch submit_slurm.sh
#
# This runs the full stability workflow:
#   1. NVT equilibration (100 ps)
#   2. NPT equilibration (1 ns)
#   3. Production MD (20 ns default)
#
# Monitor progress:
#   squeue -u $USER
#   tail -f slurm_<jobid>.out
#
# After completion, analyze with:
#   ./analyze_stability.sh
#
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "COVALENT COMPLEX STABILITY TEST"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Working directory: $SCRIPT_DIR"
echo "Start time: $(date)"
echo "=============================================="

# Load GROMACS module (adjust for your cluster)
# module load gromacs/2023

GMX="{gmx}"
GPU_FLAG="{gpu_flag}"

# Check for required input files
if [ ! -f em.gro ]; then
    echo "ERROR: em.gro not found. Run energy minimization first."
    exit 1
fi

if [ ! -f topol.top ]; then
    echo "ERROR: topol.top not found."
    exit 1
fi

# =============================================================================
# NVT Equilibration
# =============================================================================
if [ ! -f nvt.gro ]; then
    echo ""
    echo "[Step 1/3] NVT Equilibration..."
    echo "----------------------------------------"
    $GMX grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 2
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
    echo "[Step 2/3] NPT Equilibration..."
    echo "----------------------------------------"
    $GMX grompp -f npt.mdp -c nvt.gro -r nvt.gro -p topol.top -t nvt.cpt -o npt.tpr -maxwarn 2
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
    echo "[Step 3/3] Production MD..."
    echo "----------------------------------------"
    $GMX grompp -f prod.mdp -c npt.gro -p topol.top -t npt.cpt -o prod.tpr -maxwarn 2
    $GMX mdrun -deffnm prod $GPU_FLAG -ntmpi 1 -ntomp $SLURM_CPUS_PER_TASK
    echo "Production complete"
else
    echo "Production already complete, skipping..."
fi

echo ""
echo "=============================================="
echo "STABILITY TEST COMPLETE"
echo "End time: $(date)"
echo "=============================================="
echo ""
echo "Run analysis with: ./analyze_stability.sh"
"""

    submit_path = output_dir / 'submit_slurm.sh'
    with open(submit_path, 'w') as f:
        f.write(submit_script)
    os.chmod(submit_path, 0o755)

    return submit_path


def setup_stability_test(md_dir, output_dir, prod_time_ns=20, ref_temp=300, gpu=False):
    """
    Set up stability test simulation from existing MD setup.

    Args:
        md_dir: Path to existing MD simulation directory (with em.gro, topol.top)
        output_dir: Output directory for stability test
        prod_time_ns: Production simulation time in nanoseconds
        ref_temp: Reference temperature (K)
        gpu: Use GPU acceleration
    """
    md_dir = Path(md_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gmx = find_gmx()
    if gmx is None:
        print("ERROR: GROMACS not found in PATH")
        return 1

    print("=" * 60)
    print("Setting up Stability Test for Covalent Complex")
    print("=" * 60)
    print(f"Input MD directory: {md_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Production time: {prod_time_ns} ns")
    print(f"Temperature: {ref_temp} K")
    print(f"GROMACS: {gmx}")
    print()

    # Check for required input files
    required_files = ['em.gro', 'topol.top']
    for f in required_files:
        if not (md_dir / f).exists():
            print(f"ERROR: {f} not found in {md_dir}")
            return 1

    # Copy required files
    print("[1] Copying input files...")
    shutil.copy(md_dir / 'em.gro', output_dir / 'em.gro')
    shutil.copy(md_dir / 'topol.top', output_dir / 'topol.top')

    # Copy force field directory
    ff_dirs = list(md_dir.glob('*.ff'))
    if ff_dirs:
        ff_dir = ff_dirs[0]
        dst_ff = output_dir / ff_dir.name
        if dst_ff.exists():
            shutil.rmtree(dst_ff)
        shutil.copytree(ff_dir, dst_ff)
        print(f"  Copied force field: {ff_dir.name}")

    # Copy position restraint files if present
    for posre in md_dir.glob('posre*.itp'):
        shutil.copy(posre, output_dir / posre.name)
        print(f"  Copied: {posre.name}")

    # Copy any included itp files
    for itp in md_dir.glob('*.itp'):
        if not (output_dir / itp.name).exists():
            shutil.copy(itp, output_dir / itp.name)
            print(f"  Copied: {itp.name}")

    # Copy residuetypes.dat if present
    if (md_dir / 'residuetypes.dat').exists():
        shutil.copy(md_dir / 'residuetypes.dat', output_dir / 'residuetypes.dat')
        print("  Copied: residuetypes.dat")

    # Calculate steps
    dt = 0.002  # 2 fs timestep
    nvt_steps = 50000      # 100 ps
    npt_steps = 500000     # 1 ns
    prod_steps = int(prod_time_ns * 1000 / dt)  # Convert ns to steps

    # Write MDP files
    print("\n[2] Writing MDP files...")
    write_nvt_mdp(output_dir / 'nvt.mdp', nsteps=nvt_steps, dt=dt, ref_temp=ref_temp)
    print(f"  nvt.mdp: {nvt_steps * dt / 1000:.1f} ns NVT equilibration")

    write_npt_mdp(output_dir / 'npt.mdp', nsteps=npt_steps, dt=dt, ref_temp=ref_temp)
    print(f"  npt.mdp: {npt_steps * dt / 1000:.1f} ns NPT equilibration")

    write_prod_mdp(output_dir / 'prod.mdp', nsteps=prod_steps, dt=dt, ref_temp=ref_temp)
    print(f"  prod.mdp: {prod_steps * dt / 1000:.1f} ns production MD")

    # Write run scripts
    print("\n[3] Writing run scripts...")
    run_script = write_run_script(output_dir, gmx=gmx, gpu=gpu)
    print(f"  {run_script.name}")

    analysis_script = write_analysis_script(output_dir, gmx=gmx)
    print(f"  {analysis_script.name}")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print(f"\nTo run the stability test:")
    print(f"\n  Option 1: HPC/SLURM (recommended)")
    print(f"    cd {output_dir} && sbatch submit_slurm.sh")
    print(f"\n  Option 2: Local/interactive")
    print(f"    cd {output_dir} && ./run_stability.sh")
    print(f"\nAfter completion, analyze with:")
    print(f"  cd {output_dir} && ./analyze_stability.sh")
    print(f"\nTotal simulation time: {(nvt_steps + npt_steps + prod_steps) * dt / 1000:.1f} ns")
    print(f"  - NVT: {nvt_steps * dt / 1000:.1f} ns")
    print(f"  - NPT: {npt_steps * dt / 1000:.1f} ns")
    print(f"  - Production: {prod_steps * dt / 1000:.1f} ns")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Set up stability test for covalent complex',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python test_system_stability.py --md_dir path/to/md_simulation --output_dir path/to/stability

  # Custom production time
  python test_system_stability.py --md_dir path/to/md --output_dir path/to/stability --prod_time 50

  # With GPU acceleration
  python test_system_stability.py --md_dir path/to/md --output_dir path/to/stability --gpu
        """
    )
    parser.add_argument('--md_dir', required=True,
                        help='Path to MD simulation directory (with em.gro, topol.top)')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for stability test')
    parser.add_argument('--prod_time', type=float, default=20,
                        help='Production simulation time in ns (default: 20)')
    parser.add_argument('--temp', type=float, default=300,
                        help='Reference temperature in K (default: 300)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU acceleration')

    args = parser.parse_args()

    return setup_stability_test(
        md_dir=args.md_dir,
        output_dir=args.output_dir,
        prod_time_ns=args.prod_time,
        ref_temp=args.temp,
        gpu=args.gpu
    )


if __name__ == '__main__':
    sys.exit(main())
