#!/bin/bash
#SBATCH --account=project_XXXXXXXXX    # Replace with your project
#SBATCH --partition=standard-g
#SBATCH --time=24:00:00              # Walltime
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8          # 8 GCDs per node
#SBATCH --gpus-per-node=8
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# ============================================================================
# LUMI-G GROMACS MD Simulation Script
# ============================================================================

# Load GROMACS module (adjust version as needed)
module use /appl/local/csc/modulefiles
module load gromacs/2025.4-gpu

# initial files needed are .mdp, .gro, topol.top and index.ndx

TPR="step8_production.tpr"
MDP="step8_production.mdp"
GRO="step7_10.gro"
CPT="step8_production.cpt"
TOP="topol.top"
NDX="index.ndx"

# Create TPR if it doesn't exist
if [[ ! -f "$TPR" ]]; then
    echo "Creating TPR file..."
    gmx_mpi grompp -f "$MDP" -o "$TPR" -c "$GRO" -p "$TOP" -n "$NDX"
fi

# Check if simulation is already complete
LOG="step8_production.log"
dt=$(grep -E "^\s*dt\s*=" "$MDP" | sed 's/.*=\s*//' | sed 's/;.*//' | tr -d ' ')
nsteps=$(grep -E "^\s*nsteps\s*=" "$MDP" | sed 's/.*=\s*//' | sed 's/;.*//' | tr -d ' ')
target_time=$(echo "$dt * $nsteps" | bc -l)

# Run simulation (with checkpoint if available)
if [[ -f "$CPT" ]]; then
    echo "Restarting from checkpoint..."
    gmx_mpi mdrun -s "$TPR" -cpi "$CPT" -append -maxh 23.5 -deffnm step8_production -v
else
    echo "Starting fresh simulation..."
    gmx_mpi mdrun -s "$TPR" -maxh 23.5  -deffnm step8_production -v
fi
