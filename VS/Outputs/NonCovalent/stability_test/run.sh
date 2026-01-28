#!/bin/bash
# Stability test workflow for noncovalent complex
# Runs: NVT (100ps) -> NPT (1ns) -> Production (configurable)
# Run this script from within the stability test directory

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

GMX="gmx"

echo "=============================================="
echo "BINDING STABILITY TEST"
echo "=============================================="
echo "Working directory: $SCRIPT_DIR"
echo ""

# Step 1: NVT equilibration
echo "Step 1: NVT equilibration (100 ps)..."
$GMX grompp -f nvt.mdp -c input.gro -r input.gro -p topol.top -o nvt.tpr -maxwarn 2
$GMX mdrun -v -deffnm nvt 
echo "NVT complete."
echo ""

# Step 2: NPT equilibration
echo "Step 2: NPT equilibration (1 ns)..."
$GMX grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 2
$GMX mdrun -v -deffnm npt 
echo "NPT complete."
echo ""

# Step 3: Production MD
echo "Step 3: Production MD..."
$GMX grompp -f prod.mdp -c npt.gro -t npt.cpt -p topol.top -o prod.tpr -maxwarn 2
$GMX mdrun -v -deffnm prod 
echo "Production complete."
echo ""

echo "=============================================="
echo "SIMULATION COMPLETE"
echo "=============================================="
echo "Output files:"
echo "  prod.gro  - Final structure"
echo "  prod.xtc  - Trajectory"
echo "  prod.edr  - Energy file"
echo ""
echo "Run analysis script to assess binding stability:"
echo "  ./analyze.sh"
