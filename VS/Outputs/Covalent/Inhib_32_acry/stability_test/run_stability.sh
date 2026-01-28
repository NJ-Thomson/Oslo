#!/bin/bash
# Stability test workflow for covalent complex
# Runs: EM -> NVT (100ps) -> NPT (1ns) -> Production (20ns)
# Run this script from within the stability test directory

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

GMX="gmx"

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
$GMX mdrun -deffnm nvt -v 

# Step 2: NPT equilibration (1ns)
echo ""
echo "[Step 2/3] NPT Equilibration (1ns)..."
echo "----------------------------------------"
$GMX grompp -f npt.mdp -c nvt.gro -r nvt.gro -p topol.top -t nvt.cpt -o npt.tpr -maxwarn 2
$GMX mdrun -deffnm npt -v 

# Step 3: Production MD (20ns)
echo ""
echo "[Step 3/3] Production MD (20ns)..."
echo "----------------------------------------"
$GMX grompp -f prod.mdp -c npt.gro -p topol.top -t npt.cpt -o prod.tpr -maxwarn 2
$GMX mdrun -deffnm prod -v 

echo ""
echo "=========================================="
echo "Stability simulation complete!"
echo "=========================================="
echo ""
echo "Run analysis with: ./analyze_stability.sh"
