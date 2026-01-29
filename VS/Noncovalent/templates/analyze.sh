#!/bin/bash
# ABFE analysis script
# Analyzes dhdl files and computes binding free energy
# Uses alchemlyb (MBAR) if available, otherwise falls back to gmx bar (BAR)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
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
echo "Restraint correction: {{DG_RESTR}} kJ/mol"
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
