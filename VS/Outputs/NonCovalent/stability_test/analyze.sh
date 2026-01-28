#!/bin/bash
# Stability analysis script
# Run after production MD completes
# Run this script from within the stability test directory

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

GMX="gmx"

echo "=============================================="
echo "BINDING STABILITY ANALYSIS"
echo "=============================================="
echo ""

# Create analysis directory
mkdir -p analysis

# 1. Protein backbone RMSD
echo "1. Calculating protein backbone RMSD..."
echo "4 4" | $GMX rms -s prod.tpr -f prod.xtc -o analysis/rmsd_protein.xvg -tu ns
echo ""

# 2. Ligand RMSD (aligned to protein)
echo "2. Calculating ligand RMSD..."
# First create index with ligand
echo "q" | $GMX make_ndx -f prod.tpr -o index.ndx 2>/dev/null || true
# Select ligand and calculate RMSD
echo "4 LIG" | $GMX rms -s prod.tpr -f prod.xtc -n index.ndx -o analysis/rmsd_ligand.xvg -tu ns 2>/dev/null || \
echo "1 13" | $GMX rms -s prod.tpr -f prod.xtc -o analysis/rmsd_ligand.xvg -tu ns
echo ""

# 3. Protein RMSF
echo "3. Calculating protein RMSF..."
echo "4" | $GMX rmsf -s prod.tpr -f prod.xtc -o analysis/rmsf_protein.xvg -res
echo ""

# 4. Potential energy
echo "4. Extracting potential energy..."
echo "Potential" | $GMX energy -f prod.edr -o analysis/energy.xvg
echo ""

# 5. Radius of gyration
echo "5. Calculating radius of gyration..."
echo "1" | $GMX gyrate -s prod.tpr -f prod.xtc -o analysis/gyrate.xvg
echo ""

# 6. Minimum distance between ligand and protein
echo "6. Calculating protein-ligand contacts..."
echo "1 13" | $GMX mindist -s prod.tpr -f prod.xtc -od analysis/mindist.xvg -on analysis/numcont.xvg -d 0.4 2>/dev/null || \
echo "Protein LIG" | $GMX mindist -s prod.tpr -f prod.xtc -n index.ndx -od analysis/mindist.xvg -on analysis/numcont.xvg -d 0.4 2>/dev/null || true
echo ""

echo "=============================================="
echo "ANALYSIS COMPLETE"
echo "=============================================="
echo ""
echo "Results in analysis/ directory:"
echo "  rmsd_protein.xvg  - Protein backbone RMSD"
echo "  rmsd_ligand.xvg   - Ligand RMSD (after protein alignment)"
echo "  rmsf_protein.xvg  - Protein RMSF by residue"
echo "  energy.xvg        - Potential energy"
echo "  gyrate.xvg        - Radius of gyration"
echo "  mindist.xvg       - Minimum protein-ligand distance"
echo "  numcont.xvg       - Number of protein-ligand contacts"
echo ""
echo "STABILITY CRITERIA:"
echo "  - Ligand RMSD should be < 2-3 A from initial position"
echo "  - Protein RMSD should plateau (equilibrated)"
echo "  - Contacts should remain relatively stable"
echo ""
echo "View results with xmgrace:"
echo "  xmgrace analysis/rmsd_ligand.xvg"
