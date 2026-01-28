#!/bin/bash
# Stability analysis script
# Run after production MD completes
# Run this script from within the stability test directory

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
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
echo "Backbone Backbone" | gmx rms -s prod.tpr -f prod.xtc -o rmsd_backbone.xvg -tu ns

# 2. RMSD of ligand (covalent residue)
echo ""
echo "[2] Calculating ligand RMSD..."
# First create index for the covalent residue
echo "q" | gmx make_ndx -f prod.tpr -o ligand.ndx 2>/dev/null
# Use group 1 (Protein) as reference, try to select ligand manually
echo "Protein Protein" | gmx rms -s prod.tpr -f prod.xtc -o rmsd_protein.xvg -tu ns

# 3. RMSF (per-residue fluctuations)
echo ""
echo "[3] Calculating per-residue RMSF..."
echo "Backbone" | gmx rmsf -s prod.tpr -f prod.xtc -o rmsf_backbone.xvg -res

# 4. Radius of gyration
echo ""
echo "[4] Calculating radius of gyration..."
echo "Protein" | gmx gyrate -s prod.tpr -f prod.xtc -o gyrate.xvg

# 5. Potential energy
echo ""
echo "[5] Extracting potential energy..."
echo "Potential" | gmx energy -f prod.edr -o energy_potential.xvg

# 6. Temperature
echo ""
echo "[6] Extracting temperature..."
echo "Temperature" | gmx energy -f prod.edr -o energy_temperature.xvg

# 7. Density (box volume)
echo ""
echo "[7] Extracting density..."
echo "Density" | gmx energy -f prod.edr -o energy_density.xvg

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
