#!/bin/bash

# GROMACS MD Trajectory Post-Processing Script
# Handles: PBC corrections, centering, fitting, and cleaning

set -e  # Exit on error

# Default values
TPR="md.tpr"
XTC="md.xtc"
OUT_PREFIX="md_processed"
CENTER_GROUP="Protein"
OUTPUT_GROUP="System"
START_TIME=0
END_TIME=-1
SKIP=1

usage() {
    cat << EOF
Usage: $(basename $0) [OPTIONS]

Post-process GROMACS MD trajectories (centering, PBC corrections, fitting)

Options:
    -s FILE     Input .tpr file (default: md.tpr)
    -f FILE     Input .xtc file (default: md.xtc)
    -o PREFIX   Output prefix (default: md_processed)
    -c GROUP    Center group (default: Protein)
    -g GROUP    Output group (default: System)
    -b TIME     Start time in ps (default: 0)
    -e TIME     End time in ps (default: -1, meaning all)
    -dt SKIP    Write every SKIP frames (default: 1)
    -h          Show this help message

Example:
    $(basename $0) -s prod.tpr -f prod.xtc -o prod_clean -c Protein -g System
EOF
    exit 1
}

# Parse command line arguments
while getopts "s:f:o:c:g:b:e:d:h" opt; do
    case $opt in
        s) TPR="$OPTARG" ;;
        f) XTC="$OPTARG" ;;
        o) OUT_PREFIX="$OPTARG" ;;
        c) CENTER_GROUP="$OPTARG" ;;
        g) OUTPUT_GROUP="$OPTARG" ;;
        b) START_TIME="$OPTARG" ;;
        e) END_TIME="$OPTARG" ;;
        d) SKIP="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Check input files exist
if [[ ! -f "$TPR" ]]; then
    echo "Error: TPR file '$TPR' not found"
    exit 1
fi

if [[ ! -f "$XTC" ]]; then
    echo "Error: XTC file '$XTC' not found"
    exit 1
fi

echo "=============================================="
echo "GROMACS Trajectory Post-Processing"
echo "=============================================="
echo "Input TPR:      $TPR"
echo "Input XTC:      $XTC"
echo "Output prefix:  $OUT_PREFIX"
echo "Center group:   $CENTER_GROUP"
echo "Output group:   $OUTPUT_GROUP"
echo "=============================================="

# Build time options
TIME_OPTS=""
[[ $START_TIME -ne 0 ]] && TIME_OPTS="$TIME_OPTS -b $START_TIME"
[[ $END_TIME -ne -1 ]] && TIME_OPTS="$TIME_OPTS -e $END_TIME"
[[ $SKIP -ne 1 ]] && TIME_OPTS="$TIME_OPTS -dt $SKIP"

# Create index file if needed (optional, comment out if not required)
# echo "Creating index file..."
# gmx make_ndx -f "$TPR" -o index.ndx << EOF
# q
# EOF

# Step 1: Make molecules whole (fix molecules broken across PBC)
echo ""
echo "Step 1/4: Making molecules whole..."
echo "$OUTPUT_GROUP" | gmx trjconv -s "$TPR" -f "$XTC" \
    -o "${OUT_PREFIX}_whole.xtc" \
    -pbc whole \
    $TIME_OPTS

# Step 2: Center the protein in the box
echo ""
echo "Step 2/4: Centering ${CENTER_GROUP} in box..."
echo -e "${CENTER_GROUP}\n${OUTPUT_GROUP}" | gmx trjconv -s "$TPR" \
    -f "${OUT_PREFIX}_whole.xtc" \
    -o "${OUT_PREFIX}_center.xtc" \
    -center \
    -pbc mol \
    -ur compact

# Step 3: Remove jumps (cluster molecules around the centered group)
echo ""
echo "Step 3/4: Clustering and removing jumps..."
echo -e "${CENTER_GROUP}\n${OUTPUT_GROUP}" | gmx trjconv -s "$TPR" \
    -f "${OUT_PREFIX}_center.xtc" \
    -o "${OUT_PREFIX}_nojump.xtc" \
    -pbc cluster

# Step 4: Fit to reference structure (remove rotation/translation)
echo ""
echo "Step 4/4: Fitting to reference (removing rotation/translation)..."
echo -e "${CENTER_GROUP}\n${OUTPUT_GROUP}" | gmx trjconv -s "$TPR" \
    -f "${OUT_PREFIX}_nojump.xtc" \
    -o "${OUT_PREFIX}_fit.xtc" \
    -fit rot+trans

# Rename final output
mv "${OUT_PREFIX}_fit.xtc" "${OUT_PREFIX}.xtc"

# Clean up intermediate files
echo ""
echo "Cleaning up intermediate files..."
rm -f "${OUT_PREFIX}_whole.xtc" "${OUT_PREFIX}_center.xtc" "${OUT_PREFIX}_nojump.xtc"

# Also process a PDB for visualization (first frame)
echo ""
echo "Extracting first frame as PDB..."
echo -e "${CENTER_GROUP}\n${OUTPUT_GROUP}" | gmx trjconv -s "$TPR" \
    -f "${OUT_PREFIX}.xtc" \
    -o "${OUT_PREFIX}_frame0.pdb" \
    -dump 0

echo ""
echo "=============================================="
echo "Post-processing complete!"
echo "=============================================="
echo "Output files:"
echo "  Trajectory: ${OUT_PREFIX}.xtc"
echo "  First frame: ${OUT_PREFIX}_frame0.pdb"
echo ""
echo "Verify with: gmx check -f ${OUT_PREFIX}.xtc"
echo "=============================================="