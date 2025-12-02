#!/bin/bash
#
# run_modeller.sh
#
# Automated MODELLER loop modeling workflow.
# This script:
#   1. Prompts for a PDB ID
#   2. Runs prepare_modeller_from_pdb.py to download and prepare inputs
#   3. Runs model_loops.py to generate loop models
#   4. Organizes all outputs into a folder named by the PDB ID
#
# Usage:
#   ./run_modeller.sh
#   ./run_modeller.sh 8WU1   # Can also pass PDB ID as argument
#
# Requirements:
#   - Python 3
#   - MODELLER (with license configured)
#   - requests module (pip install requests)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}              MODELLER Loop Modeling Workflow                        ${NC}"
echo -e "${BLUE}======================================================================${NC}"

# Get PDB ID from argument or prompt
if [ -n "$1" ]; then
    PDB_ID=$(echo "$1" | tr '[:lower:]' '[:upper:]' | sed 's/\.PDB//g' | sed 's/\.CIF//g')
else
    echo -e "\n${YELLOW}Enter PDB ID:${NC} "
    read -r PDB_INPUT
    PDB_ID=$(echo "$PDB_INPUT" | tr '[:lower:]' '[:upper:]' | sed 's/\.PDB//g' | sed 's/\.CIF//g')
fi

if [ -z "$PDB_ID" ]; then
    echo -e "${RED}Error: No PDB ID provided${NC}"
    exit 1
fi

echo -e "\n${GREEN}Processing PDB: ${PDB_ID}${NC}\n"

# Check for required files
if [ ! -f "prepare_modeller_from_pdb.py" ]; then
    echo -e "${RED}Error: prepare_modeller_from_pdb.py not found in current directory${NC}"
    exit 1
fi

# Step 1: Prepare MODELLER inputs
echo -e "${BLUE}[Step 1/3] Preparing MODELLER inputs...${NC}\n"
python prepare_modeller_from_pdb.py "$PDB_ID"

# Check if preparation was successful
if [ ! -f "${PDB_ID}_loop.ali" ]; then
    echo -e "${RED}Error: Alignment file not created. Check the output above for errors.${NC}"
    exit 1
fi

if [ ! -f "${PDB_ID}_for_modeller.pdb" ]; then
    echo -e "${RED}Error: MODELLER PDB file not created. Check the output above for errors.${NC}"
    exit 1
fi

# Step 2: Run MODELLER
echo -e "\n${BLUE}[Step 2/3] Running MODELLER loop modeling...${NC}\n"
echo -e "${YELLOW}This may take several minutes depending on loop sizes...${NC}\n"
python model_loops.py "$PDB_ID"

# Step 3: Organize outputs
echo -e "\n${BLUE}[Step 3/3] Organizing output files...${NC}\n"

# Create output directory
OUTPUT_DIR="${PDB_ID}_modeller_results"
mkdir -p "$OUTPUT_DIR"

# Move all relevant files
echo "Moving files to ${OUTPUT_DIR}/"

# Input files
mv -f "${PDB_ID}.*" "$OUTPUT_DIR/" 2>/dev/null || true
mv -f "${PDB_ID}_chains.fasta" "$OUTPUT_DIR/" 2>/dev/null || true
mv -f "${PDB_ID}_for_modeller.pdb" "$OUTPUT_DIR/" 2>/dev/null || true
mv -f "${PDB_ID}_loop.ali" "$OUTPUT_DIR/" 2>/dev/null || true

# MODELLER output files
mv -f ${PDB_ID}_fill.B*.pdb "$OUTPUT_DIR/" 2>/dev/null || true
mv -f ${PDB_ID}_fill.V* "$OUTPUT_DIR/" 2>/dev/null || true
mv -f ${PDB_ID}_fill.D* "$OUTPUT_DIR/" 2>/dev/null || true
mv -f ${PDB_ID}_fill.ini "$OUTPUT_DIR/" 2>/dev/null || true
mv -f ${PDB_ID}_fill.rsr "$OUTPUT_DIR/" 2>/dev/null || true
mv -f ${PDB_ID}_fill.sch "$OUTPUT_DIR/" 2>/dev/null || true

# Log files
mv -f *.log "$OUTPUT_DIR/" 2>/dev/null || true
mv -f *.ini "$OUTPUT_DIR/" 2>/dev/null || true
mv -f *.rsr "$OUTPUT_DIR/" 2>/dev/null || true
mv -f *.sch "$OUTPUT_DIR/" 2>/dev/null || true

# Find best model (lowest DOPE score)
echo -e "\n${BLUE}======================================================================${NC}"
echo -e "${GREEN}WORKFLOW COMPLETE${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo -e "\nOutput directory: ${YELLOW}${OUTPUT_DIR}/${NC}"
echo -e "\nFiles created:"
ls -la "$OUTPUT_DIR/"

# List the models with their scores if available
echo -e "\n${YELLOW}Model files (select lowest DOPE score):${NC}"
ls -1 "$OUTPUT_DIR"/${PDB_ID}_fill.B*.pdb 2>/dev/null || echo "  No model files found"

echo -e "\n${GREEN}Next steps:${NC}"
echo "  1. Compare models in ${OUTPUT_DIR}/"
echo "  2. Select the model with the lowest DOPE score"
echo "  3. Visually inspect loops in PyMOL/VMD"
echo "  4. Proceed with MD simulation setup"
echo -e "\n${BLUE}======================================================================${NC}"
