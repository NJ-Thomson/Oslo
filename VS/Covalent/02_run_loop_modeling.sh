#!/bin/bash
#
# run_loop_modeling.sh
#
# Batch loop modeling for multiple pre-prepared PDB structures.
#
# Usage:
#   ./run_loop_modeling.sh 4CXA_prepared.pdb 5ACB_prepared.pdb 5EFQ_prepared.pdb 7NXJ_prepared.pdb
#
# Or run interactively for a single structure:
#   ./run_loop_modeling.sh 4CXA_prepared.pdb
#
# Requirements:
#   - Python 3
#   - MODELLER (with license configured)
#   - requests module (pip install requests)
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}              Batch MODELLER Loop Modeling                           ${NC}"
echo -e "${BLUE}======================================================================${NC}"

if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No PDB files provided${NC}"
    echo "Usage: $0 <pdb_file1> [pdb_file2] ..."
    echo "Example: $0 4CXA_prepared.pdb 5ACB_prepared.pdb"
    exit 1
fi

# Check for required scripts
if [ ! -f "02_prepare_modeller_local.py" ]; then
    echo -e "${RED}Error: prepare_modeller_local.py not found${NC}"
    exit 1
fi

if [ ! -f "02_model_loops.py" ]; then
    echo -e "${RED}Error: model_loops.py not found${NC}"
    exit 1
fi

# Process each PDB file
for PDB_FILE in "$@"; do
    echo -e "\n${BLUE}======================================================================${NC}"
    echo -e "${GREEN}Processing: ${PDB_FILE}${NC}"
    echo -e "${BLUE}======================================================================${NC}"
    
    if [ ! -f "$PDB_FILE" ]; then
        echo -e "${RED}Error: File not found: ${PDB_FILE}${NC}"
        continue
    fi
    
    # Extract PDB ID from filename (e.g., 4CXA from 4CXA_prepared.pdb)
    PDB_ID=$(basename "$PDB_FILE" | sed 's/_prepared.pdb//' | sed 's/.pdb//' | tr '[:lower:]' '[:upper:]')
    
    echo -e "${YELLOW}PDB ID: ${PDB_ID}${NC}"
    
    # Step 1: Prepare MODELLER inputs
    echo -e "\n${BLUE}[Step 1/3] Preparing MODELLER inputs...${NC}"
    python prepare_modeller_local.py "$PDB_FILE"
    
    # Check if preparation was successful
    if [ ! -f "${PDB_ID}_loop.ali" ]; then
        echo -e "${RED}Error: Alignment file not created for ${PDB_ID}${NC}"
        continue
    fi
    
    # Step 2: Run MODELLER
    echo -e "\n${BLUE}[Step 2/3] Running MODELLER loop modeling...${NC}"
    echo -e "${YELLOW}This may take several minutes...${NC}"
    python model_loops.py "$PDB_ID"
    
    # Step 3: Organize outputs
    echo -e "\n${BLUE}[Step 3/3] Organizing outputs...${NC}"
    
    OUTPUT_DIR="${PDB_ID}_modeller_results"
    mkdir -p "$OUTPUT_DIR"
    
    # Move files
    mv -f "${PDB_ID}."* "$OUTPUT_DIR/" 2>/dev/null || true
    mv -f "${PDB_ID}_"* "$OUTPUT_DIR/" 2>/dev/null || true
    mv -f ${PDB_ID}_fill.B*.pdb "$OUTPUT_DIR/" 2>/dev/null || true
    mv -f ${PDB_ID}_fill.V* "$OUTPUT_DIR/" 2>/dev/null || true
    mv -f ${PDB_ID}_fill.D* "$OUTPUT_DIR/" 2>/dev/null || true
    mv -f ${PDB_ID}_fill.ini "$OUTPUT_DIR/" 2>/dev/null || true
    mv -f ${PDB_ID}_fill.rsr "$OUTPUT_DIR/" 2>/dev/null || true
    mv -f ${PDB_ID}_fill.sch "$OUTPUT_DIR/" 2>/dev/null || true
    
    echo -e "${GREEN}Output saved to: ${OUTPUT_DIR}/${NC}"
    
    # List models
    echo -e "\n${YELLOW}Generated models:${NC}"
    ls -la "$OUTPUT_DIR"/${PDB_ID}_fill.B*.pdb 2>/dev/null || echo "  No models found"
done

echo -e "\n${BLUE}======================================================================${NC}"
echo -e "${GREEN}BATCH PROCESSING COMPLETE${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo -e "\nResults directories:"
for PDB_FILE in "$@"; do
    PDB_ID=$(basename "$PDB_FILE" | sed 's/_prepared.pdb//' | sed 's/.pdb//' | tr '[:lower:]' '[:upper:]')
    if [ -d "${PDB_ID}_modeller_results" ]; then
        echo "  ${PDB_ID}_modeller_results/"
    fi
done
echo -e "\n${GREEN}Next steps:${NC}"
echo "  1. Compare models in each *_modeller_results/ directory"
echo "  2. Select models with lowest DOPE scores"
echo "  3. Proceed to ligand docking"
