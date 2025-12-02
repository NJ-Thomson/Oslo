#!/usr/bin/env python3
"""
MODELLER script for loop modeling.

Usage:
    python model_loops.py <PDB_ID>
    
Example:
    python model_loops.py 8WU1

Output:
    <PDB_ID>_fill.B99990001.pdb (and more models based on ending_model setting)
"""

import sys
from modeller import *
from modeller.automodel import *

# Get PDB ID from command line or prompt
if len(sys.argv) > 1:
    pdb_id = sys.argv[1].strip().upper().replace(".PDB", "").replace(".CIF", "")
else:
    pdb_id = input("Enter PDB ID: ").strip().upper().replace(".PDB", "").replace(".CIF", "")

if not pdb_id:
    print("Error: No PDB ID provided")
    sys.exit(1)

# File names based on PDB ID
ali_file = f"{pdb_id}_loop.ali"
modeller_pdb = f"{pdb_id}_for_modeller.pdb"
template_name = f"{pdb_id}_template"
target_name = f"{pdb_id}_fill"

print(f"Running MODELLER for {pdb_id}...")
print(f"  Alignment file: {ali_file}")
print(f"  Template PDB: {modeller_pdb}")

# Set up environment
env = Environ()
env.io.atom_files_directory = ['.']

# Use AutoModel to build models with loops filled
a = AutoModel(env,
              alnfile=ali_file,
              knowns=template_name,
              sequence=target_name,
              assess_methods=(assess.DOPE, assess.GA341))

# Generate multiple models to sample different loop conformations
a.starting_model = 1
a.ending_model = 5  # Generate 5 models - increase for better sampling

# Optimization settings
a.library_schedule = autosched.slow
a.max_var_iterations = 300

# MD refinement settings
a.md_level = refine.slow

# Build models
a.make()

# Print summary
print("\n" + "="*60)
print("MODEL SUMMARY")
print("="*60)
for model in a.outputs:
    if model['failure'] is None:
        print(f"{model['name']:30s} DOPE: {model['DOPE score']:10.3f}  GA341: {model['GA341 score'][0]:6.3f}")
print("="*60)
print("\nLower DOPE score = better model")
print("Select the model with the lowest DOPE score for your MD simulation")
