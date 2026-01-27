#!/usr/bin/env python3
"""
MODELLER script for loop modeling with preserved residue numbering.

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

# Read the alignment file to get the starting residue number
starting_residue = 1
try:
    with open(ali_file, 'r') as f:
        for line in f:
            if line.startswith('structureX:') or line.startswith('structure:'):
                # Format: structureX:filename:start_res:chain:end_res:chain::::
                parts = line.split(':')
                if len(parts) > 2 and parts[2].strip():
                    starting_residue = int(parts[2].strip())
                    print(f"  Starting residue from alignment: {starting_residue}")
                break
except Exception as e:
    print(f"  Warning: Could not parse starting residue: {e}")

# Custom AutoModel class that preserves residue numbering
class MyModel(AutoModel):
    def special_patches(self, aln):
        # Renumber residues to match original PDB numbering
        # segment_ids should match your chain IDs
        # renumber_residues=starting_residue sets the first residue number
        self.rename_segments(segment_ids=['A'], renumber_residues=[starting_residue])


# Use custom AutoModel to build models with loops filled
a = MyModel(env,
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