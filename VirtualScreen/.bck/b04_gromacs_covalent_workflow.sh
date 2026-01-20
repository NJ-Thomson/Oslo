#!/bin/bash
# =============================================================================
# GROMACS Workflow for Covalent Complex
# =============================================================================
# 
# This is a step-by-step guide for setting up MD of a covalent complex.
# Run each section manually and check outputs before proceeding.
#
# Prerequisites:
#   - GROMACS installed
#   - acpype installed (pip install acpype)
#   - Your files: covalent_complex.pdb, best_pose.sdf
#
# =============================================================================

# Set your variables
COMPLEX="covalent_complex.pdb"
LIGAND_SDF="best_pose.sdf"
CYS_RESID=1039
WORKDIR="md_covalent"

mkdir -p $WORKDIR
cd $WORKDIR

# =============================================================================
# STEP 1: Extract protein and ligand separately
# =============================================================================
echo "=== Step 1: Extracting protein and ligand ==="

# Extract protein only (ATOM records)
grep "^ATOM" ../$COMPLEX > protein.pdb
echo "END" >> protein.pdb

# Extract ligand only (HETATM records with LIG)
grep "^HETATM.*LIG" ../$COMPLEX > ligand.pdb
echo "END" >> ligand.pdb

echo "Created: protein.pdb, ligand.pdb"

# =============================================================================
# STEP 2: Parameterize ligand with acpype
# =============================================================================
echo ""
echo "=== Step 2: Parameterizing ligand ==="

# Convert SDF to MOL2 if needed, then run acpype
acpype -i ../$LIGAND_SDF -c bcc -n 0 -a gaff2 -o gmx -b LIG

# Check output
ls -la LIG.acpype/

echo ""
echo "Ligand files created in LIG.acpype/"
echo "Key files:"
echo "  - LIG.acpype/LIG_GMX.itp  (topology)"
echo "  - LIG.acpype/LIG_GMX.gro  (coordinates)"

# =============================================================================
# STEP 3: Process protein with pdb2gmx
# =============================================================================
echo ""
echo "=== Step 3: Processing protein ==="

# Use AMBER force field (compatible with GAFF for ligand)
gmx pdb2gmx -f protein.pdb -o protein.gro -p protein.top -i posre.itp \
    -ff amber99sb-ildn -water tip3p -ignh

echo "Created: protein.gro, protein.top"

# =============================================================================
# STEP 4: Find atom indices for covalent bond
# =============================================================================
echo ""
echo "=== Step 4: Finding atom indices ==="

# Find Cys SG in protein.gro
echo "Cysteine SG atom:"
grep -n "CYS.*SG" protein.gro | head -1

# The atom number is in column 3 (after residue info)
SG_INDEX=$(grep "CYS.*SG" protein.gro | head -1 | awk '{print $3}')
echo "SG atom index: $SG_INDEX"
#THIS PRODUCES WRONG SG INDEX, NEED TO SPECIFY BY RESNUM NEIL
# For ligand, find the beta carbon (the one that was attached to placeholder S)
echo ""
echo "Ligand atoms (check LIG.acpype/LIG_GMX.itp for atom types):"
echo "Look for the carbon that should bond to SG"
echo "Typically named C26 or similar based on your structure"

# =============================================================================
# STEP 5: Combine coordinates
# =============================================================================
echo ""
echo "=== Step 5: Combining coordinates ==="

# Get number of atoms in protein
PROT_ATOMS=$(head -2 protein.gro | tail -1)

# Get number of atoms in ligand
LIG_ATOMS=$(head -2 LIG.acpype/LIG_GMX.gro | tail -1)

# Total atoms
TOTAL=$((PROT_ATOMS + LIG_ATOMS))

# Create combined GRO
echo "Covalent Complex" > complex.gro
echo "$TOTAL" >> complex.gro

# Add protein atoms (skip header and box line)
head -n -1 protein.gro | tail -n +3 >> complex.gro

# Add ligand atoms (skip header and box line)
head -n -1 LIG.acpype/LIG_GMX.gro | tail -n +3 >> complex.gro

# Add box from protein
tail -1 protein.gro >> complex.gro

echo "Created: complex.gro with $TOTAL atoms"

# =============================================================================
# STEP 6: Create combined topology
# =============================================================================
echo ""
echo "=== Step 6: Creating topology ==="

# Copy ligand ITP
cp LIG.acpype/LIG_GMX.itp ligand.itp

# Create combined topology
cat > topol.top << 'EOF'
; Covalent complex topology

; Force field
#include "amber99sb-ildn.ff/forcefield.itp"

; Ligand parameters (GAFF2)
#include "ligand.itp"

; Protein topology
#include "protein.top"

; Water
#include "amber99sb-ildn.ff/tip3p.itp"

; Ions
#include "amber99sb-ildn.ff/ions.itp"

[ system ]
Covalent Complex

[ molecules ]
; Compound        nmols
Protein_chain_A   1
LIG               1
EOF

echo "Created: topol.top"
echo ""
echo "IMPORTANT: You need to manually edit topol.top to include:"
echo "  #include for protein topology"
echo "  Check that it compiles with: gmx grompp -f em.mdp -c complex.gro -p topol.top -o test.tpr"

# =============================================================================
# STEP 7: Add intermolecular covalent bond
# =============================================================================
echo ""
echo "=== Step 7: Adding covalent bond ==="

cat >> topol.top << EOF

; =============================================
; COVALENT BOND: Cys SG - Ligand Carbon
; =============================================
[ intermolecular_interactions ]
[ bonds ]
; ai      aj    funct   r0 (nm)    fc (kJ/mol/nm^2)
; Protein_SG  Ligand_C   1     0.1810     238000.0
; EDIT THE LINE BELOW WITH CORRECT ATOM INDICES:
; $SG_INDEX      XXX       1     0.1810     238000.0

EOF

echo ""
echo "MANUAL EDIT REQUIRED:"
echo "  1. Open topol.top"
echo "  2. Find the [ intermolecular_interactions ] section at the bottom"
echo "  3. Uncomment and edit the bond line with correct atom numbers:"
echo "     - First number: SG index from protein ($SG_INDEX)"
echo "     - Second number: Beta carbon index from ligand (check ligand.itp)"
echo ""
echo "  The ligand atom index needs to be offset by protein atoms!"
echo "  If protein has $PROT_ATOMS atoms and ligand beta-C is atom 26,"
echo "  then use: $SG_INDEX  $((PROT_ATOMS + 26))  1  0.1810  238000.0"

# =============================================================================
# STEP 8: Define box and solvate
# =============================================================================
echo ""
echo "=== Step 8: Solvating system ==="

# Define box
gmx editconf -f complex.gro -o complex_box.gro -c -d 1.2 -bt dodecahedron

# Add water
gmx solvate -cp complex_box.gro -cs spc216.gro -o complex_solv.gro -p topol.top

# =============================================================================
# STEP 9: Add ions
# =============================================================================
echo ""
echo "=== Step 9: Adding ions ==="

# Create minimal MDP for ion addition
cat > ions.mdp << 'EOF'
; Minimal MDP for genion
integrator = steep
nsteps = 0
EOF

gmx grompp -f ions.mdp -c complex_solv.gro -p topol.top -o ions.tpr -maxwarn 5

# Add ions (neutralize system)
echo "SOL" | gmx genion -s ions.tpr -o complex_ions.gro -p topol.top \
    -pname NA -nname CL -neutral

echo "Created: complex_ions.gro (solvated and ionized)"

# =============================================================================
# STEP 10: Create MDP files
# =============================================================================
echo ""
echo "=== Step 10: Creating MDP files ==="

# Energy minimization
cat > em.mdp << 'EOF'
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
EOF

# NVT
cat > nvt.mdp << 'EOF'
integrator  = md
nsteps      = 50000
dt          = 0.002
nstxout     = 5000
nstvout     = 5000
nstenergy   = 500
nstlog      = 500
continuation = no
constraint_algorithm = lincs
constraints = h-bonds
lincs_iter  = 1
lincs_order = 4
cutoff-scheme = Verlet
ns_type     = grid
nstlist     = 20
rcoulomb    = 1.0
rvdw        = 1.0
coulombtype = PME
pme_order   = 4
fourierspacing = 0.16
tcoupl      = V-rescale
tc-grps     = Protein_LIG Water_and_ions
tau_t       = 0.1       0.1
ref_t       = 300       300
pcoupl      = no
pbc         = xyz
gen_vel     = yes
gen_temp    = 300
gen_seed    = -1
EOF

# NPT
cat > npt.mdp << 'EOF'
integrator  = md
nsteps      = 50000
dt          = 0.002
nstxout     = 5000
nstvout     = 5000
nstenergy   = 500
nstlog      = 500
continuation = yes
constraint_algorithm = lincs
constraints = h-bonds
lincs_iter  = 1
lincs_order = 4
cutoff-scheme = Verlet
ns_type     = grid
nstlist     = 20
rcoulomb    = 1.0
rvdw        = 1.0
coulombtype = PME
pme_order   = 4
fourierspacing = 0.16
tcoupl      = V-rescale
tc-grps     = Protein_LIG Water_and_ions
tau_t       = 0.1       0.1
ref_t       = 300       300
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5
pbc         = xyz
gen_vel     = no
EOF

# Production (20 ns)
cat > prod.mdp << 'EOF'
integrator  = md
nsteps      = 10000000   ; 20 ns
dt          = 0.002
nstxout     = 0
nstvout     = 0
nstfout     = 0
nstxout-compressed = 5000  ; 10 ps
nstenergy   = 5000
nstlog      = 5000
continuation = yes
constraint_algorithm = lincs
constraints = h-bonds
lincs_iter  = 1
lincs_order = 4
cutoff-scheme = Verlet
ns_type     = grid
nstlist     = 20
rcoulomb    = 1.0
rvdw        = 1.0
coulombtype = PME
pme_order   = 4
fourierspacing = 0.16
tcoupl      = V-rescale
tc-grps     = Protein_LIG Water_and_ions
tau_t       = 0.1       0.1
ref_t       = 300       300
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5
pbc         = xyz
gen_vel     = no
EOF

echo "Created: em.mdp, nvt.mdp, npt.mdp, prod.mdp"

# =============================================================================
# STEP 11: Create index file with Protein_LIG group
# =============================================================================
echo ""
echo "=== Step 11: Creating index groups ==="

# Create index with combined Protein_LIG group
gmx make_ndx -f complex_ions.gro -o index.ndx << 'EOF'
1 | 13
name 19 Protein_LIG
q
EOF

# Note: Group numbers may vary - check output of make_ndx
echo "Created: index.ndx"
echo "VERIFY: Check that Protein_LIG group exists"

# =============================================================================
# STEP 12: Run simulations
# =============================================================================
echo ""
echo "=== Step 12: Running simulations ==="
echo ""
echo "Run these commands in order:"
echo ""
echo "# Energy minimization"
echo "gmx grompp -f em.mdp -c complex_ions.gro -p topol.top -o em.tpr -maxwarn 2"
echo "gmx mdrun -v -deffnm em"
echo ""
echo "# NVT equilibration (100 ps)"
echo "gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -n index.ndx -o nvt.tpr -maxwarn 2"
echo "gmx mdrun -v -deffnm nvt"
echo ""
echo "# NPT equilibration (100 ps)"
echo "gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -n index.ndx -o npt.tpr -maxwarn 2"
echo "gmx mdrun -v -deffnm npt"
echo ""
echo "# Production (20 ns)"
echo "gmx grompp -f prod.mdp -c npt.gro -t npt.cpt -p topol.top -n index.ndx -o prod.tpr -maxwarn 2"
echo "gmx mdrun -v -deffnm prod"
echo ""
echo "=== IMPORTANT NOTES ==="
echo "1. EDIT topol.top to add the covalent bond (see Step 7)"
echo "2. Check that tc-grps in MDP files match your index groups"
echo "3. If grompp fails, check -maxwarn output carefully"
echo "4. For HPC: use 'gmx mdrun -nt X -gpu_id 0' for GPU acceleration"
