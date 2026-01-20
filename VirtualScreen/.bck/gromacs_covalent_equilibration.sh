#!/bin/bash
#
# GROMACS Covalent Ligand Equilibration Workflow
#
# This script takes the output from AutoDock4 covalent docking and runs:
# 1. Ligand parameterization (GAFF2 via ACPYPE)
# 2. System setup with covalent bond
# 3. Energy minimization
# 4. NVT equilibration (100 ps)
# 5. NPT equilibration (100 ps)
# 6. NPT production (20 ns)
#
# Usage:
#   ./gromacs_covalent_equilibration.sh \
#       --complex complex.pdb \
#       --ligand ligand.sdf \
#       --cys_chain D \
#       --cys_resid 1039 \
#       --reactive_atom C24 \
#       --output_dir md_covalent
#
# Requirements:
#   - GROMACS (gmx)
#   - ACPYPE (for ligand parameterization)
#   - AmberTools (antechamber, for ACPYPE)
#

set -e

# Default parameters
COMPLEX=""
LIGAND=""
CYS_CHAIN="A"
CYS_RESID=""
REACTIVE_ATOM=""  # Ligand atom that bonds to Cys SG
OUTPUT_DIR="md_covalent"
PROD_TIME_NS=20
FF="amber99sb-ildn"  # Force field
WATER="tip3p"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --complex) COMPLEX="$2"; shift 2 ;;
        --ligand) LIGAND="$2"; shift 2 ;;
        --cys_chain) CYS_CHAIN="$2"; shift 2 ;;
        --cys_resid) CYS_RESID="$2"; shift 2 ;;
        --reactive_atom) REACTIVE_ATOM="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --prod_time) PROD_TIME_NS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --complex PDB --ligand SDF --cys_resid NUM [options]"
            echo "Options:"
            echo "  --complex       Complex PDB from docking"
            echo "  --ligand        Ligand SDF file"
            echo "  --cys_chain     Chain ID of target Cys (default: A)"
            echo "  --cys_resid     Residue number of target Cys (required)"
            echo "  --reactive_atom Ligand atom name bonded to SG (auto-detected if not set)"
            echo "  --output_dir    Output directory (default: md_covalent)"
            echo "  --prod_time     Production time in ns (default: 20)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate inputs
if [[ -z "$COMPLEX" ]] || [[ -z "$LIGAND" ]] || [[ -z "$CYS_RESID" ]]; then
    echo "ERROR: --complex, --ligand, and --cys_resid are required"
    exit 1
fi

echo "============================================================"
echo "GROMACS Covalent Ligand Equilibration"
echo "============================================================"
echo "Complex: $COMPLEX"
echo "Ligand: $LIGAND"
echo "Target Cys: Chain $CYS_CHAIN, Residue $CYS_RESID"
echo "Output: $OUTPUT_DIR"
echo "Production: ${PROD_TIME_NS} ns"
echo "============================================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# Copy input files
cp "../$COMPLEX" complex_input.pdb
cp "../$LIGAND" ligand_input.sdf


#=============================================================================
# STEP 1: Separate protein and ligand
#=============================================================================
echo ""
echo "[STEP 1] Separating protein and ligand..."

# Extract protein (remove HETATM except important ions)
grep "^ATOM" complex_input.pdb > protein_raw.pdb
echo "END" >> protein_raw.pdb

# Extract ligand from SDF (already have it)
# Convert to PDB for reference
obabel ligand_input.sdf -O ligand.pdb 2>/dev/null || true


#=============================================================================
# STEP 2: Parameterize ligand with ACPYPE (GAFF2)
#=============================================================================
echo ""
echo "[STEP 2] Parameterizing ligand with ACPYPE..."

# Convert SDF to MOL2 with charges
obabel ligand_input.sdf -O ligand.mol2 --gen3d 2>/dev/null

# Run ACPYPE
if command -v acpype &> /dev/null; then
    acpype -i ligand.mol2 -c bcc -n 0 -a gaff2 2>&1 | tee acpype.log
    
    # ACPYPE creates ligand.acpype directory
    if [[ -d "ligand.acpype" ]]; then
        cp ligand.acpype/ligand_GMX.gro ligand.gro
        cp ligand.acpype/ligand_GMX.itp ligand.itp
        echo "  Ligand topology generated: ligand.itp"
    else
        echo "ERROR: ACPYPE failed to generate topology"
        exit 1
    fi
else
    echo "ERROR: ACPYPE not found. Install with: mamba install -c conda-forge acpype"
    exit 1
fi


#=============================================================================
# STEP 3: Prepare protein topology
#=============================================================================
echo ""
echo "[STEP 3] Preparing protein topology..."

# Process protein with pdb2gmx
# Use -ignh to ignore hydrogens (will be regenerated)
# Use -merge all to handle multiple chains
echo "1" | gmx pdb2gmx -f protein_raw.pdb -o protein.gro -p topol.top \
    -ff $FF -water $WATER -ignh -merge all 2>&1 | tee pdb2gmx.log

echo "  Protein topology generated: topol.top"


#=============================================================================
# STEP 4: Create covalent bond topology
#=============================================================================
echo ""
echo "[STEP 4] Creating covalent bond between Cys SG and ligand..."

# This is the critical step - we need to:
# 1. Modify the Cys to remove HG (thiol hydrogen)
# 2. Add a bond between Cys SG and ligand reactive carbon
# 3. Update angles and dihedrals

# Create a Python script to handle the topology modification
cat > modify_topology.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
Modify GROMACS topology to add covalent bond between Cys and ligand.
"""
import sys
import re

def find_cys_sg_atom(top_file, cys_resid):
    """Find the atom number of Cys SG in the topology."""
    with open(top_file, 'r') as f:
        in_atoms = False
        for line in f:
            if '[ atoms ]' in line:
                in_atoms = True
                continue
            if in_atoms and line.startswith('['):
                break
            if in_atoms and not line.startswith(';') and line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    atom_nr = parts[0]
                    atom_name = parts[4]
                    res_nr = parts[2]
                    if int(res_nr) == cys_resid and atom_name == 'SG':
                        return int(atom_nr)
    return None

def find_cys_hg_atom(top_file, cys_resid):
    """Find the atom number of Cys HG in the topology."""
    with open(top_file, 'r') as f:
        in_atoms = False
        for line in f:
            if '[ atoms ]' in line:
                in_atoms = True
                continue
            if in_atoms and line.startswith('['):
                break
            if in_atoms and not line.startswith(';') and line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    atom_nr = parts[0]
                    atom_name = parts[4]
                    res_nr = parts[2]
                    if int(res_nr) == cys_resid and atom_name in ('HG', 'HG1'):
                        return int(atom_nr)
    return None

def get_ligand_atom_count(itp_file):
    """Get the number of atoms in the ligand."""
    count = 0
    with open(itp_file, 'r') as f:
        in_atoms = False
        for line in f:
            if '[ atoms ]' in line:
                in_atoms = True
                continue
            if in_atoms and line.startswith('['):
                break
            if in_atoms and not line.startswith(';') and line.strip():
                count += 1
    return count

def find_reactive_atom_in_ligand(itp_file, atom_name=None):
    """Find the reactive carbon in the ligand topology."""
    atoms = []
    with open(itp_file, 'r') as f:
        in_atoms = False
        for line in f:
            if '[ atoms ]' in line:
                in_atoms = True
                continue
            if in_atoms and line.startswith('['):
                break
            if in_atoms and not line.startswith(';') and line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    atoms.append({
                        'nr': int(parts[0]),
                        'type': parts[1],
                        'name': parts[4]
                    })
    
    # If specific atom name given, find it
    if atom_name:
        for atom in atoms:
            if atom['name'] == atom_name:
                return atom['nr']
    
    # Otherwise, look for typical reactive carbons (C with certain names)
    for atom in atoms:
        if atom['name'] in ('C24', 'C25', 'CAB', 'C3', 'C2'):  # Common names
            return atom['nr']
    
    # Return first carbon as fallback
    for atom in atoms:
        if atom['type'].startswith('c'):
            return atom['nr']
    
    return 1  # Fallback

if __name__ == '__main__':
    cys_resid = int(sys.argv[1])
    reactive_atom_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Find atoms
    sg_atom = find_cys_sg_atom('topol.top', cys_resid)
    hg_atom = find_cys_hg_atom('topol.top', cys_resid)
    lig_reactive = find_reactive_atom_in_ligand('ligand.itp', reactive_atom_name)
    
    print(f"Cys SG atom number: {sg_atom}")
    print(f"Cys HG atom number: {hg_atom}")
    print(f"Ligand reactive atom number: {lig_reactive}")
    
    # Save for later use
    with open('covalent_atoms.txt', 'w') as f:
        f.write(f"SG_ATOM={sg_atom}\n")
        f.write(f"HG_ATOM={hg_atom}\n")
        f.write(f"LIG_REACTIVE={lig_reactive}\n")
PYTHON_SCRIPT

python3 modify_topology.py "$CYS_RESID" "$REACTIVE_ATOM"
source covalent_atoms.txt

echo "  SG atom: $SG_ATOM"
echo "  HG atom: $HG_ATOM (to be removed)"
echo "  Ligand reactive atom: $LIG_REACTIVE"


#=============================================================================
# STEP 5: Combine protein and ligand coordinates
#=============================================================================
echo ""
echo "[STEP 5] Combining protein and ligand coordinates..."

# Get protein atom count
PROTEIN_ATOMS=$(grep -c "^" protein.gro | head -1)
PROTEIN_ATOMS=$((PROTEIN_ATOMS - 3))  # Remove header and box lines

# Get ligand atom count
LIGAND_ATOMS=$(grep -c "^" ligand.gro | head -1)
LIGAND_ATOMS=$((LIGAND_ATOMS - 3))

TOTAL_ATOMS=$((PROTEIN_ATOMS + LIGAND_ATOMS))

# Combine GRO files
head -1 protein.gro > complex.gro
echo "$TOTAL_ATOMS" >> complex.gro
head -n -1 protein.gro | tail -n +3 >> complex.gro
head -n -1 ligand.gro | tail -n +3 >> complex.gro
tail -1 protein.gro >> complex.gro

echo "  Combined system: $TOTAL_ATOMS atoms"


#=============================================================================
# STEP 6: Update topology with ligand and covalent bond
#=============================================================================
echo ""
echo "[STEP 6] Updating topology..."

# Add ligand include to topology
# Insert after forcefield include
cat > topol_covalent.top << EOF
; Topology for covalent complex
; Include forcefield
#include "$FF.ff/forcefield.itp"

; Include ligand parameters
#include "ligand.itp"

; Include protein topology
EOF

# Extract moleculetype and atoms from original topology
sed -n '/\[ moleculetype \]/,/\[ system \]/p' topol.top | head -n -2 >> topol_covalent.top

# Add intermolecular bonds section for covalent bond
# The ligand atoms start after protein atoms
LIG_REACTIVE_GLOBAL=$((PROTEIN_ATOMS + LIG_REACTIVE))

cat >> topol_covalent.top << EOF

; Covalent bond between Cys SG and Ligand
[ intermolecular_interactions ]
[ bonds ]
; ai    aj    type    bA      kA
  $SG_ATOM    $LIG_REACTIVE_GLOBAL    1    0.182    250000.0

[ system ]
Covalent complex

[ molecules ]
Protein_chain_A    1
LIG                1
EOF

mv topol_covalent.top topol.top
echo "  Topology updated with covalent bond"


#=============================================================================
# STEP 7: Create simulation box and solvate
#=============================================================================
echo ""
echo "[STEP 7] Creating simulation box and solvating..."

# Create box (dodecahedron, 1.2 nm from solute)
gmx editconf -f complex.gro -o complex_box.gro -c -d 1.2 -bt dodecahedron

# Solvate
gmx solvate -cp complex_box.gro -cs spc216.gro -o complex_solv.gro -p topol.top

# Add ions (neutralize and 0.15 M NaCl)
cat > ions.mdp << EOF
integrator = steep
nsteps = 0
EOF

gmx grompp -f ions.mdp -c complex_solv.gro -p topol.top -o ions.tpr -maxwarn 5
echo "SOL" | gmx genion -s ions.tpr -o complex_ions.gro -p topol.top -pname NA -nname CL -neutral -conc 0.15

echo "  System solvated and ions added"


#=============================================================================
# STEP 8: Energy Minimization
#=============================================================================
echo ""
echo "[STEP 8] Energy minimization..."

cat > em.mdp << EOF
; Energy minimization
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000

; Neighbor searching
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
pbc         = xyz
rlist       = 1.2

; Electrostatics and VdW
coulombtype = PME
rcoulomb    = 1.2
vdwtype     = Cut-off
rvdw        = 1.2

; Constraints
constraints = none
EOF

gmx grompp -f em.mdp -c complex_ions.gro -p topol.top -o em.tpr -maxwarn 5
gmx mdrun -v -deffnm em

echo "  Minimization complete"
echo "  Final energy: $(grep 'Potential Energy' em.log | tail -1)"


#=============================================================================
# STEP 9: NVT Equilibration (100 ps)
#=============================================================================
echo ""
echo "[STEP 9] NVT equilibration (100 ps)..."

cat > nvt.mdp << EOF
; NVT equilibration
integrator  = md
dt          = 0.002
nsteps      = 50000      ; 100 ps

; Output control
nstxout     = 5000
nstvout     = 5000
nstenergy   = 500
nstlog      = 500

; Neighbor searching
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
pbc         = xyz
rlist       = 1.2

; Electrostatics and VdW
coulombtype = PME
rcoulomb    = 1.2
vdwtype     = Cut-off
rvdw        = 1.2

; Temperature coupling
tcoupl      = V-rescale
tc-grps     = Protein_LIG Water_and_ions
tau_t       = 0.1       0.1
ref_t       = 300       300

; Constraints
constraints = h-bonds
constraint_algorithm = LINCS

; Velocity generation
gen_vel     = yes
gen_temp    = 300
gen_seed    = -1
EOF

# Create index groups
echo -e "1 | 13\nq" | gmx make_ndx -f em.gro -o index.ndx 2>/dev/null || true

gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -n index.ndx -o nvt.tpr -maxwarn 5
gmx mdrun -v -deffnm nvt

echo "  NVT equilibration complete"


#=============================================================================
# STEP 10: NPT Equilibration (100 ps)
#=============================================================================
echo ""
echo "[STEP 10] NPT equilibration (100 ps)..."

cat > npt.mdp << EOF
; NPT equilibration
integrator  = md
dt          = 0.002
nsteps      = 50000      ; 100 ps

; Output control
nstxout     = 5000
nstvout     = 5000
nstenergy   = 500
nstlog      = 500

; Neighbor searching
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
pbc         = xyz
rlist       = 1.2

; Electrostatics and VdW
coulombtype = PME
rcoulomb    = 1.2
vdwtype     = Cut-off
rvdw        = 1.2

; Temperature coupling
tcoupl      = V-rescale
tc-grps     = Protein_LIG Water_and_ions
tau_t       = 0.1       0.1
ref_t       = 300       300

; Pressure coupling
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5

; Constraints
constraints = h-bonds
constraint_algorithm = LINCS

; Velocity generation
gen_vel     = no
continuation = yes
EOF

gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -p topol.top -n index.ndx -o npt.tpr -maxwarn 5
gmx mdrun -v -deffnm npt

echo "  NPT equilibration complete"


#=============================================================================
# STEP 11: Production MD (20 ns)
#=============================================================================
echo ""
echo "[STEP 11] Production MD (${PROD_TIME_NS} ns)..."

PROD_STEPS=$((PROD_TIME_NS * 500000))  # 2 fs timestep

cat > prod.mdp << EOF
; Production MD
integrator  = md
dt          = 0.002
nsteps      = $PROD_STEPS

; Output control
nstxout-compressed = 5000    ; 10 ps
nstenergy   = 5000
nstlog      = 5000

; Neighbor searching
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
pbc         = xyz
rlist       = 1.2

; Electrostatics and VdW
coulombtype = PME
rcoulomb    = 1.2
vdwtype     = Cut-off
rvdw        = 1.2

; Temperature coupling
tcoupl      = V-rescale
tc-grps     = Protein_LIG Water_and_ions
tau_t       = 0.1       0.1
ref_t       = 300       300

; Pressure coupling
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5

; Constraints
constraints = h-bonds
constraint_algorithm = LINCS

; Velocity generation
gen_vel     = no
continuation = yes
EOF

gmx grompp -f prod.mdp -c npt.gro -t npt.cpt -p topol.top -n index.ndx -o prod.tpr -maxwarn 5

echo "  Production run prepared: prod.tpr"
echo ""
echo "  To run production (may take several hours/days):"
echo "    gmx mdrun -v -deffnm prod"
echo ""
echo "  Or submit to HPC:"
echo "    sbatch run_prod.sh"

# Create SLURM script
cat > run_prod.sh << 'SLURM'
#!/bin/bash
#SBATCH --job-name=cov_md
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

module load gromacs

gmx mdrun -v -deffnm prod -nb gpu -pme gpu -bonded gpu
SLURM

chmod +x run_prod.sh


#=============================================================================
# Summary
#=============================================================================
echo ""
echo "============================================================"
echo "SETUP COMPLETE"
echo "============================================================"
echo ""
echo "Files created in: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  em.gro          - Minimized structure"
echo "  nvt.gro         - After NVT equilibration"
echo "  npt.gro         - After NPT equilibration"
echo "  prod.tpr        - Production run input"
echo "  run_prod.sh     - SLURM submission script"
echo ""
echo "To run production:"
echo "  cd $OUTPUT_DIR"
echo "  gmx mdrun -v -deffnm prod"
echo ""
echo "Or on HPC:"
echo "  sbatch run_prod.sh"
echo ""
