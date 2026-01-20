#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup GROMACS topology for covalent ligand

This script:
1. Parameterizes the ligand with ACPYPE (GAFF2)
2. Finds the covalent bond atoms (Cys SG and ligand reactive C)
3. Creates proper topology with intermolecular bond
4. Combines coordinates

Usage:
    python setup_covalent_topology.py \
        --protein protein.pdb \
        --ligand ligand.sdf \
        --cys_resid 1039 \
        --output_dir md_setup
"""

import argparse
import subprocess
import os
import sys
import shutil
from pathlib import Path
import re


def run_command(cmd, cwd=None, check=True):
    """Run a shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if check and result.returncode != 0:
        print(f"ERROR: Command failed: {cmd}")
        print(f"STDERR: {result.stderr}")
        return None
    return result.stdout


def parameterize_ligand(sdf_file, output_dir):
    """Parameterize ligand with ACPYPE."""
    print("\n[1] Parameterizing ligand with ACPYPE...")
    
    # First, load and fix the molecule with RDKit
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        mol = Chem.SDMolSupplier(sdf_file, removeHs=False)[0]
        if mol is None:
            print("ERROR: Could not read SDF file")
            return None
        
        print(f"  Loaded molecule: {mol.GetNumAtoms()} atoms")
        
        # Ensure 3D coordinates
        if mol.GetNumConformers() == 0:
            print("  Generating 3D coordinates...")
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)
        
        # Write clean SDF
        clean_sdf = os.path.join(output_dir, "ligand_clean.sdf")
        writer = Chem.SDWriter(clean_sdf)
        writer.write(mol)
        writer.close()
        
        # Write PDB (ACPYPE can use PDB too)
        pdb_file = os.path.join(output_dir, "ligand.pdb")
        Chem.MolToPDBFile(mol, pdb_file)
        print(f"  Written: {pdb_file}")
        
    except ImportError:
        clean_sdf = sdf_file
        pdb_file = None
    
    # Try converting to MOL2 with obabel
    mol2_file = os.path.join(output_dir, "ligand.mol2")
    
    # Try different conversion methods
    converted = False
    
    # Method 1: obabel from SDF
    result = subprocess.run(
        ['obabel', clean_sdf, '-O', mol2_file, '-h'],  # -h adds hydrogens if missing
        capture_output=True, text=True, cwd=output_dir
    )
    if os.path.exists(mol2_file) and os.path.getsize(mol2_file) > 100:
        converted = True
        print(f"  Converted to MOL2: {mol2_file}")
    
    # Method 2: obabel from PDB
    if not converted and pdb_file:
        result = subprocess.run(
            ['obabel', pdb_file, '-O', mol2_file],
            capture_output=True, text=True, cwd=output_dir
        )
        if os.path.exists(mol2_file) and os.path.getsize(mol2_file) > 100:
            converted = True
            print(f"  Converted PDB to MOL2: {mol2_file}")
    
    # Method 3: Try antechamber directly on PDB
    if not converted and pdb_file:
        print("  Trying antechamber directly...")
        result = subprocess.run(
            ['antechamber', '-i', os.path.basename(pdb_file), '-fi', 'pdb',
             '-o', 'ligand.mol2', '-fo', 'mol2', '-c', 'bcc', '-nc', '0', '-pf', 'y'],
            capture_output=True, text=True, cwd=output_dir
        )
        if os.path.exists(mol2_file) and os.path.getsize(mol2_file) > 100:
            converted = True
            print(f"  Antechamber created MOL2")
    
    if not converted:
        print("ERROR: Could not convert ligand to MOL2")
        print("  Check if obabel and antechamber are installed:")
        print("    mamba install -c conda-forge openbabel ambertools")
        return None
    
    # Run ACPYPE
    print("  Running ACPYPE...")
    result = subprocess.run(
        ['acpype', '-i', 'ligand.mol2', '-c', 'bcc', '-n', '0', '-a', 'gaff2'],
        capture_output=True, text=True, cwd=output_dir
    )
    
    if result.returncode != 0:
        print(f"  ACPYPE output: {result.stdout}")
        print(f"  ACPYPE errors: {result.stderr}")
    
    # Find ACPYPE output
    acpype_dir = os.path.join(output_dir, "ligand.acpype")
    if not os.path.exists(acpype_dir):
        print("ERROR: ACPYPE failed to create output directory")
        print("Try installing: mamba install -c conda-forge acpype ambertools")
        return None
    
    # Check for output files
    gro_file = os.path.join(acpype_dir, "ligand_GMX.gro")
    itp_file = os.path.join(acpype_dir, "ligand_GMX.itp")
    
    if not os.path.exists(gro_file) or not os.path.exists(itp_file):
        print("ERROR: ACPYPE did not generate GROMACS files")
        print(f"  Check {acpype_dir} for details")
        return None
    
    # Copy files
    shutil.copy(gro_file, os.path.join(output_dir, "ligand.gro"))
    shutil.copy(itp_file, os.path.join(output_dir, "ligand.itp"))
    
    print("  Ligand topology: ligand.itp")
    print("  Ligand coordinates: ligand.gro")
    
    return os.path.join(output_dir, "ligand.itp")


def prepare_protein(pdb_file, output_dir, ff="amber99sb-ildn", water="tip3p"):
    """Process protein with pdb2gmx."""
    print("\n[2] Preparing protein topology...")
    
    protein_pdb = os.path.join(output_dir, "protein.pdb")
    
    # Extract protein atoms only
    with open(pdb_file, 'r') as fin, open(protein_pdb, 'w') as fout:
        for line in fin:
            if line.startswith('ATOM'):
                fout.write(line)
        fout.write("END\n")
    
    # Run pdb2gmx
    cmd = f"echo '1' | gmx pdb2gmx -f {protein_pdb} -o protein.gro -p topol.top " \
          f"-ff {ff} -water {water} -ignh -merge all"
    
    result = run_command(cmd, cwd=output_dir, check=False)
    
    if not os.path.exists(os.path.join(output_dir, "topol.top")):
        print("ERROR: pdb2gmx failed")
        return None
    
    print("  Protein topology: topol.top")
    print("  Protein coordinates: protein.gro")
    
    return os.path.join(output_dir, "topol.top")


def find_atom_in_topology(top_file, resid, atom_name):
    """Find atom number in GROMACS topology."""
    with open(top_file, 'r') as f:
        content = f.read()
    
    # Find atoms section
    atoms_match = re.search(r'\[ atoms \](.*?)(?=\[|\Z)', content, re.DOTALL)
    if not atoms_match:
        return None
    
    for line in atoms_match.group(1).split('\n'):
        if line.strip() and not line.strip().startswith(';'):
            parts = line.split()
            if len(parts) >= 5:
                atom_nr = parts[0]
                res_nr = parts[2]
                name = parts[4]
                
                if int(res_nr) == resid and name == atom_name:
                    return int(atom_nr)
    
    return None


def find_reactive_carbon_in_ligand(itp_file, atom_name=None):
    """Find the reactive carbon in ligand topology."""
    with open(itp_file, 'r') as f:
        content = f.read()
    
    atoms_match = re.search(r'\[ atoms \](.*?)(?=\[|\Z)', content, re.DOTALL)
    if not atoms_match:
        return None, None
    
    atoms = []
    for line in atoms_match.group(1).split('\n'):
        if line.strip() and not line.strip().startswith(';'):
            parts = line.split()
            if len(parts) >= 5:
                atoms.append({
                    'nr': int(parts[0]),
                    'type': parts[1],
                    'name': parts[4]
                })
    
    # If specific name given
    if atom_name:
        for atom in atoms:
            if atom['name'] == atom_name:
                return atom['nr'], atom['name']
    
    # Look for typical beta carbon names
    for name in ['C3', 'C2', 'C24', 'C25', 'CAB', 'CAC']:
        for atom in atoms:
            if atom['name'] == name:
                return atom['nr'], atom['name']
    
    # Find any sp3 carbon (type c3 in GAFF)
    for atom in atoms:
        if atom['type'] == 'c3':
            return atom['nr'], atom['name']
    
    # Fallback: first carbon
    for atom in atoms:
        if atom['type'].startswith('c'):
            return atom['nr'], atom['name']
    
    return 1, atoms[0]['name'] if atoms else 'C1'


def count_atoms_in_gro(gro_file):
    """Count atoms in GRO file."""
    with open(gro_file, 'r') as f:
        lines = f.readlines()
    return int(lines[1].strip())


def combine_coordinates(protein_gro, ligand_gro, output_gro):
    """Combine protein and ligand GRO files."""
    print("\n[3] Combining coordinates...")
    
    with open(protein_gro, 'r') as f:
        prot_lines = f.readlines()
    
    with open(ligand_gro, 'r') as f:
        lig_lines = f.readlines()
    
    prot_atoms = int(prot_lines[1].strip())
    lig_atoms = int(lig_lines[1].strip())
    total = prot_atoms + lig_atoms
    
    with open(output_gro, 'w') as f:
        f.write("Combined system\n")
        f.write(f"{total}\n")
        
        # Protein atoms (skip header and box)
        for line in prot_lines[2:-1]:
            f.write(line)
        
        # Ligand atoms (skip header and box, renumber residue)
        for line in lig_lines[2:-1]:
            # Keep ligand as separate residue
            f.write(line)
        
        # Box from protein
        f.write(prot_lines[-1])
    
    print(f"  Combined: {total} atoms ({prot_atoms} protein + {lig_atoms} ligand)")
    return total, prot_atoms, lig_atoms


def create_covalent_topology(top_file, lig_itp, sg_atom, lig_reactive, 
                             protein_atoms, output_top):
    """Create topology with covalent bond."""
    print("\n[4] Creating covalent bond topology...")
    
    # Calculate global ligand atom number
    lig_reactive_global = protein_atoms + lig_reactive
    
    print(f"  Cys SG atom: {sg_atom}")
    print(f"  Ligand reactive C: {lig_reactive} (global: {lig_reactive_global})")
    
    # Read original topology
    with open(top_file, 'r') as f:
        content = f.read()
    
    # Read ligand itp
    with open(lig_itp, 'r') as f:
        lig_content = f.read()
    
    # Extract ligand moleculetype name
    lig_mol_match = re.search(r'\[ moleculetype \].*?(\w+)\s+\d+', lig_content, re.DOTALL)
    lig_mol_name = lig_mol_match.group(1) if lig_mol_match else "LIG"
    
    # Build new topology
    new_top = []
    
    # Add header and forcefield
    ff_match = re.search(r'(#include.*?forcefield\.itp["\'])', content)
    if ff_match:
        new_top.append(ff_match.group(1))
    else:
        new_top.append('#include "amber99sb-ildn.ff/forcefield.itp"')
    new_top.append("")
    
    # Add ligand atomtypes if present
    atomtypes_match = re.search(r'(\[ atomtypes \].*?)(?=\[ moleculetype \])', 
                                 lig_content, re.DOTALL)
    if atomtypes_match:
        new_top.append(atomtypes_match.group(1))
    
    # Include ligand itp (without atomtypes)
    new_top.append(f'; Include ligand topology')
    new_top.append(f'#include "ligand.itp"')
    new_top.append("")
    
    # Extract protein moleculetype section
    mol_match = re.search(r'(\[ moleculetype \].*?)(?=\[ system \]|\Z)', content, re.DOTALL)
    if mol_match:
        new_top.append(mol_match.group(1))
    
    # Add intermolecular interactions for covalent bond
    new_top.append("; Covalent bond between protein and ligand")
    new_top.append("[ intermolecular_interactions ]")
    new_top.append("[ bonds ]")
    new_top.append("; ai    aj    type    b0 (nm)    kb (kJ/mol/nm^2)")
    new_top.append(f"  {sg_atom}    {lig_reactive_global}    1    0.182    250000.0")
    new_top.append("")
    
    # System section
    new_top.append("[ system ]")
    new_top.append("Covalent complex")
    new_top.append("")
    
    # Molecules section
    new_top.append("[ molecules ]")
    
    # Find protein molecule name
    prot_mol_match = re.search(r'\[ moleculetype \]\s*\n[;\s\w]*\n\s*(\w+)', content)
    prot_mol_name = prot_mol_match.group(1) if prot_mol_match else "Protein"
    
    new_top.append(f"{prot_mol_name}    1")
    new_top.append(f"{lig_mol_name}    1")
    
    # Write new topology
    with open(output_top, 'w') as f:
        f.write('\n'.join(new_top))
    
    print(f"  Covalent topology: {output_top}")
    
    # Also clean up ligand.itp to remove atomtypes (avoid duplicates)
    lig_clean = re.sub(r'\[ atomtypes \].*?(?=\[ moleculetype \])', '', 
                       lig_content, flags=re.DOTALL)
    
    lig_itp_clean = lig_itp.replace('.itp', '_clean.itp')
    with open(lig_itp_clean, 'w') as f:
        f.write(lig_clean)
    
    return output_top


def create_mdp_files(output_dir):
    """Create MDP files for minimization and equilibration."""
    print("\n[5] Creating MDP files...")
    
    # Energy minimization
    em_mdp = """; Energy minimization
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000

nstlist     = 10
cutoff-scheme = Verlet
pbc         = xyz
rlist       = 1.2

coulombtype = PME
rcoulomb    = 1.2
vdwtype     = Cut-off
rvdw        = 1.2

constraints = none
"""
    
    with open(os.path.join(output_dir, "em.mdp"), 'w') as f:
        f.write(em_mdp)
    
    # NVT equilibration
    nvt_mdp = """; NVT equilibration
integrator  = md
dt          = 0.002
nsteps      = 50000      ; 100 ps

nstxout     = 5000
nstvout     = 5000
nstenergy   = 500
nstlog      = 500

nstlist     = 10
cutoff-scheme = Verlet
pbc         = xyz
rlist       = 1.2

coulombtype = PME
rcoulomb    = 1.2
vdwtype     = Cut-off
rvdw        = 1.2

tcoupl      = V-rescale
tc-grps     = System
tau_t       = 0.1
ref_t       = 300

constraints = h-bonds
constraint_algorithm = LINCS

gen_vel     = yes
gen_temp    = 300
gen_seed    = -1
"""
    
    with open(os.path.join(output_dir, "nvt.mdp"), 'w') as f:
        f.write(nvt_mdp)
    
    # NPT equilibration
    npt_mdp = """; NPT equilibration
integrator  = md
dt          = 0.002
nsteps      = 50000      ; 100 ps

nstxout     = 5000
nstvout     = 5000
nstenergy   = 500
nstlog      = 500

nstlist     = 10
cutoff-scheme = Verlet
pbc         = xyz
rlist       = 1.2

coulombtype = PME
rcoulomb    = 1.2
vdwtype     = Cut-off
rvdw        = 1.2

tcoupl      = V-rescale
tc-grps     = System
tau_t       = 0.1
ref_t       = 300

pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5

constraints = h-bonds
constraint_algorithm = LINCS

gen_vel     = no
continuation = yes
"""
    
    with open(os.path.join(output_dir, "npt.mdp"), 'w') as f:
        f.write(npt_mdp)
    
    # Production (20 ns)
    prod_mdp = """; Production MD (20 ns)
integrator  = md
dt          = 0.002
nsteps      = 10000000   ; 20 ns

nstxout-compressed = 5000    ; 10 ps
nstenergy   = 5000
nstlog      = 5000

nstlist     = 10
cutoff-scheme = Verlet
pbc         = xyz
rlist       = 1.2

coulombtype = PME
rcoulomb    = 1.2
vdwtype     = Cut-off
rvdw        = 1.2

tcoupl      = V-rescale
tc-grps     = System
tau_t       = 0.1
ref_t       = 300

pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5

constraints = h-bonds
constraint_algorithm = LINCS

gen_vel     = no
continuation = yes
"""
    
    with open(os.path.join(output_dir, "prod.mdp"), 'w') as f:
        f.write(prod_mdp)
    
    print("  Created: em.mdp, nvt.mdp, npt.mdp, prod.mdp")


def create_run_script(output_dir):
    """Create script to run the full workflow."""
    
    script = """#!/bin/bash
# Run GROMACS covalent ligand equilibration
set -e

echo "=== Creating box and solvating ==="
gmx editconf -f complex.gro -o complex_box.gro -c -d 1.2 -bt dodecahedron
gmx solvate -cp complex_box.gro -cs spc216.gro -o complex_solv.gro -p topol.top

echo "=== Adding ions ==="
gmx grompp -f em.mdp -c complex_solv.gro -p topol.top -o ions.tpr -maxwarn 10
echo "SOL" | gmx genion -s ions.tpr -o complex_ions.gro -p topol.top -pname NA -nname CL -neutral -conc 0.15

echo "=== Energy minimization ==="
gmx grompp -f em.mdp -c complex_ions.gro -p topol.top -o em.tpr -maxwarn 10
gmx mdrun -v -deffnm em

echo "=== NVT equilibration ==="
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 10
gmx mdrun -v -deffnm nvt

echo "=== NPT equilibration ==="
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 10
gmx mdrun -v -deffnm npt

echo "=== Preparing production ==="
gmx grompp -f prod.mdp -c npt.gro -t npt.cpt -p topol.top -o prod.tpr -maxwarn 10

echo ""
echo "=== Setup complete ==="
echo "Run production with: gmx mdrun -v -deffnm prod"
"""
    
    with open(os.path.join(output_dir, "run_equilibration.sh"), 'w') as f:
        f.write(script)
    
    os.chmod(os.path.join(output_dir, "run_equilibration.sh"), 0o755)
    print(f"  Run script: run_equilibration.sh")


def main():
    parser = argparse.ArgumentParser(description="Setup GROMACS covalent topology")
    parser.add_argument('--protein', '-p', required=True, help='Protein PDB file')
    parser.add_argument('--ligand', '-l', required=True, help='Ligand SDF file')
    parser.add_argument('--cys_resid', type=int, required=True, help='Cys residue number')
    parser.add_argument('--reactive_atom', default=None, help='Ligand reactive atom name')
    parser.add_argument('--output_dir', '-o', default='md_covalent')
    parser.add_argument('--ff', default='amber99sb-ildn', help='Force field')
    
    args = parser.parse_args()
    
    print("="*60)
    print("GROMACS Covalent Topology Setup")
    print("="*60)
    print(f"Protein: {args.protein}")
    print(f"Ligand: {args.ligand}")
    print(f"Cys residue: {args.cys_resid}")
    print(f"Output: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Copy input files
    shutil.copy(args.protein, os.path.join(args.output_dir, "protein_input.pdb"))
    shutil.copy(args.ligand, os.path.join(args.output_dir, "ligand_input.sdf"))
    
    # Step 1: Parameterize ligand
    lig_itp = parameterize_ligand(args.ligand, args.output_dir)
    if not lig_itp:
        sys.exit(1)
    
    # Step 2: Prepare protein
    top_file = prepare_protein(
        os.path.join(args.output_dir, "protein_input.pdb"),
        args.output_dir,
        ff=args.ff
    )
    if not top_file:
        sys.exit(1)
    
    # Find SG atom
    sg_atom = find_atom_in_topology(top_file, args.cys_resid, 'SG')
    if not sg_atom:
        print(f"ERROR: Could not find SG in Cys {args.cys_resid}")
        sys.exit(1)
    print(f"\n  Found Cys SG: atom {sg_atom}")
    
    # Find reactive carbon in ligand
    lig_reactive, lig_reactive_name = find_reactive_carbon_in_ligand(
        lig_itp, args.reactive_atom
    )
    print(f"  Found ligand reactive C: atom {lig_reactive} ({lig_reactive_name})")
    
    # Step 3: Combine coordinates
    protein_gro = os.path.join(args.output_dir, "protein.gro")
    ligand_gro = os.path.join(args.output_dir, "ligand.gro")
    complex_gro = os.path.join(args.output_dir, "complex.gro")
    
    total_atoms, protein_atoms, ligand_atoms = combine_coordinates(
        protein_gro, ligand_gro, complex_gro
    )
    
    # Step 4: Create covalent topology
    covalent_top = os.path.join(args.output_dir, "topol.top")
    create_covalent_topology(
        top_file, lig_itp, sg_atom, lig_reactive,
        protein_atoms, covalent_top
    )
    
    # Step 5: Create MDP files
    create_mdp_files(args.output_dir)
    
    # Step 6: Create run script
    create_run_script(args.output_dir)
    
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print(f"\nFiles in: {args.output_dir}/")
    print("""
Next steps:
  cd {output_dir}
  ./run_equilibration.sh
  
Or run steps manually:
  gmx editconf -f complex.gro -o complex_box.gro -c -d 1.2 -bt dodecahedron
  gmx solvate -cp complex_box.gro -cs spc216.gro -o complex_solv.gro -p topol.top
  gmx grompp -f em.mdp -c complex_solv.gro -p topol.top -o ions.tpr -maxwarn 10
  echo "SOL" | gmx genion -s ions.tpr -o complex_ions.gro -p topol.top -pname NA -nname CL -neutral
  gmx grompp -f em.mdp -c complex_ions.gro -p topol.top -o em.tpr -maxwarn 10
  gmx mdrun -v -deffnm em
  ... (continue with nvt, npt, prod)
""".format(output_dir=args.output_dir))


if __name__ == "__main__":
    main()