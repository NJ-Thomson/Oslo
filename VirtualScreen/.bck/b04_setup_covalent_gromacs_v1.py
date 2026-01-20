#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GROMACS Setup for Covalent Complex

This script prepares a covalent protein-ligand complex for MD simulation.

Workflow:
1. Extract and parameterize ligand (using acpype/GAFF2)
2. Prepare protein topology (with modified cysteine)
3. Define covalent bond between Cys SG and ligand
4. Combine topologies
5. Solvate and add ions
6. Generate MDP files for EM, NVT, NPT, production

Requirements:
    - GROMACS (gmx)
    - acpype (pip install acpype)
    - AmberTools (for antechamber) OR
    - OpenBabel

Usage:
    python setup_covalent_gromacs.py \
        --complex covalent_complex.pdb \
        --ligand best_pose.sdf \
        --cys_chain A --cys_resid 1039 \
        --output md_setup

Then run the simulations:
    cd md_setup
    ./run_simulation.sh
"""

import argparse
import subprocess
import os
import sys
import shutil
from pathlib import Path

def check_dependencies():
    """Check required tools are available."""
    tools = {
        'gmx': 'GROMACS',
        'acpype': 'acpype (pip install acpype)',
    }
    
    missing = []
    for cmd, name in tools.items():
        result = subprocess.run(['which', cmd], capture_output=True)
        if result.returncode != 0:
            missing.append(name)
    
    if missing:
        print("WARNING: Missing tools:")
        for t in missing:
            print(f"  - {t}")
        print("\nWill attempt to proceed, but may fail.")
    
    return len(missing) == 0


def extract_ligand_pdb(complex_pdb, output_pdb):
    """Extract ligand (LIG) from complex PDB."""
    with open(complex_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out:
        for line in f_in:
            if line.startswith('HETATM') and ' LIG ' in line:
                f_out.write(line)
        f_out.write("END\n")
    return output_pdb


def parameterize_ligand_acpype(ligand_sdf, output_dir, net_charge=0):
    """
    Parameterize ligand using acpype (GAFF2 force field).
    
    Returns paths to ligand topology files.
    """
    print("\n  Running acpype for ligand parameterization...")
    
    # acpype works best with mol2 or sdf
    cmd = [
        'acpype',
        '-i', str(ligand_sdf),
        '-c', 'bcc',  # AM1-BCC charges
        '-n', str(net_charge),
        '-o', 'gmx',  # GROMACS output
        '-b', 'LIG'   # Base name
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)
    
    if result.returncode != 0:
        print(f"  acpype stderr: {result.stderr}")
        # Try with gaff2
        cmd.extend(['-a', 'gaff2'])
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)
    
    # Find output directory (acpype creates LIG.acpype/)
    acpype_dir = output_dir / "LIG.acpype"
    if not acpype_dir.exists():
        # Try alternative naming
        for d in output_dir.iterdir():
            if d.is_dir() and 'acpype' in d.name:
                acpype_dir = d
                break
    
    if not acpype_dir.exists():
        print(f"  ERROR: acpype output not found")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return None
    
    return {
        'itp': acpype_dir / "LIG_GMX.itp",
        'gro': acpype_dir / "LIG_GMX.gro",
        'top': acpype_dir / "LIG_GMX.top",
    }


def prepare_protein_topology(complex_pdb, output_dir, cys_chain, cys_resid, ff='amber99sb-ildn'):
    """
    Prepare protein topology with pdb2gmx.
    
    The cysteine will be treated as a regular CYS initially.
    We'll modify it after to be covalently bound.
    """
    print("\n  Running pdb2gmx for protein...")
    
    # First, extract protein only (no LIG)
    protein_pdb = output_dir / "protein_only.pdb"
    with open(complex_pdb, 'r') as f_in, open(protein_pdb, 'w') as f_out:
        for line in f_in:
            if line.startswith('ATOM'):
                f_out.write(line)
            elif line.startswith('TER'):
                f_out.write(line)
        f_out.write("END\n")
    
    # Run pdb2gmx
    cmd = [
        'gmx', 'pdb2gmx',
        '-f', str(protein_pdb),
        '-o', str(output_dir / 'protein.gro'),
        '-p', str(output_dir / 'protein.top'),
        '-i', str(output_dir / 'protein_posre.itp'),
        '-ff', ff,
        '-water', 'tip3p',
        '-ignh',  # Ignore hydrogens in input
        '-his',   # Interactive histidine - we'll handle this
    ]
    
    # For non-interactive, use echo to select defaults
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True, 
        cwd=output_dir,
        input='0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'  # Default HIS states
    )
    
    if result.returncode != 0:
        print(f"  pdb2gmx stderr: {result.stderr}")
        # Try without -his flag
        cmd.remove('-his')
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)
    
    return {
        'gro': output_dir / 'protein.gro',
        'top': output_dir / 'protein.top',
    }


def find_cys_sg_index(gro_file, cys_resid):
    """Find the atom index of Cys SG in the GRO file."""
    with open(gro_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines[2:-1], start=1):  # Skip header and box
        if len(line) >= 20:
            try:
                resid = int(line[0:5].strip())
                atom_name = line[10:15].strip()
                if resid == cys_resid and atom_name == 'SG':
                    return i
            except:
                continue
    return None


def create_covalent_topology(protein_top, ligand_itp, output_top, 
                             cys_resid, sg_index, ligand_c_index,
                             bond_length=0.181, bond_fc=238000):
    """
    Create combined topology with covalent bond defined.
    
    Parameters:
        bond_length: C-S bond length in nm (default 0.181 nm = 1.81 A)
        bond_fc: Force constant in kJ/mol/nm^2
    """
    print("\n  Creating combined topology with covalent bond...")
    
    # Read protein topology
    with open(protein_top, 'r') as f:
        protein_content = f.read()
    
    # Read ligand itp
    with open(ligand_itp, 'r') as f:
        ligand_content = f.read()
    
    # Create the combined topology
    combined = []
    
    # Add header and forcefield
    combined.append("; Covalent complex topology\n")
    combined.append("; Generated by setup_covalent_gromacs.py\n\n")
    
    # Extract forcefield include from protein topology
    for line in protein_content.split('\n'):
        if '#include' in line and 'forcefield' in line:
            combined.append(line + '\n')
            break
    
    combined.append('\n; Include ligand parameters\n')
    combined.append('#include "ligand.itp"\n\n')
    
    # Add rest of protein topology (skip the forcefield include)
    in_moleculetype = False
    skip_until_moleculetype = True
    
    for line in protein_content.split('\n'):
        if '[ moleculetype ]' in line:
            skip_until_moleculetype = False
        if skip_until_moleculetype:
            if '#include' in line and 'forcefield' not in line:
                combined.append(line + '\n')
            continue
        combined.append(line + '\n')
    
    # Add intermolecular bond section
    combined.append('\n; Covalent bond between protein and ligand\n')
    combined.append('[ intermolecular_interactions ]\n')
    combined.append('[ bonds ]\n')
    combined.append(f'; ai   aj   funct   r0(nm)   fc(kJ/mol/nm^2)\n')
    combined.append(f'  {sg_index}   {ligand_c_index}   1   {bond_length:.4f}   {bond_fc}\n')
    
    # Modify [ molecules ] section to include ligand
    output_lines = []
    for line in combined:
        output_lines.append(line)
    
    # Find and modify molecules section
    final_content = ''.join(output_lines)
    if '[ molecules ]' in final_content:
        final_content = final_content.replace(
            '[ molecules ]\n',
            '[ molecules ]\n; Compound        nmols\n'
        )
        # Add ligand to molecules
        lines = final_content.split('\n')
        new_lines = []
        for i, line in enumerate(lines):
            new_lines.append(line)
            if 'Protein' in line and lines[i-1].strip() == '' or 'Protein_chain' in line:
                # Add ligand after protein
                pass
        
        # Append ligand to end
        final_content = final_content.rstrip() + '\nLIG              1\n'
    
    with open(output_top, 'w') as f:
        f.write(final_content)
    
    # Also copy and rename ligand itp
    shutil.copy(ligand_itp, output_top.parent / 'ligand.itp')
    
    return output_top


def combine_coordinates(protein_gro, ligand_gro, output_gro):
    """Combine protein and ligand GRO files."""
    print("\n  Combining coordinates...")
    
    # Read protein
    with open(protein_gro, 'r') as f:
        prot_lines = f.readlines()
    
    # Read ligand
    with open(ligand_gro, 'r') as f:
        lig_lines = f.readlines()
    
    # Parse
    prot_title = prot_lines[0]
    prot_natoms = int(prot_lines[1].strip())
    prot_atoms = prot_lines[2:2+prot_natoms]
    prot_box = prot_lines[-1]
    
    lig_natoms = int(lig_lines[1].strip())
    lig_atoms = lig_lines[2:2+lig_natoms]
    
    # Renumber ligand residue
    new_lig_atoms = []
    for line in lig_atoms:
        # GRO format: resid(5) resname(5) atomname(5) atomnr(5) x y z
        new_line = f"{prot_natoms+1:5d}{'LIG':>5s}{line[10:]}"
        new_lig_atoms.append(new_line)
    
    # Combine
    total_atoms = prot_natoms + lig_natoms
    
    with open(output_gro, 'w') as f:
        f.write(f"Covalent complex\n")
        f.write(f"{total_atoms}\n")
        for line in prot_atoms:
            f.write(line)
        for line in new_lig_atoms:
            f.write(line)
        f.write(prot_box)
    
    return output_gro


def create_mdp_files(output_dir):
    """Create MDP files for minimization, NVT, NPT, and production."""
    
    # Energy minimization
    em_mdp = """; Energy minimization
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
"""
    
    # NVT equilibration
    nvt_mdp = """; NVT equilibration
integrator  = md
nsteps      = 50000      ; 100 ps
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
"""
    
    # NPT equilibration
    npt_mdp = """; NPT equilibration
integrator  = md
nsteps      = 50000      ; 100 ps
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
"""
    
    # Production (20 ns)
    prod_mdp = """; Production MD - 20 ns NPT
integrator  = md
nsteps      = 10000000   ; 20 ns
dt          = 0.002

nstxout     = 0          ; Don't write coords to trr
nstvout     = 0
nstfout     = 0
nstxout-compressed = 5000  ; Write to xtc every 10 ps
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
"""
    
    # Write files
    (output_dir / 'em.mdp').write_text(em_mdp)
    (output_dir / 'nvt.mdp').write_text(nvt_mdp)
    (output_dir / 'npt.mdp').write_text(npt_mdp)
    (output_dir / 'prod.mdp').write_text(prod_mdp)
    
    print(f"  Created: em.mdp, nvt.mdp, npt.mdp, prod.mdp")


def create_run_script(output_dir):
    """Create shell script to run the full simulation."""
    
    script = """#!/bin/bash
# GROMACS simulation script for covalent complex
# Run from the md_setup directory

set -e  # Exit on error

echo "=== Step 1: Energy Minimization ==="
gmx grompp -f em.mdp -c complex_solv_ions.gro -p topol.top -o em.tpr -maxwarn 2
gmx mdrun -v -deffnm em

echo "=== Step 2: NVT Equilibration (100 ps) ==="
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 2
gmx mdrun -v -deffnm nvt

echo "=== Step 3: NPT Equilibration (100 ps) ==="
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 2
gmx mdrun -v -deffnm npt

echo "=== Step 4: Production MD (20 ns) ==="
gmx grompp -f prod.mdp -c npt.gro -t npt.cpt -p topol.top -o prod.tpr -maxwarn 2
gmx mdrun -v -deffnm prod

echo "=== Simulation Complete ==="
echo "Trajectory: prod.xtc"
echo "Final structure: prod.gro"

# Basic analysis
echo ""
echo "=== Quick Analysis ==="
echo "Energy:"
echo "0" | gmx energy -f prod.edr -o energy.xvg

echo ""
echo "RMSD:"
echo "4 4" | gmx rms -s prod.tpr -f prod.xtc -o rmsd.xvg

echo "Done!"
"""
    
    script_path = output_dir / 'run_simulation.sh'
    script_path.write_text(script)
    script_path.chmod(0o755)
    
    print(f"  Created: run_simulation.sh")


def solvate_and_ionize(gro_file, top_file, output_dir):
    """Add water box and ions."""
    print("\n  Solvating system...")
    
    # Define box
    box_gro = output_dir / 'complex_box.gro'
    cmd = ['gmx', 'editconf', '-f', str(gro_file), '-o', str(box_gro),
           '-c', '-d', '1.2', '-bt', 'dodecahedron']
    subprocess.run(cmd, capture_output=True, cwd=output_dir)
    
    # Solvate
    solv_gro = output_dir / 'complex_solv.gro'
    cmd = ['gmx', 'solvate', '-cp', str(box_gro), '-cs', 'spc216.gro',
           '-o', str(solv_gro), '-p', str(top_file)]
    subprocess.run(cmd, capture_output=True, cwd=output_dir)
    
    # Add ions
    ions_tpr = output_dir / 'ions.tpr'
    ions_mdp = output_dir / 'ions.mdp'
    ions_mdp.write_text("; Ions\n")
    
    cmd = ['gmx', 'grompp', '-f', str(ions_mdp), '-c', str(solv_gro),
           '-p', str(top_file), '-o', str(ions_tpr), '-maxwarn', '5']
    subprocess.run(cmd, capture_output=True, cwd=output_dir)
    
    final_gro = output_dir / 'complex_solv_ions.gro'
    cmd = ['gmx', 'genion', '-s', str(ions_tpr), '-o', str(final_gro),
           '-p', str(top_file), '-pname', 'NA', '-nname', 'CL', '-neutral']
    # Need to select SOL group
    subprocess.run(cmd, input='SOL\n', capture_output=True, text=True, cwd=output_dir)
    
    return final_gro


def main():
    parser = argparse.ArgumentParser(
        description="Setup GROMACS simulation for covalent complex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python %(prog)s \\
        --complex covalent_complex.pdb \\
        --ligand best_pose.sdf \\
        --cys_chain A --cys_resid 1039 \\
        --output md_setup

After setup:
    cd md_setup
    ./run_simulation.sh
        """
    )
    
    parser.add_argument('--complex', '-c', required=True,
                        help='Covalent complex PDB (from build_covalent_complex.py)')
    parser.add_argument('--ligand', '-l', required=True,
                        help='Ligand SDF (with placeholder S, for parameterization)')
    parser.add_argument('--cys_chain', default='A')
    parser.add_argument('--cys_resid', type=int, required=True)
    parser.add_argument('--output', '-o', default='md_setup')
    parser.add_argument('--charge', type=int, default=0,
                        help='Net charge of ligand')
    parser.add_argument('--ff', default='amber99sb-ildn',
                        help='Force field for protein')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GROMACS SETUP FOR COVALENT COMPLEX")
    print("=" * 60)
    
    # Check dependencies
    check_dependencies()
    
    # Create output directory
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Parameterize ligand
    print("\n[1] Parameterizing ligand with acpype...")
    ligand_files = parameterize_ligand_acpype(
        Path(args.ligand), outdir, args.charge
    )
    
    if ligand_files is None:
        print("ERROR: Ligand parameterization failed")
        print("\nManual alternative:")
        print("  1. Use CGenFF: https://cgenff.umaryland.edu/")
        print("  2. Use ATB: https://atb.uq.edu.au/")
        print("  3. Use LigParGen: http://zarbi.chem.yale.edu/ligpargen/")
        sys.exit(1)
    
    print(f"    ITP: {ligand_files['itp']}")
    
    # Step 2: Prepare protein topology
    print("\n[2] Preparing protein topology...")
    protein_files = prepare_protein_topology(
        Path(args.complex), outdir, args.cys_chain, args.cys_resid, args.ff
    )
    
    # Step 3: Find atom indices for covalent bond
    print("\n[3] Finding covalent bond atoms...")
    sg_index = find_cys_sg_index(protein_files['gro'], args.cys_resid)
    print(f"    Cys SG atom index: {sg_index}")
    
    # For the ligand, we need to find the beta carbon
    # This is typically atom 26 or similar - need to check ligand ITP
    # For now, we'll need manual input or detection
    print("    NOTE: Check ligand.itp to find the beta carbon index")
    print("    Look for the carbon that should bond to SG")
    
    # Step 4: Combine coordinates
    print("\n[4] Combining coordinates...")
    complex_gro = combine_coordinates(
        protein_files['gro'],
        ligand_files['gro'],
        outdir / 'complex.gro'
    )
    
    # Step 5: Create MDP files
    print("\n[5] Creating MDP files...")
    create_mdp_files(outdir)
    
    # Step 6: Create run script
    print("\n[6] Creating run script...")
    create_run_script(outdir)
    
    # Step 7: Solvate (optional - can do manually)
    print("\n[7] Solvating system...")
    try:
        # Copy topology first
        shutil.copy(protein_files['top'], outdir / 'topol.top')
        # Add ligand include to topology
        with open(outdir / 'topol.top', 'r') as f:
            content = f.read()
        # Insert ligand include after forcefield
        lines = content.split('\n')
        new_lines = []
        for line in lines:
            new_lines.append(line)
            if '#include' in line and 'forcefield' in line:
                new_lines.append('')
                new_lines.append('; Include ligand topology')
                new_lines.append('#include "ligand.itp"')
        # Add LIG to molecules
        final_lines = []
        for line in new_lines:
            final_lines.append(line)
        content = '\n'.join(final_lines)
        content = content.rstrip() + '\nLIG              1\n'
        with open(outdir / 'topol.top', 'w') as f:
            f.write(content)
        
        # Copy ligand itp
        shutil.copy(ligand_files['itp'], outdir / 'ligand.itp')
        
        solvate_and_ionize(complex_gro, outdir / 'topol.top', outdir)
    except Exception as e:
        print(f"  Solvation failed: {e}")
        print("  You may need to solvate manually")
    
    # Summary
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {outdir}/")
    print(f"\nIMPORTANT - Manual steps required:")
    print(f"  1. Check ligand.itp for the beta carbon atom number")
    print(f"  2. Add [ intermolecular_interactions ] to topol.top:")
    print(f"     [ intermolecular_interactions ]")
    print(f"     [ bonds ]")
    print(f"     ; ai   aj   funct   r0(nm)   fc")
    print(f"     {sg_index}    XXX   1       0.1810   238000")
    print(f"     (Replace XXX with beta carbon atom number from ligand)")
    print(f"\n  3. Run: cd {outdir} && ./run_simulation.sh")


if __name__ == "__main__":
    main()
