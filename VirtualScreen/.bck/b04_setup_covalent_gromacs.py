#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GROMACS Setup for Covalent Complex (OpenBabel-prepared MOL2, ACPYPE -c rc)

Workflow:
1. Prepare ligand MOL2 (prefer OpenBabel: add Hs, gen3d, Gasteiger charges)
2. Parameterize ligand with ACPYPE/GAFF2 (reads charges, avoids sqm)
3. Prepare protein topology (pdb2gmx)
4. Combine coordinates
5. Solvate and add ions
6. Generate MDP files

Requirements:
    - GROMACS (gmx)
    - acpype (pip install acpype)
    - OpenBabel (conda/pip; provides 'obabel' CLI). RDKit is optional fallback.

Usage:
    python setup_covalent_gromacs.py \
        --complex covalent_complex.pdb \
        --ligand docking_results/best_pose.sdf \
        --cys_chain A --cys_resid 1039 \
        --output md_setup
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path

# Optional RDKit fallback
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False


def which(cmd):
    from shutil import which as _which
    return _which(cmd)


def check_dependencies():
    """Check required tools are available."""
    tools = {
        'gmx': 'GROMACS',
        'acpype': 'acpype (pip install acpype)',
    }
    missing = []
    for cmd, name in tools.items():
        if which(cmd) is None:
            missing.append(name)
    if missing:
        print("WARNING: Missing tools:")
        for t in missing:
            print(f"  - {t}")
        print("\nWill attempt to proceed, but may fail.")
    return len(missing) == 0


def prepare_ligand_mol2(input_path: Path, output_path: Path) -> Path:
    """
    Prepare ligand MOL2 suitable for ACPYPE:
    - Prefer OpenBabel: add Hs, generate 3D, assign Gasteiger charges, write MOL2
    - Fallback to RDKit if 'obabel' not available
    Returns the path to the written MOL2.
    """
    print("\n  Preparing ligand MOL2 for ACPYPE...")
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if which("obabel"):
        # Use OpenBabel to ensure Tripos MOL2 with SUBSTRUCTURE and charges
        # --addhydrogens, --gen3d, --partialcharge gasteiger
        cmd = [
            "obabel",
            str(input_path),
            "-O", str(output_path),
            "--addhydrogens",
            "--gen3d",
            "--partialcharge", "gasteiger",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print("  OpenBabel failed to generate MOL2. stderr:")
            print(res.stderr)
            if not HAS_RDKIT:
                raise RuntimeError("Neither OpenBabel nor RDKit could prepare MOL2.")
        else:
            print(f"  OpenBabel MOL2 written to: {output_path}")
            return output_path

    # Fallback: RDKit
    if not HAS_RDKIT:
        raise RuntimeError("RDKit not installed and OpenBabel conversion failed.")

    suffix = input_path.suffix.lower()
    mol = None
    if suffix == ".sdf":
        suppl = Chem.SDMolSupplier(str(input_path), sanitize=True)
        for m in suppl:
            if m is not None:
                mol = m
                break
    elif suffix in (".mol", ".mol2"):
        mol = Chem.MolFromMolFile(str(input_path), sanitize=True)
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(input_path), sanitize=True)
    else:
        raise ValueError(f"Unsupported ligand format: {input_path.suffix}. Use SDF/MOL2/PDB.")

    if mol is None:
        raise ValueError(f"RDKit could not read ligand: {input_path}")

    mol = Chem.AddHs(mol)

    # Embed and minimize
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        status = AllChem.EmbedMolecule(mol, randomSeed=42)
        if status != 0:
            raise RuntimeError("RDKit embedding failed")
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
        if props is not None:
            AllChem.MMFFOptimizeMolecule(mol, props=props, maxIters=500)
        else:
            raise ValueError("MMFF properties not available")
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)

    # Write MOL2; if ACPYPE still complains, we will try passing through obabel for formatting
    Chem.MolToMolFile(mol, str(output_path))
    print(f"  RDKit MOL2 written to: {output_path}")

    # If OpenBabel is available, re-write via obabel to ensure Tripos formatting with SUBSTRUCTURE
    if which("obabel"):
        tmp_mol2 = output_path.parent / "ligand_rdkit_tmp.mol2"
        shutil.copy(output_path, tmp_mol2)
        cmd = ["obabel", str(tmp_mol2), "-O", str(output_path)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print("  Reformatted MOL2 via OpenBabel for ACPYPE compatibility.")
        else:
            print("  WARNING: OpenBabel reformat failed; proceeding with RDKit MOL2.")
        tmp_mol2.unlink(missing_ok=True)

    return output_path

def parameterize_ligand_acpype(ligand_mol2, output_dir, net_charge=0, use_user_charges=True):
    """
    Parameterize ligand using ACPYPE (GAFF2).
    - Runs ACPYPE in output_dir (cwd)
    - If use_user_charges=True, use -c user to read charges from MOL2 (avoids sqm).
    Returns paths to ligand topology files if successful, else None.
    """
    print("\n  Running acpype for ligand parameterization...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ligand_in = Path(ligand_mol2).resolve()
    if not ligand_in.exists():
        print(f"  ERROR: Ligand MOL2 not found: {ligand_in}")
        return None

    charge_method = "user" if use_user_charges else "bcc"
    cmd = [
        'acpype',
        '-i', str(ligand_in),
        '-c', charge_method,   # 'user' reads charges from MOL2; 'bcc' runs AM1-BCC
        '-n', str(net_charge),
        '-o', 'gmx',
        '-b', 'LIG',
        '-a', 'gaff2',
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)
    print("  ACPYPE exit code:", res.returncode)
    if res.stdout.strip():
        print("  ACPYPE stdout:\n", res.stdout.strip())
    if res.stderr.strip():
        print("  ACPYPE stderr:\n", res.stderr.strip())

    acpype_dir = output_dir / "LIG.acpype"
    itp = acpype_dir / "LIG_GMX.itp"
    gro = acpype_dir / "LIG_GMX.gro"
    top = acpype_dir / "LIG_GMX.top"
    log = acpype_dir / "acpype.log"

    if res.returncode != 0 or not (itp.exists() and gro.exists() and top.exists()):
        if log.exists():
            try:
                tail = "\n".join(log.read_text().splitlines()[-50:])
                print("\n  acpype.log (tail):\n", tail)
            except Exception:
                pass
        else:
            print("  No acpype.log found.")
        if charge_method == "bcc":
            # Help diagnose AmberTools environment if using AM1-BCC
            for prog in ["antechamber", "parmchk2", "sqm"]:
                p = subprocess.run(["which", prog], capture_output=True, text=True)
                if p.returncode != 0:
                    print(f"  WARNING: Missing AmberTools program: {prog}")
        print("\n  ERROR: ACPYPE did not produce LIG_GMX.itp/gro/top in", acpype_dir)
        return None

    return {'itp': itp, 'gro': gro, 'top': top}



def prepare_protein_topology(complex_pdb, output_dir, cys_chain, cys_resid, ff='amber99sb-ildn'):
    """
    Prepare protein topology with pdb2gmx.
    Writes outputs inside output_dir (cwd used); pass only basenames to gmx.
    """
    print("\n  Running pdb2gmx for protein...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract protein-only PDB into md_setup
    protein_pdb = output_dir / "protein_only.pdb"
    complex_pdb = Path(complex_pdb).resolve()
    with open(complex_pdb, 'r') as f_in, open(protein_pdb, 'w') as f_out:
        for line in f_in:
            if line.startswith('ATOM'):
                f_out.write(line)
            elif line.startswith('TER'):
                f_out.write(line)
        f_out.write("END\n")

    # Run pdb2gmx with cwd=output_dir, pass basenames only
    cmd = [
        'gmx', 'pdb2gmx',
        '-f', 'protein_only.pdb',
        '-o', 'protein.gro',
        '-p', 'protein.top',
        '-i', 'protein_posre.itp',
        '-ff', ff,
        '-water', 'tip3p',
        '-ignh',
        '-his',
    ]

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=output_dir,
        input='0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
    )
    if res.returncode != 0:
        print(f"  pdb2gmx stderr:\n{res.stderr}")
        try:
            cmd.remove('-his')
        except ValueError:
            pass
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)
        if res.returncode != 0:
            print(f"  pdb2gmx stderr (retry):\n{res.stderr}")
            raise RuntimeError("pdb2gmx failed")

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
            except Exception:
                continue
    return None


def combine_coordinates(protein_gro, ligand_gro, output_gro):
    """Combine protein and ligand GRO files."""
    print("\n  Combining coordinates...")

    if not Path(protein_gro).exists():
        raise FileNotFoundError(f"Protein GRO missing: {protein_gro}")
    if not Path(ligand_gro).exists():
        raise FileNotFoundError(f"Ligand GRO missing: {ligand_gro}")

    with open(protein_gro, 'r') as f:
        prot_lines = f.readlines()
    with open(ligand_gro, 'r') as f:
        lig_lines = f.readlines()

    prot_natoms = int(prot_lines[1].strip())
    prot_atoms = prot_lines[2:2+prot_natoms]
    prot_box = prot_lines[-1]

    lig_natoms = int(lig_lines[1].strip())
    lig_atoms = lig_lines[2:2+lig_natoms]

    new_lig_atoms = []
    for line in lig_atoms:
        new_line = f"{prot_natoms+1:5d}{'LIG':>5s}{line[10:]}"
        new_lig_atoms.append(new_line)

    total_atoms = prot_natoms + lig_natoms

    with open(output_gro, 'w') as f:
        f.write("Covalent complex\n")
        f.write(f"{total_atoms}\n")
        for line in prot_atoms:
            f.write(line)
        for line in new_lig_atoms:
            f.write(line)
        f.write(prot_box)

    return output_gro


def create_mdp_files(output_dir):
    """Create MDP files for minimization, NVT, NPT, and production."""
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
    nvt_mdp = """; NVT equilibration
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
"""
    npt_mdp = """; NPT equilibration
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
"""
    prod_mdp = """; Production MD - 20 ns NPT
integrator  = md
nsteps      = 10000000
dt          = 0.002

nstxout     = 0
nstvout     = 0
nstfout     = 0
nstxout-compressed = 5000
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
    output_dir = Path(output_dir)
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
    script_path = Path(output_dir) / 'run_simulation.sh'
    script_path.write_text(script)
    script_path.chmod(0o755)
    print(f"  Created: run_simulation.sh")


def solvate_and_ionize(gro_file, top_file, output_dir):
    """Add water box and ions. Run all tools in output_dir and pass basenames."""
    print("\n  Solvating system...")

    output_dir = Path(output_dir)

    if not Path(gro_file).exists():
        raise FileNotFoundError(f"Input GRO not found: {gro_file}")
    if not Path(top_file).exists():
        raise FileNotFoundError(f"Input TOP not found: {top_file}")

    gro_local = output_dir / 'complex.gro'
    if Path(gro_file).resolve() != gro_local.resolve():
        shutil.copy(gro_file, gro_local)
    else:
        gro_local = Path(gro_file)

    top_local = output_dir / 'topol.top'
    if Path(top_file).resolve() != top_local.resolve():
        shutil.copy(top_file, top_local)
    else:
        top_local = Path(top_file)

    cmd = ['gmx', 'editconf', '-f', 'complex.gro', '-o', 'complex_box.gro',
           '-c', '-d', '1.2', '-bt', 'dodecahedron']
    subprocess.run(cmd, capture_output=True, cwd=output_dir)

    cmd = ['gmx', 'solvate', '-cp', 'complex_box.gro', '-cs', 'spc216.gro',
           '-o', 'complex_solv.gro', '-p', 'topol.top']
    subprocess.run(cmd, capture_output=True, cwd=output_dir)

    (output_dir / 'ions.mdp').write_text("; Ions\n")

    cmd = ['gmx', 'grompp', '-f', 'ions.mdp', '-c', 'complex_solv.gro',
           '-p', 'topol.top', '-o', 'ions.tpr', '-maxwarn', '5']
    subprocess.run(cmd, capture_output=True, cwd=output_dir)

    cmd = ['gmx', 'genion', '-s', 'ions.tpr', '-o', 'complex_solv_ions.gro',
           '-p', 'topol.top', '-pname', 'NA', '-nname', 'CL', '-neutral']
    subprocess.run(cmd, input='SOL\n', capture_output=True, text=True, cwd=output_dir)

    return output_dir / 'complex_solv_ions.gro'


def main():
    parser = argparse.ArgumentParser(
        description="Setup GROMACS simulation for covalent complex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python %(prog)s \\
        --complex covalent_complex.pdb \\
        --ligand docking_results/best_pose.sdf \\
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
                        help='Ligand SDF/MOL2/PDB input')
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

    check_dependencies()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    # Step 1: Prepare ligand MOL2 and parameterize
    print("\n[1] Preparing and parameterizing ligand...")
    mol2_path = outdir / 'ligand_prepared.mol2'
    mol2_path = prepare_ligand_mol2(Path(args.ligand), mol2_path)

    ligand_files = parameterize_ligand_acpype(
        mol2_path, outdir, net_charge=args.charge, use_user_charges=True
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
    print("    NOTE: Check ligand.itp to find the carbon atom that will bond to SG")

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

    # Step 7: Solvate
    print("\n[7] Solvating system...")
    try:
        topol = outdir / 'topol.top'
        shutil.copy(protein_files['top'], topol)
        with open(topol, 'r') as f:
            content = f.read()
        lines = content.split('\n')
        new_lines = []
        ligand_included = False
        for line in lines:
            new_lines.append(line)
            if '#include' in line and 'forcefield' in line and not ligand_included:
                new_lines.append('')
                new_lines.append('; Include ligand topology')
                new_lines.append('#include "ligand.itp"')
                ligand_included = True
        content = '\n'.join(new_lines)
        content = content.rstrip() + '\nLIG              1\n'
        with open(topol, 'w') as f:
            f.write(content)

        shutil.copy(ligand_files['itp'], outdir / 'ligand.itp')

        solvate_and_ionize(complex_gro, topol, outdir)
    except Exception as e:
        print(f"  Solvation failed: {e}")
        print("  You may need to solvate manually")

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {outdir}/")
    print(f"\nIMPORTANT - Post-parameterization covalent corrections:")
    print(f"  1) Identify in ligand.itp the carbon atom index that will bond to Cys SG (index {sg_index}).")
    print(f"  2) Add in topol.top:")
    print(f"     [ intermolecular_interactions ]")
    print(f"     [ bonds ]")
    print(f"     ; ai   aj   funct   r0(nm)   fc")
    print(f"     {sg_index}    XXX   1       0.1810   238000")
    print(f"     Replace XXX with the ligand carbon index.")
    print(f"  3) Remove the replaced by the covalent bond:")
    print(f"     - Delete that H from complex.gro and renumber atoms.")
    print(f"     - Remove that H from ligand.itp: delete its atom entry and any bonds/angles/dihedrals with it, then renumber.")
    print(f"  4) gmx grompp for EM should run without missing atom/bond errors. Visualize the structure to confirm geometry.")
    print(f"  5) If you prefer AM1-BCC charges later, fix AmberTools (antechamber/sqm) and rerun ACPYPE with -c bcc on the same MOL2.")
    print(f"     Ensure your conda env has ambertools, libtinfo, libgfortran, llvm-openmp from conda-forge.")

if __name__ == "__main__":
    main()
