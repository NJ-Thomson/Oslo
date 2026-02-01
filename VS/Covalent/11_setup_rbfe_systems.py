#!/usr/bin/env python3
"""
05_setup_rbfe_systems.py

Set up MD systems for RBFE calculations from prepared poses.

Takes output from 04_prepare_rbfe_poses.py and:
1. Runs b05 to create CYL residues (CYA for acry, CYC for chlo)
2. Runs b06 to assemble complexes with CYL residues
3. Runs pdb2gmx, solvation, and energy minimization

Outputs 4 energy-minimized systems ready for setup_rbfe_pmx.py

Usage:
    python 05_setup_rbfe_systems.py \\
        --prep_dir Outputs/RBFE_prep \\
        --cys_resid 1039 \\
        --output_dir Outputs/RBFE_systems
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_gmx() -> str:
    """Find GROMACS executable."""
    for cmd in ['gmx', 'gmx_mpi']:
        if shutil.which(cmd):
            return cmd
    raise RuntimeError("GROMACS not found in PATH")


def run_cmd(cmd, cwd=None, desc=None):
    """Run command and check for errors."""
    if desc:
        print(f"    {desc}")
    result = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd,
                           capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[:500]}")
        raise RuntimeError(f"Command failed: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    return result


def run_b05(adduct_mol2: Path, acpype_itp: Path, resname: str, output_dir: Path,
            skip_ff_copy: bool = False) -> Path:
    """Run b05 to create CYL residue."""
    script = Path(__file__).parent / "b05_create_cyl_residue.py"

    cmd = [
        sys.executable, str(script),
        "--adduct-mol2", str(adduct_mol2),
        "--residue-name", resname,
        "--output-dir", str(output_dir),
        "--acpype-itp", str(acpype_itp)
    ]
    if skip_ff_copy:
        cmd.append("--skip-ff-copy")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    b05 output: {result.stdout}")
        print(f"    b05 error: {result.stderr}")
        raise RuntimeError("b05 failed")

    # Return path to metadata
    return output_dir / f"{resname.lower()}_residue" / f"{resname.lower()}_meta.json"


def run_b06(complex_pdb: Path, adduct_mol2: Path, cyl_meta: Path,
            cys_resid: int, cyl_resname: str, output_pdb: Path):
    """Run b06 to assemble complex with CYL residue."""
    script = Path(__file__).parent / "b06_assemble_covalent_complex.py"

    cmd = [
        sys.executable, str(script),
        "--complex", str(complex_pdb),
        "--ligand-mol2", str(adduct_mol2),
        "--cyl-meta", str(cyl_meta),
        "--cys-resid", str(cys_resid),
        "--cyl-resname", cyl_resname,
        "--output", str(output_pdb)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    b06 error: {result.stderr}")
        raise RuntimeError("b06 failed")


def setup_md_system(complex_pdb: Path, output_dir: Path, ff_dir: Path,
                    gmx: str, box_size: float = 1.2) -> Path:
    """
    Set up MD system: pdb2gmx → editconf → solvate → genion → em.

    Returns path to energy-minimized structure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy force field to working directory
    ff_name = ff_dir.stem
    local_ff = output_dir / ff_dir.name
    if not local_ff.exists():
        shutil.copytree(ff_dir, local_ff)

    # Also copy residuetypes.dat
    restypes_src = ff_dir.parent / "residuetypes.dat"
    restypes_dst = output_dir / "residuetypes.dat"
    if restypes_src.exists() and not restypes_dst.exists():
        shutil.copy(restypes_src, restypes_dst)

    # Set GMXLIB
    env = os.environ.copy()
    env['GMXLIB'] = str(output_dir)

    # pdb2gmx
    run_cmd(
        f"{gmx} pdb2gmx -f {complex_pdb} -o protein.gro -p topol.top "
        f"-ff {ff_name} -water tip3p -ignh",
        cwd=output_dir, desc="Running pdb2gmx..."
    )

    # editconf - create box
    run_cmd(
        f"{gmx} editconf -f protein.gro -o boxed.gro -c -d {box_size} -bt dodecahedron",
        cwd=output_dir, desc="Creating box..."
    )

    # solvate
    run_cmd(
        f"{gmx} solvate -cp boxed.gro -cs spc216.gro -o solvated.gro -p topol.top",
        cwd=output_dir, desc="Solvating..."
    )

    # Create ions.mdp
    ions_mdp = output_dir / "ions.mdp"
    ions_mdp.write_text("integrator = steep\nnsteps = 0\n")

    # grompp for genion
    run_cmd(
        f"{gmx} grompp -f ions.mdp -c solvated.gro -p topol.top -o ions.tpr -maxwarn 2",
        cwd=output_dir, desc="Preparing for ions..."
    )

    # genion - neutralize with NaCl
    result = subprocess.run(
        f"echo SOL | {gmx} genion -s ions.tpr -o ionized.gro -p topol.top -pname NA -nname CL -neutral -conc 0.15",
        shell=True, cwd=output_dir, capture_output=True, text=True
    )

    # Create em.mdp
    em_mdp = output_dir / "em.mdp"
    em_mdp.write_text("""integrator  = steep
emtol       = 100.0
emstep      = 0.01
nsteps      = 50000
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
""")

    # grompp for EM
    run_cmd(
        f"{gmx} grompp -f em.mdp -c ionized.gro -p topol.top -o em.tpr -maxwarn 2",
        cwd=output_dir, desc="Preparing for EM..."
    )

    # Run EM
    run_cmd(
        f"{gmx} mdrun -v -deffnm em -ntmpi 1",
        cwd=output_dir, desc="Running energy minimization..."
    )

    return output_dir / "em.gro"


def main():
    parser = argparse.ArgumentParser(
        description='Set up MD systems for RBFE from prepared poses'
    )
    parser.add_argument('--prep_dir', required=True,
                        help='Output directory from 04_prepare_rbfe_poses.py')
    parser.add_argument('--cys_resid', type=int, required=True,
                        help='Covalent cysteine residue ID')
    parser.add_argument('--output_dir', default='RBFE_systems',
                        help='Output directory for MD systems')
    parser.add_argument('--box_size', type=float, default=1.2,
                        help='Box padding in nm (default: 1.2)')
    parser.add_argument('--acry_resname', default='CYA',
                        help='Residue name for acrylamide CYL (default: CYA)')
    parser.add_argument('--chlo_resname', default='CYC',
                        help='Residue name for chloroacetamide CYL (default: CYC)')
    args = parser.parse_args()

    prep_dir = Path(args.prep_dir)
    output_dir = Path(args.output_dir)

    # Load metadata from prep step
    metadata_file = prep_dir / "rbfe_prep_metadata.json"
    if not metadata_file.exists():
        print(f"ERROR: Metadata not found: {metadata_file}")
        print("Run 04_prepare_rbfe_poses.py first")
        return 1

    with open(metadata_file) as f:
        metadata = json.load(f)

    # Check parameterization exists
    param_file = prep_dir / "parameterization.json"
    if not param_file.exists():
        print(f"ERROR: Parameterization not found: {param_file}")
        print("Run 04_prepare_rbfe_poses.py without --skip_param")
        return 1

    with open(param_file) as f:
        params = json.load(f)

    gmx = find_gmx()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RBFE SYSTEM SETUP")
    print("=" * 70)
    print(f"\nPrep dir:    {prep_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"CYS residue: {args.cys_resid}")
    print(f"GROMACS:     {gmx}")

    # Step 1: Create CYL residues with b05
    print("\n" + "=" * 70)
    print("STEP 1: Creating CYL residues (b05)")
    print("=" * 70)

    md_prep_dir = output_dir / "md_prep"
    md_prep_dir.mkdir(exist_ok=True)

    cyl_meta = {}

    # Acrylamide CYL
    print(f"\n  Creating {args.acry_resname} residue...")
    acry_mol2 = Path(params['acry']['mol2'])
    acry_itp = Path(params['acry']['itp'])
    cyl_meta['acry'] = run_b05(acry_mol2, acry_itp, args.acry_resname, md_prep_dir)
    print(f"    Created: {cyl_meta['acry']}")

    # Chloroacetamide CYL (skip ff copy - reuse same ff)
    print(f"\n  Creating {args.chlo_resname} residue...")
    chlo_mol2 = Path(params['chlo']['mol2'])
    chlo_itp = Path(params['chlo']['itp'])
    cyl_meta['chlo'] = run_b05(chlo_mol2, chlo_itp, args.chlo_resname, md_prep_dir, skip_ff_copy=True)
    print(f"    Created: {cyl_meta['chlo']}")

    # Step 2: Assemble complexes with b06
    print("\n" + "=" * 70)
    print("STEP 2: Assembling complexes (b06)")
    print("=" * 70)

    complexes_dir = output_dir / "complexes"
    complexes_dir.mkdir(exist_ok=True)

    # Define which systems use which parameters
    systems = {
        'acry_best': {'type': 'acry', 'resname': args.acry_resname},
        'acry_to_chlo': {'type': 'acry', 'resname': args.acry_resname},
        'chlo_best': {'type': 'chlo', 'resname': args.chlo_resname},
        'chlo_to_acry': {'type': 'chlo', 'resname': args.chlo_resname},
    }

    assembled = {}
    for system_name, cfg in systems.items():
        print(f"\n  Assembling {system_name}...")

        input_complex = Path(metadata['selections'][system_name]['complex_pdb'])
        mol2 = Path(params[cfg['type']]['mol2'])
        meta = cyl_meta[cfg['type']]
        output_pdb = complexes_dir / f"complex_{system_name}_{cfg['resname']}.pdb"

        run_b06(input_complex, mol2, meta, args.cys_resid, cfg['resname'], output_pdb)
        assembled[system_name] = output_pdb
        print(f"    Created: {output_pdb.name}")

    # Step 3: Set up MD systems and run EM
    print("\n" + "=" * 70)
    print("STEP 3: Setting up MD systems and energy minimization")
    print("=" * 70)

    # Find the force field directory
    ff_dir = None
    for ff in md_prep_dir.glob("*.ff"):
        ff_dir = ff
        break

    if not ff_dir:
        print("ERROR: Force field not found in md_prep")
        return 1

    em_structures = {}
    for system_name, complex_pdb in assembled.items():
        print(f"\n  Setting up {system_name}...")

        system_dir = output_dir / "systems" / system_name
        em_gro = setup_md_system(complex_pdb, system_dir, ff_dir, gmx, args.box_size)
        em_structures[system_name] = em_gro
        print(f"    EM complete: {em_gro}")

    # Save results
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results = {
        'cys_resid': args.cys_resid,
        'acry_resname': args.acry_resname,
        'chlo_resname': args.chlo_resname,
        'force_field': str(ff_dir),
        'systems': {}
    }

    for name, em_gro in em_structures.items():
        system_dir = em_gro.parent
        results['systems'][name] = {
            'em_gro': str(em_gro),
            'topol': str(system_dir / 'topol.top'),
            'type': 'acry' if 'acry' in name else 'chlo',
            'resname': args.acry_resname if 'acry' in name.split('_')[0] else args.chlo_resname
        }
        print(f"\n  {name}:")
        print(f"    EM structure: {em_gro}")
        print(f"    Topology:     {system_dir / 'topol.top'}")

    with open(output_dir / "rbfe_systems.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("RBFE PAIRS FOR setup_rbfe_pmx.py")
    print("=" * 70)
    print("""
  Pair 1 (ACRY-aligned): acry_best → chlo_to_acry
    python setup_rbfe_pmx.py \\
        --stateA_dir {out}/systems/acry_best \\
        --stateB_dir {out}/systems/chlo_to_acry \\
        --stateA_resname {acry} --stateB_resname {chlo} \\
        --output_dir {out}/RBFE_acry_aligned

  Pair 2 (CHLO-aligned): acry_to_chlo → chlo_best
    python setup_rbfe_pmx.py \\
        --stateA_dir {out}/systems/acry_to_chlo \\
        --stateB_dir {out}/systems/chlo_best \\
        --stateA_resname {acry} --stateB_resname {chlo} \\
        --output_dir {out}/RBFE_chlo_aligned
""".format(out=output_dir, acry=args.acry_resname, chlo=args.chlo_resname))

    return 0


if __name__ == '__main__':
    sys.exit(main())
