#!/usr/bin/env python3
"""
Ligand Parameterization for GROMACS

Prepares a docked ligand for MD simulations using GAFF2 force field with
AM1-BCC charges. Optionally performs QM geometry optimization with xtb.

Workflow:
1. Convert SDF to MOL2 (with hydrogens if needed)
2. Run antechamber to assign AM1-BCC charges and GAFF2 atom types
3. Run parmchk2 to check/generate missing parameters
4. (Optional) Run xtb for semi-empirical QM geometry optimization
5. Run acpype to convert AMBER parameters to GROMACS format

Dependencies:
    - AmberTools (antechamber, parmchk2)
    - Open Babel (obabel)
    - acpype
    - xtb (optional, for QM optimization)

Usage:
    python 04_parameterize_ligand.py --ligand DOCKED.sdf --output_dir OUTPUT [options]

Examples:
    # Basic parameterization
    python 04_parameterize_ligand.py \\
        --ligand Outputs/NonCovalent/docking/4CXA_Inhib_42_best.sdf \\
        --output_dir Outputs/NonCovalent/params/Inhib_42

    # With xtb optimization
    python 04_parameterize_ligand.py \\
        --ligand Outputs/NonCovalent/docking/4CXA_Inhib_42_best.sdf \\
        --output_dir Outputs/NonCovalent/params/Inhib_42 \\
        --xtb_optimize

    # Custom residue name and charge
    python 04_parameterize_ligand.py \\
        --ligand ligand.sdf \\
        --output_dir params \\
        --resname LIG \\
        --charge 0
"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, description, cwd=None, verbose=True):
    """Execute a shell command and handle errors."""
    if verbose:
        print(f"  {description}...")
        if isinstance(cmd, list):
            print(f"    $ {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
        shell=isinstance(cmd, str)
    )

    if result.returncode != 0:
        print(f"  ERROR: {description} failed!")
        print(f"  STDOUT: {result.stdout}")
        print(f"  STDERR: {result.stderr}")
        return False, result

    if verbose:
        print(f"  Done")
    return True, result


def check_dependencies():
    """Check that required tools are available."""
    required = {
        'antechamber': 'AmberTools',
        'parmchk2': 'AmberTools',
        'obabel': 'Open Babel',
        'acpype': 'acpype'
    }

    missing = []
    for cmd, package in required.items():
        if shutil.which(cmd) is None:
            missing.append(f"{cmd} ({package})")

    if missing:
        print("ERROR: Missing required dependencies:")
        for m in missing:
            print(f"  - {m}")
        return False
    return True


def get_net_charge_from_sdf(sdf_path):
    """
    Attempt to determine net charge from SDF file.
    Returns 0 if cannot be determined.
    """
    try:
        # Try using Open Babel to get formal charge
        result = subprocess.run(
            ['obabel', str(sdf_path), '-osmi', '--append', 'charge'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            # Parse charge from output
            for line in result.stdout.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return int(float(parts[-1]))
                    except ValueError:
                        pass
    except Exception:
        pass
    return 0


def convert_sdf_to_mol2(sdf_path, mol2_path, add_hydrogens=True):
    """Convert SDF to MOL2 format using Open Babel."""
    cmd = ['obabel', str(sdf_path), '-O', str(mol2_path)]
    if add_hydrogens:
        cmd.append('-h')

    return run_command(cmd, "Converting SDF to MOL2")


def run_antechamber(mol2_input, mol2_output, resname='LIG', charge=0,
                    charge_method='bcc', atom_type='gaff2'):
    """
    Run antechamber to assign charges and atom types.

    Args:
        mol2_input: Input MOL2 file
        mol2_output: Output MOL2 file with charges
        resname: Residue name (max 3 chars)
        charge: Net charge of molecule
        charge_method: Charge method (bcc = AM1-BCC, gas = Gasteiger)
        atom_type: Atom type (gaff or gaff2)
    """
    mol2_input = Path(mol2_input)
    mol2_output = Path(mol2_output)
    work_dir = mol2_output.parent

    # Use filenames only since we run in the output directory
    cmd = [
        'antechamber',
        '-i', mol2_input.name,
        '-fi', 'mol2',
        '-o', mol2_output.name,
        '-fo', 'mol2',
        '-c', charge_method,
        '-at', atom_type,
        '-rn', resname[:3],
        '-nc', str(charge),
        '-pf', 'y'  # Remove intermediate files
    ]

    return run_command(cmd, f"Running antechamber (charges={charge_method}, types={atom_type})",
                       cwd=str(work_dir))


def run_parmchk2(mol2_input, frcmod_output, atom_type='gaff2'):
    """
    Run parmchk2 to check/generate missing parameters.

    Args:
        mol2_input: Input MOL2 file with atom types
        frcmod_output: Output frcmod file with missing parameters
        atom_type: Force field type (gaff or gaff2)
    """
    mol2_input = Path(mol2_input)
    frcmod_output = Path(frcmod_output)
    work_dir = frcmod_output.parent

    cmd = [
        'parmchk2',
        '-i', mol2_input.name,
        '-f', 'mol2',
        '-o', frcmod_output.name,
        '-s', atom_type
    ]

    return run_command(cmd, "Running parmchk2 to check parameters", cwd=str(work_dir))


def run_xtb_optimization(mol2_input, mol2_output, charge=0, n_cores=4):
    """
    Run xtb semi-empirical QM geometry optimization.

    Args:
        mol2_input: Input MOL2 file
        mol2_output: Output optimized MOL2 file
        charge: Net charge
        n_cores: Number of CPU cores
    """
    if shutil.which('xtb') is None:
        print("  WARNING: xtb not found, skipping QM optimization")
        shutil.copy(mol2_input, mol2_output)
        return True, None

    work_dir = Path(mol2_input).parent

    # Convert MOL2 to XYZ for xtb
    xyz_input = work_dir / 'xtb_input.xyz'
    success, _ = run_command(
        ['obabel', str(mol2_input), '-O', str(xyz_input)],
        "Converting to XYZ for xtb"
    )
    if not success:
        return False, None

    # Run xtb optimization (use filename only since we run in work_dir)
    cmd = [
        'xtb',
        xyz_input.name,
        '--opt',
        '--chrg', str(charge),
        '-P', str(n_cores)
    ]

    success, result = run_command(cmd, "Running xtb geometry optimization", cwd=str(work_dir))
    if not success:
        return False, result

    # Convert optimized XYZ back to MOL2
    xyz_output = work_dir / 'xtbopt.xyz'
    if xyz_output.exists():
        success, _ = run_command(
            ['obabel', str(xyz_output), '-O', str(mol2_output)],
            "Converting optimized geometry to MOL2"
        )
        if not success:
            return False, None
    else:
        print("  WARNING: xtb optimization output not found, using input geometry")
        shutil.copy(mol2_input, mol2_output)

    # Clean up xtb files
    for f in ['xtb_input.xyz', 'xtbopt.xyz', 'xtbopt.log', 'xtbrestart',
              'wbo', 'charges', 'xtbtopo.mol', '.xtboptok']:
        (work_dir / f).unlink(missing_ok=True)

    return True, result


def run_acpype(mol2_input, output_dir, resname='LIG', charge=0, atom_type='gaff2'):
    """
    Run acpype to convert AMBER parameters to GROMACS format.

    Args:
        mol2_input: Input MOL2 file with charges
        output_dir: Output directory for GROMACS files
        resname: Residue name
        charge: Net charge
        atom_type: Atom type (gaff or gaff2)
    """
    mol2_input = Path(mol2_input).resolve()
    output_dir = Path(output_dir).resolve()

    cmd = [
        'acpype',
        '-i', str(mol2_input),
        '-b', resname,
        '-n', str(charge),
        '-a', atom_type,
        '-c', 'user'  # Use charges from input MOL2
    ]

    success, result = run_command(cmd, "Running acpype to generate GROMACS topology", cwd=str(output_dir))

    return success, result


def organize_acpype_outputs(output_dir, resname='LIG'):
    """Extract GROMACS files from acpype output and clean up."""
    output_dir = Path(output_dir)

    # Find acpype output directory
    acpype_dir = None
    for d in output_dir.iterdir():
        if d.is_dir() and d.name.endswith('.acpype'):
            acpype_dir = d
            break

    if acpype_dir is None:
        print("  WARNING: Could not find acpype output directory")
        return False

    # Copy key files to output directory
    file_mappings = [
        (f'{resname}_GMX.itp', f'{resname}.itp'),
        (f'{resname}_GMX.gro', f'{resname}.gro'),
        (f'{resname}_GMX.top', f'{resname}.top'),
    ]

    for src_name, dst_name in file_mappings:
        src = acpype_dir / src_name
        if src.exists():
            shutil.copy(src, output_dir / dst_name)
            print(f"  Created: {dst_name}")

    # Also copy posre if it exists
    posre = acpype_dir / f'posre_{resname}.itp'
    if posre.exists():
        shutil.copy(posre, output_dir / f'posre_{resname}.itp')

    # Clean up acpype directory
    shutil.rmtree(acpype_dir)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Parameterize ligand for GROMACS MD simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument('--ligand', '-l', required=True,
                        help='Input ligand file (SDF or MOL2)')
    parser.add_argument('--output_dir', '-o', required=True,
                        help='Output directory for parameterized ligand')

    # Optional arguments
    parser.add_argument('--resname', '-r', default='LIG',
                        help='Residue name for ligand (default: LIG, max 3 chars)')
    parser.add_argument('--charge', '-c', type=int, default=None,
                        help='Net charge (auto-detected if not specified)')
    parser.add_argument('--charge_method', default='bcc',
                        choices=['bcc', 'gas', 'mul'],
                        help='Charge method: bcc (AM1-BCC), gas (Gasteiger), mul (Mulliken)')
    parser.add_argument('--atom_type', default='gaff2',
                        choices=['gaff', 'gaff2'],
                        help='Atom type force field (default: gaff2)')

    # XTB optimization
    parser.add_argument('--xtb_optimize', action='store_true',
                        help='Run xtb QM geometry optimization')
    parser.add_argument('--xtb_cores', type=int, default=4,
                        help='Number of cores for xtb (default: 4)')

    # Other options
    parser.add_argument('--keep_intermediates', action='store_true',
                        help='Keep intermediate files')

    args = parser.parse_args()

    # Validate inputs
    ligand_path = Path(args.ligand)
    if not ligand_path.exists():
        print(f"ERROR: Ligand file not found: {args.ligand}")
        sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Truncate residue name to 3 characters
    resname = args.resname[:3].upper()

    # Determine charge
    if args.charge is not None:
        charge = args.charge
    else:
        charge = get_net_charge_from_sdf(ligand_path)
        print(f"  Auto-detected charge: {charge}")

    print("\n" + "="*60)
    print("LIGAND PARAMETERIZATION")
    print("="*60)
    print(f"\nInputs:")
    print(f"  Ligand:       {ligand_path}")
    print(f"  Residue name: {resname}")
    print(f"  Net charge:   {charge}")
    print(f"  Atom types:   {args.atom_type}")
    print(f"  Charge method:{args.charge_method}")
    print(f"  XTB optimize: {args.xtb_optimize}")
    print(f"\nOutput:")
    print(f"  Directory:    {output_dir}")

    # Step 1: Convert to MOL2 if needed
    print("\n" + "-"*60)
    print("Step 1: Prepare input structure")
    print("-"*60)

    mol2_input = output_dir / f'{resname}_input.mol2'
    if ligand_path.suffix.lower() == '.mol2':
        shutil.copy(ligand_path, mol2_input)
    else:
        success, _ = convert_sdf_to_mol2(ligand_path, mol2_input)
        if not success:
            print("ERROR: Failed to convert ligand to MOL2")
            sys.exit(1)

    # Step 2: Optional xtb optimization
    if args.xtb_optimize:
        print("\n" + "-"*60)
        print("Step 2: XTB geometry optimization")
        print("-"*60)

        mol2_optimized = output_dir / f'{resname}_xtb.mol2'
        success, _ = run_xtb_optimization(mol2_input, mol2_optimized, charge, args.xtb_cores)
        if not success:
            print("WARNING: XTB optimization failed, using original geometry")
            mol2_optimized = mol2_input
        mol2_for_antechamber = mol2_optimized
    else:
        print("\n" + "-"*60)
        print("Step 2: Skipping XTB optimization (use --xtb_optimize to enable)")
        print("-"*60)
        mol2_for_antechamber = mol2_input

    # Step 3: Run antechamber
    print("\n" + "-"*60)
    print("Step 3: Antechamber - assign charges and atom types")
    print("-"*60)

    mol2_charged = output_dir / f'{resname}_charged.mol2'
    success, _ = run_antechamber(
        mol2_for_antechamber, mol2_charged,
        resname=resname, charge=charge,
        charge_method=args.charge_method, atom_type=args.atom_type
    )
    if not success:
        print("ERROR: Antechamber failed")
        sys.exit(1)

    # Step 4: Run parmchk2
    print("\n" + "-"*60)
    print("Step 4: Parmchk2 - check/generate missing parameters")
    print("-"*60)

    frcmod = output_dir / f'{resname}.frcmod'
    success, _ = run_parmchk2(mol2_charged, frcmod, atom_type=args.atom_type)
    if not success:
        print("ERROR: parmchk2 failed")
        sys.exit(1)

    # Check for missing parameters
    with open(frcmod) as f:
        frcmod_content = f.read()
        if 'ATTN' in frcmod_content:
            print("  WARNING: Some parameters may be estimated (check frcmod file)")

    # Step 5: Run acpype
    print("\n" + "-"*60)
    print("Step 5: Acpype - convert to GROMACS format")
    print("-"*60)

    success, _ = run_acpype(mol2_charged, output_dir, resname=resname, charge=charge, atom_type=args.atom_type)
    if not success:
        print("ERROR: acpype failed")
        sys.exit(1)

    # Step 6: Organize outputs
    print("\n" + "-"*60)
    print("Step 6: Organize output files")
    print("-"*60)

    success = organize_acpype_outputs(output_dir, resname)
    if not success:
        print("WARNING: Could not organize all output files")

    # Clean up intermediate files
    if not args.keep_intermediates:
        for f in [f'{resname}_input.mol2', f'{resname}_xtb.mol2']:
            (output_dir / f).unlink(missing_ok=True)
        # Clean up antechamber temp files
        for f in output_dir.glob('ANTECHAMBER*'):
            f.unlink(missing_ok=True)
        for f in output_dir.glob('ATOMTYPE*'):
            f.unlink(missing_ok=True)
        (output_dir / 'sqm.in').unlink(missing_ok=True)
        (output_dir / 'sqm.out').unlink(missing_ok=True)
        (output_dir / 'sqm.pdb').unlink(missing_ok=True)

    # Summary
    print("\n" + "="*60)
    print("PARAMETERIZATION COMPLETE")
    print("="*60)
    print(f"\nOutput files in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name}")

    print(f"\nKey files for GROMACS:")
    print(f"  Topology:    {output_dir / f'{resname}.itp'}")
    print(f"  Structure:   {output_dir / f'{resname}.gro'}")
    print(f"  MOL2:        {output_dir / f'{resname}_charged.mol2'}")

    print(f"\nNext step:")
    print(f"  python 05_setup_complex.py \\")
    print(f"      --receptor RECEPTOR.pdb \\")
    print(f"      --ligand_itp {output_dir / f'{resname}.itp'} \\")
    print(f"      --ligand_gro {output_dir / f'{resname}.gro'} \\")
    print(f"      --output_dir OUTPUT_DIR")
    print()


if __name__ == "__main__":
    main()
