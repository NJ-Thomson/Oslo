#!/usr/bin/env python3
"""
Bridge Script: User Prep Pipeline -> Forked Biggin ABFE Engine

Takes pre-parameterised GROMACS systems from the user's pipeline (scripts 04-06)
and restructures them into the format expected by cli-abfe-gmx, then launches
the Snakemake FEP workflow.

Input (from user's prep pipeline):
  - Parameterised ligand files (from 04_parameterize_ligand.py):
      LIG.itp, LIG.gro, LIG_atomtypes.itp, LIG_charged.mol2
  - Complex system (from 05_setup_complex.py):
      complex_em.gro, topol.top, LIG.itp, LIG_atomtypes.itp
  - Equilibrated complex (from 06_test_binding_stability.py, optional):
      npt.gro, prod.gro
  - Per-ligand YAML config (abfe_config.yml)

Output (for cli-abfe-gmx):
  abfe_input/
  ├── <ligand_1>/
  │   ├── ligand/    (ligand-in-water solvated system)
  │   │   ├── ligand.gro
  │   │   └── ligand.top
  │   └── complex/   (protein + ligand solvated system)
  │       ├── complex.gro
  │       └── complex.top
  └── <ligand_2>/
      └── ...

Critical design choice:
  The SAME ligand topology is used for both solvent and complex legs.
  This fixes the charge inconsistency (root cause #1).

Dependencies:
    - GROMACS (gmx)
    - PyYAML
    - Output from scripts 04-06

Usage:
    python 09_launch_abfe.py \\
        --prep_dir Outputs/non_covalent/<receptor> \\
        --config abfe_config.yml \\
        --output_dir Outputs/non_covalent/ABFE/<receptor> \\
        --ligands <ligand_1> <ligand_2>

    # Dry-run (build input structure only, don't launch Snakemake)
    python 09_launch_abfe.py \\
        --prep_dir Outputs/non_covalent/<receptor> \\
        --config abfe_config.yml \\
        --output_dir Outputs/non_covalent/ABFE/<receptor> \\
        --dry_run
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def find_gmx():
    """Find GROMACS executable."""
    for cmd in ['gmx', 'gmx_mpi']:
        try:
            result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                return cmd
        except FileNotFoundError:
            continue
    return None


def run_command(cmd, description, cwd=None, input_text=None):
    """Execute a shell command."""
    print(f"  {description}...")
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=cwd,
        input=input_text, shell=isinstance(cmd, str)
    )
    if result.returncode != 0:
        print(f"  WARNING: {description} returned non-zero")
        if result.stderr:
            print(f"  STDERR: {result.stderr[:500]}")
        return False, result
    return True, result


def load_config(config_path):
    """Load and merge YAML config."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    defaults = raw.get("global_defaults", {})
    ligands_raw = raw.get("ligands", {})

    merged = {}
    for lig_name, lig_overrides in ligands_raw.items():
        cfg = dict(defaults)
        if lig_overrides:
            if "lambda_windows" in defaults and "lambda_windows" in (lig_overrides or {}):
                lw = dict(defaults.get("lambda_windows", {}))
                lw.update(lig_overrides["lambda_windows"])
                cfg.update(lig_overrides)
                cfg["lambda_windows"] = lw
            else:
                cfg.update(lig_overrides)
        merged[lig_name] = cfg
    return merged, defaults


def find_prep_files(prep_dir, ligand_name):
    """
    Locate the user's prep output files for a given ligand.

    Searches common directory layouts:
      prep_dir/params/<ligand>/                  (from 04_parameterize_ligand.py)
      prep_dir/01_params/<ligand>/               (numbered variant)
      prep_dir/complex/<receptor>_<ligand>/      (from 05_setup_complex.py)
      prep_dir/02_setup/<receptor>/<ligand>/pose_*/ (numbered variant)
      prep_dir/stability/<receptor>_<ligand>/    (from 06_test_binding_stability.py)
      prep_dir/03_stability/<receptor>/<ligand>/pose_*/ (numbered variant)

    Returns dict of found paths or None for missing.
    """
    prep_dir = Path(prep_dir)
    files = {}

    # Ligand parameters (from script 04)
    param_candidates = [
        prep_dir / "params" / ligand_name,
        prep_dir / "01_params" / ligand_name,
        prep_dir / ligand_name / "params",
        prep_dir / ligand_name,
    ]
    for p in param_candidates:
        itp = p / "LIG.itp"
        if itp.exists():
            files["ligand_itp"] = itp
            files["ligand_gro"] = p / "LIG.gro"
            files["ligand_top"] = p / "LIG.top" if (p / "LIG.top").exists() else None
            atomtypes = p / "LIG_atomtypes.itp"
            files["atomtypes_itp"] = atomtypes if atomtypes.exists() else None
            files["param_dir"] = p
            break

    # Complex system (from script 05)
    # Search multiple layout conventions
    complex_candidates = list(prep_dir.glob(f"complex/*{ligand_name}*")) + \
                          list(prep_dir.glob(f"*{ligand_name}*/complex*")) + \
                          list(prep_dir.glob(f"md_simulation/*{ligand_name}*")) + \
                          list(prep_dir.glob(f"02_setup/*/{ligand_name}/pose_*")) + \
                          list(prep_dir.glob(f"02_setup/*{ligand_name}*"))
    for c in complex_candidates:
        if c.is_dir():
            top = c / "topol.top"
            gro = c / "complex_em.gro"
            if not gro.exists():
                gro = c / "complex_ions.gro"
            if not gro.exists():
                gro = c / "em.gro"
            if top.exists() and gro.exists():
                files["complex_top"] = top
                files["complex_gro"] = gro
                files["complex_dir"] = c
                break

    # Stability / equilibrated (from script 06, optional)
    stab_candidates = list(prep_dir.glob(f"stability/*{ligand_name}*")) + \
                      list(prep_dir.glob(f"*{ligand_name}*/stability")) + \
                      list(prep_dir.glob(f"03_stability/*/{ligand_name}/pose_*")) + \
                      list(prep_dir.glob(f"03_stability/*{ligand_name}*"))
    for s in stab_candidates:
        if s.is_dir():
            prod_gro = s / "prod.gro"
            npt_gro = s / "npt.gro"
            input_gro = s / "input.gro"
            if prod_gro.exists():
                files["equilibrated_gro"] = prod_gro
                files["stability_dir"] = s
                break
            elif npt_gro.exists():
                files["equilibrated_gro"] = npt_gro
                files["stability_dir"] = s
                break
            elif input_gro.exists():
                files["equilibrated_gro"] = input_gro
                files["stability_dir"] = s
                break

    return files


def build_solvent_leg(gmx, ligand_itp, ligand_gro, atomtypes_itp, complex_top,
                      output_dir, ligand_resname="LIG", ion_conc=0.15):
    """
    Build the ligand-in-water (solvent leg) system.

    Solvates the parameterised ligand alone in TIP3P, adds ions,
    and energy-minimises. Uses the SAME ligand topology as the complex
    leg to ensure charge consistency.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a standalone topology for ligand-in-water
    # Extract forcefield include from complex topology
    ff_include = ""
    water_include = ""
    ion_includes = ""
    with open(complex_top) as f:
        for line in f:
            if "forcefield.itp" in line and line.strip().startswith("#include"):
                ff_include = line
            elif "tip3p.itp" in line or "spc.itp" in line or "water" in line.lower():
                if line.strip().startswith("#include"):
                    water_include = line
            elif ("ions.itp" in line) and line.strip().startswith("#include"):
                ion_includes += line

    # If we didn't find water include, use default
    if not water_include:
        water_include = '#include "amber99sb-ildn.ff/tip3p.itp"\n'
    if not ff_include:
        ff_include = '#include "amber99sb-ildn.ff/forcefield.itp"\n'

    # Write ligand-only topology
    ligand_top = output_dir / "ligand.top"
    with open(ligand_top, "w") as f:
        f.write("; Ligand-in-water topology for ABFE solvent leg\n")
        f.write("; Uses identical ligand parameters as complex leg\n\n")
        f.write(ff_include)
        f.write("\n")
        # Ligand atomtypes + topology MUST come before water/ions
        if atomtypes_itp:
            f.write(f'#include "{atomtypes_itp.resolve()}"\n')
        f.write(f'#include "{ligand_itp.resolve()}"\n\n')
        f.write(water_include)
        if ion_includes:
            f.write(ion_includes)
        f.write("\n")
        f.write("[ system ]\n")
        f.write("Ligand in water\n\n")
        f.write("[ molecules ]\n")
        f.write(f"{ligand_resname}     1\n")

    # Create box around ligand
    boxed_gro = output_dir / "ligand_box.gro"
    success, _ = run_command(
        [gmx, 'editconf', '-f', str(ligand_gro), '-o', str(boxed_gro),
         '-bt', 'dodecahedron', '-d', '1.2', '-c'],
        "Creating box around ligand"
    )
    if not success:
        return False

    # Solvate
    solvated_gro = output_dir / "ligand_solv.gro"
    success, _ = run_command(
        [gmx, 'solvate', '-cp', str(boxed_gro), '-cs', 'spc216',
         '-o', str(solvated_gro), '-p', str(ligand_top)],
        "Solvating ligand"
    )
    if not success:
        return False

    # Add ions
    ions_mdp = output_dir / "ions.mdp"
    with open(ions_mdp, "w") as f:
        f.write("integrator = steep\nnsteps = 0\n")

    tpr = output_dir / "ions.tpr"
    success, _ = run_command(
        [gmx, 'grompp', '-f', str(ions_mdp), '-c', str(solvated_gro),
         '-p', str(ligand_top), '-o', str(tpr), '-po', str(output_dir / "mdout_ions.mdp"),
         '-maxwarn', '2'],
        "Preparing ion addition", cwd=str(output_dir)
    )
    if not success:
        return False

    ionized_gro = output_dir / "ligand_ions.gro"
    success, _ = run_command(
        [gmx, 'genion', '-s', str(tpr), '-o', str(ionized_gro),
         '-p', str(ligand_top), '-pname', 'NA', '-nname', 'CL',
         '-neutral', '-conc', str(ion_conc)],
        "Adding ions to ligand system",
        input_text='SOL\n', cwd=str(output_dir)
    )
    if not success:
        return False

    # Energy minimisation
    em_mdp = output_dir / "em.mdp"
    with open(em_mdp, "w") as f:
        f.write("""; Energy minimization for solvent leg
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000
nstlist     = 10
cutoff-scheme = Verlet
pbc         = xyz
coulombtype = PME
rcoulomb    = 1.0
vdwtype     = Cut-off
rvdw        = 1.0
constraints = none
""")

    tpr_em = output_dir / "em.tpr"
    success, _ = run_command(
        [gmx, 'grompp', '-f', str(em_mdp), '-c', str(ionized_gro),
         '-p', str(ligand_top), '-o', str(tpr_em),
         '-po', str(output_dir / "mdout_em.mdp"), '-maxwarn', '2'],
        "Preparing EM for solvent leg", cwd=str(output_dir)
    )
    if not success:
        return False

    success, _ = run_command(
        [gmx, 'mdrun', '-v', '-deffnm', str(output_dir / "em")],
        "Running EM for solvent leg", cwd=str(output_dir)
    )
    if not success:
        return False

    # Final output: copy to standard names
    final_gro = output_dir / "ligand.gro"
    shutil.copy(output_dir / "em.gro", final_gro)

    # Clean up intermediates
    for f in ["ions.mdp", "ions.tpr", "mdout_ions.mdp", "em.mdp", "em.tpr",
              "mdout_em.mdp", "em.log", "em.edr", "em.trr",
              "ligand_box.gro", "ligand_solv.gro", "ligand_ions.gro"]:
        (output_dir / f).unlink(missing_ok=True)

    print(f"  Solvent leg ready: {final_gro}")
    return True


def restructure_for_biggin(ligand_name, prep_files, output_dir, gmx, config=None):
    """
    Restructure user's prep output into the format expected by cli-abfe-gmx.

    Expected output:
      output_dir/<ligand_name>/
        complex/complex.gro, complex.top, *.itp
        ligand/ligand.gro, ligand.top, *.itp
    """
    lig_dir = Path(output_dir) / ligand_name
    complex_out = lig_dir / "complex"
    ligand_out = lig_dir / "ligand"
    complex_out.mkdir(parents=True, exist_ok=True)
    ligand_out.mkdir(parents=True, exist_ok=True)

    # --- Complex leg ---
    # Use equilibrated structure if available, else EM structure
    src_gro = prep_files.get("equilibrated_gro", prep_files.get("complex_gro"))
    if src_gro is None or not src_gro.exists():
        print(f"  ERROR: No complex GRO found for {ligand_name}")
        return False

    shutil.copy(src_gro, complex_out / "complex.gro")
    shutil.copy(prep_files["complex_top"], complex_out / "complex.top")

    # Copy all ITP files from complex directory
    if "complex_dir" in prep_files:
        for itp in prep_files["complex_dir"].glob("*.itp"):
            shutil.copy(itp, complex_out / itp.name)

    # --- Solvent leg ---
    # Build ligand-in-water system using same ligand topology
    ligand_itp = prep_files.get("ligand_itp")
    ligand_gro = prep_files.get("ligand_gro")
    atomtypes_itp = prep_files.get("atomtypes_itp")
    complex_top = prep_files["complex_top"]

    if ligand_itp is None or ligand_gro is None:
        print(f"  ERROR: Missing ligand ITP/GRO for {ligand_name}")
        return False

    success = build_solvent_leg(
        gmx=gmx,
        ligand_itp=ligand_itp,
        ligand_gro=ligand_gro,
        atomtypes_itp=atomtypes_itp,
        complex_top=complex_top,
        output_dir=ligand_out
    )
    if not success:
        print(f"  ERROR: Failed to build solvent leg for {ligand_name}")
        return False

    # Copy ITP files to ligand dir too
    if ligand_itp.exists():
        shutil.copy(ligand_itp, ligand_out / ligand_itp.name)
    if atomtypes_itp and atomtypes_itp.exists():
        shutil.copy(atomtypes_itp, ligand_out / atomtypes_itp.name)

    print(f"  Restructured {ligand_name} for ABFE engine")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Bridge prep pipeline output to forked Biggin ABFE engine",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--prep_dir', '-p', required=True,
                        help='Root directory of prep pipeline output (contains params/, complex/, stability/)')
    parser.add_argument('--config', '-c', required=True,
                        help='Per-ligand YAML config file (abfe_config.yml)')
    parser.add_argument('--output_dir', '-o', required=True,
                        help='Output directory for ABFE calculations')
    parser.add_argument('--ligands', '-l', nargs='+', default=None,
                        help='Specific ligand names to process (default: all in config)')
    parser.add_argument('--abfe_code', default=None,
                        help='Path to forked ABFE code (default: ~/git/Oslo/ABFE)')
    parser.add_argument('--n_replicates', '-nr', type=int, default=None,
                        help='Override number of replicates from config')
    parser.add_argument('--n_jobs', '-nj', type=int, default=40,
                        help='Number of parallel jobs per ligand (default: 40)')
    parser.add_argument('--n_cpus', '-nc', type=int, default=7,
                        help='CPUs per job (default: 7)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Build input structure only, do not launch Snakemake')
    parser.add_argument('--no_submit', action='store_true',
                        help='Generate job scripts but do not submit to SLURM')
    parser.add_argument('--gmx', default=None,
                        help='GROMACS executable (auto-detected if not specified)')

    args = parser.parse_args()

    # Find GROMACS
    gmx = args.gmx or find_gmx()
    if gmx is None:
        print("ERROR: GROMACS not found")
        sys.exit(1)

    # Load config
    ligand_configs, defaults = load_config(args.config)

    # Determine ligand list
    if args.ligands:
        ligand_names = args.ligands
    else:
        ligand_names = list(ligand_configs.keys())

    n_replicates = args.n_replicates or defaults.get("n_replicates", 3)

    print("=" * 60)
    print("ABFE BRIDGE: Prep Pipeline -> Biggin ABFE Engine")
    print("=" * 60)
    print(f"\nPrep directory:  {args.prep_dir}")
    print(f"Config:          {args.config}")
    print(f"Output:          {args.output_dir}")
    print(f"Ligands:         {ligand_names}")
    print(f"Replicates:      {n_replicates}")
    print(f"Dry run:         {args.dry_run}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    abfe_input_dir = output_dir / "abfe_input"
    abfe_input_dir.mkdir(exist_ok=True)

    # Step 1: Find prep files and restructure for each ligand
    print("\n" + "-" * 60)
    print("Step 1: Restructure prep output for ABFE engine")
    print("-" * 60)

    success_count = 0
    for lig_name in ligand_names:
        print(f"\n  Processing {lig_name}...")
        prep_files = find_prep_files(args.prep_dir, lig_name)

        if not prep_files.get("ligand_itp"):
            print(f"  WARNING: Could not find prep files for {lig_name}")
            print(f"  Searched in: {args.prep_dir}")
            print(f"  Expected: params/{lig_name}/LIG.itp or similar")
            continue

        lig_config = ligand_configs.get(lig_name, {})
        ok = restructure_for_biggin(
            ligand_name=lig_name,
            prep_files=prep_files,
            output_dir=abfe_input_dir,
            gmx=gmx,
            config=lig_config
        )
        if ok:
            success_count += 1

    print(f"\n  Successfully prepared {success_count}/{len(ligand_names)} ligands")

    if success_count == 0:
        print("ERROR: No ligands were successfully prepared")
        sys.exit(1)

    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN COMPLETE")
        print("=" * 60)
        print(f"\nInput structure built at: {abfe_input_dir}")
        print(f"\nTo launch ABFE manually:")
        print(f"  cli-abfe-gmx \\")
        print(f"    -d {abfe_input_dir} \\")
        print(f"    -o {output_dir / 'abfe_output'} \\")
        print(f"    -lc {args.config} \\")
        print(f"    -nr {n_replicates} \\")
        print(f"    -njl {args.n_jobs} \\")
        print(f"    -ncl {args.n_cpus}")
        return

    # Step 2: Launch the forked Biggin ABFE engine
    print("\n" + "-" * 60)
    print("Step 2: Launch ABFE Snakemake workflow")
    print("-" * 60)

    abfe_output_dir = output_dir / "abfe_output"

    # Determine path to cli-abfe-gmx
    abfe_code = args.abfe_code or os.path.expanduser("~/git/Oslo/ABFE")

    # Build command
    cmd = [
        sys.executable, "-m", "abfe_cli.ABFECalculatorGmx",
        "-d", str(abfe_input_dir),
        "-o", str(abfe_output_dir),
        "-lc", str(Path(args.config).resolve()),
        "-nr", str(n_replicates),
        "-njl", str(args.n_jobs),
        "-ncl", str(args.n_cpus),
        "-pn", "ABFE",
    ]

    if args.no_submit:
        cmd.append("-nosubmit")

    # Add ABFE src to PYTHONPATH
    env = os.environ.copy()
    abfe_src = os.path.join(abfe_code, "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = abfe_src + ":" + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = abfe_src

    print(f"  ABFE code:   {abfe_code}")
    print(f"  Input:       {abfe_input_dir}")
    print(f"  Output:      {abfe_output_dir}")
    print(f"  Config:      {args.config}")
    print(f"  Command:     {' '.join(cmd)}")

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"\n  WARNING: ABFE engine returned non-zero exit code: {result.returncode}")
    else:
        print(f"\n  ABFE workflow launched successfully")

    # Step 3: Summary
    print("\n" + "=" * 60)
    print("LAUNCH COMPLETE")
    print("=" * 60)
    print(f"\nABFE input:  {abfe_input_dir}")
    print(f"ABFE output: {abfe_output_dir}")
    print(f"Config:      {args.config}")
    print(f"\nNext steps:")
    print(f"  1. Monitor jobs: squeue -u $USER")
    print(f"  2. Check progress: ls {abfe_output_dir}/*/*/dG_results.tsv")
    print(f"  3. Analyze results: python 10_analyze_abfe.py \\")
    print(f"       --results_dir {abfe_output_dir} \\")
    print(f"       --config {args.config}")


if __name__ == "__main__":
    main()
