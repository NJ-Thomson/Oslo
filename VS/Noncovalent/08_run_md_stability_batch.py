#!/usr/bin/env python3
"""
Batch MD Stability Testing for Noncovalent Complexes

Orchestrates the full MD stability workflow for multiple docked poses using
the existing pipeline scripts:
  - 04_parameterize_ligand.py: Ligand parameterization (GAFF2)
  - 05_setup_complex.py: GROMACS system setup
  - 06_test_binding_stability.py: MD equilibration and production

Output directory structure:
    output_dir/
    ├── 01_params/                    # Parameterized ligands (cached)
    │   └── <ligand>/
    │       ├── LIG.itp
    │       └── LIG.gro
    ├── 02_setup/                     # GROMACS setup files
    │   └── <receptor>/
    │       └── <ligand>/
    │           └── pose_1/
    │               ├── complex_em.gro
    │               └── topol.top
    ├── 03_stability/                 # MD simulations and analysis
    │   └── <receptor>/
    │       └── <ligand>/
    │           └── pose_1/
    │               ├── nvt.xtc
    │               ├── npt.xtc
    │               ├── prod.xtc
    │               └── analysis/
    └── results_summary.csv           # Overall stability results

Pose Selection:
    By default, uses H-bond filtered poses (*_hbond_filtered.sdf) if available.
    Falls back to all poses (*_poses.sdf) sorted by docking affinity.
    Use --no_hbond_filter to always use all poses regardless.

Usage:
    # Generate setup for all ligands (best 3 poses, 20ns production)
    # Uses H-bond filtered poses preferentially
    python 08_run_md_stability_batch.py \\
        --input_dir Outputs/non_covalent \\
        --output_dir Outputs/non_covalent_md \\
        --n_poses 3 --prod_time 20

    # Specific kinase and ligands
    python 08_run_md_stability_batch.py \\
        --input_dir Outputs/non_covalent \\
        --output_dir Outputs/non_covalent_md \\
        --kinases <receptor> --ligands <ligand_1>,<ligand_2>

    # Use all poses (ignore H-bond filtering)
    python 08_run_md_stability_batch.py \\
        --input_dir Outputs/non_covalent \\
        --output_dir Outputs/non_covalent_md \\
        --no_hbond_filter

Dependencies:
    Requires the existing scripts:
    - 04_parameterize_ligand.py
    - 05_setup_complex.py
    - 06_test_binding_stability.py
    - analyze_stability.py
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_script(script_path, args_str, cwd=None):
    """Run a Python script with arguments."""
    cmd = f"python {script_path} {args_str}"
    print(f"  $ {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        capture_output=False, text=True
    )
    return result.returncode == 0


def extract_pose_from_sdf(sdf_path, pose_idx, output_sdf):
    """Extract a specific pose from multi-conformer SDF."""
    try:
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mols = [mol for mol in suppl if mol is not None]

        if pose_idx >= len(mols):
            print(f"  ERROR: Pose {pose_idx} not found (only {len(mols)} poses)")
            return False

        mol = mols[pose_idx]
        writer = Chem.SDWriter(str(output_sdf))
        writer.write(mol)
        writer.close()
        return True
    except ImportError:
        # Fallback to obabel
        cmd = f"obabel {sdf_path} -O {output_sdf} -f {pose_idx + 1} -l {pose_idx + 1}"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        return result.returncode == 0


def get_docking_scores(results_json):
    """Get sorted docking scores from results JSON."""
    if not results_json.exists():
        return []

    with open(results_json) as f:
        data = json.load(f)

    poses = data.get('poses', [])
    return [(p['mode'] - 1, p['affinity']) for p in sorted(poses, key=lambda x: x['affinity'])]


def main():
    parser = argparse.ArgumentParser(
        description='Batch MD stability testing for noncovalent complexes',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--input_dir', '-i', required=True,
                        help='Input directory with docking results (<receptor>/, modeller/)')
    parser.add_argument('--output_dir', '-o', required=True,
                        help='Output directory for MD workflow')
    parser.add_argument('--kinases', '-k', required=True,
                        help='Kinases to process (comma-separated)')
    parser.add_argument('--ligands', '-l', default=None,
                        help='Ligands to process (comma-separated, default: all)')
    parser.add_argument('--n_poses', '-n', type=int, default=3,
                        help='Number of best poses per ligand (default: 3)')
    parser.add_argument('--prod_time', type=float, default=20.0,
                        help='Production MD time in ns (default: 20)')
    parser.add_argument('--scripts_only', action='store_true',
                        help='Generate scripts only, do not run')
    parser.add_argument('--skip_parameterize', action='store_true',
                        help='Skip parameterization (use cached params)')
    parser.add_argument('--skip_setup', action='store_true',
                        help='Skip GROMACS setup (use existing)')
    parser.add_argument('--no_hbond_filter', action='store_true',
                        help='Use all poses instead of H-bond filtered poses')

    args = parser.parse_args()

    # Parse inputs
    kinases = [k.strip() for k in args.kinases.split(',')]
    ligands = [l.strip() for l in args.ligands.split(',')] if args.ligands else None

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    script_dir = Path(__file__).parent.resolve()

    # Create output structure
    params_dir = output_dir / '01_params'
    setup_dir = output_dir / '02_setup'
    stability_dir = output_dir / '03_stability'

    for d in [params_dir, setup_dir, stability_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # PDB codes for each kinase
    pdb_codes = {}  # Add your receptor name -> PDB code mappings here

    print("="*70)
    print("BATCH MD STABILITY WORKFLOW")
    print("="*70)
    print(f"\nInput:   {input_dir}")
    print(f"Output:  {output_dir}")
    print(f"Kinases: {', '.join(kinases)}")
    print(f"Ligands: {', '.join(ligands) if ligands else 'all'}")
    print(f"Poses:   {args.n_poses}")
    print(f"Prod MD: {args.prod_time} ns")
    print(f"H-bond filter: {'disabled' if args.no_hbond_filter else 'enabled (using *_hbond_filtered.sdf if available)'}")

    jobs = []

    for kinase in kinases:
        kinase_dir = input_dir / kinase
        if not kinase_dir.exists():
            print(f"\nWARNING: {kinase_dir} not found")
            continue

        pdb_code = pdb_codes.get(kinase)
        if not pdb_code:
            print(f"\nWARNING: Unknown PDB code for {kinase}")
            continue

        # Find receptor
        receptor_candidates = [
            input_dir / 'modeller' / f'{pdb_code}_predock.pdb',
            input_dir / f'{kinase}_ABFE' / 'receptor.pdb',
        ]
        receptor_pdb = None
        for r in receptor_candidates:
            if r.exists():
                receptor_pdb = r
                break

        if not receptor_pdb:
            print(f"\nWARNING: Receptor not found for {kinase}")
            continue

        print(f"\n{'='*70}")
        print(f"Processing {kinase} (receptor: {receptor_pdb.name})")
        print("="*70)

        # Find ligands
        available_ligands = [d.name for d in kinase_dir.iterdir()
                           if d.is_dir() and d.name.startswith('Inhib_')]

        if ligands:
            available_ligands = [l for l in available_ligands if l in ligands]

        for ligand in sorted(available_ligands):
            ligand_dir = kinase_dir / ligand

            # Preferentially use H-bond filtered poses unless disabled
            poses_sdf = None
            using_hbond_filter = False

            if not args.no_hbond_filter:
                hbond_sdf = ligand_dir / f'{pdb_code}_{ligand}_hbond_filtered.sdf'
                if hbond_sdf.exists():
                    poses_sdf = hbond_sdf
                    using_hbond_filter = True

            # Fallback to all poses
            if poses_sdf is None:
                poses_sdf = ligand_dir / f'{pdb_code}_{ligand}_poses.sdf'
                if not poses_sdf.exists():
                    poses_sdf = ligand_dir / f'{pdb_code}_{ligand}_poses_pH7.4.sdf'

            if not poses_sdf.exists():
                print(f"\n  WARNING: No poses SDF for {ligand}")
                continue

            # Get poses - either from H-bond filtered file or by docking score
            if using_hbond_filter:
                # H-bond filtered file: poses are already filtered, use them in order
                try:
                    from rdkit import Chem
                    suppl = Chem.SDMolSupplier(str(poses_sdf), removeHs=False)
                    n_poses_available = len([m for m in suppl if m is not None])
                except:
                    n_poses_available = 20  # Fallback
                scores = [(i, 0.0) for i in range(n_poses_available)]
                print(f"\n  Processing {ligand} - using H-bond filtered poses ({min(args.n_poses, n_poses_available)} of {n_poses_available})")
            else:
                # Use docking scores to rank poses
                results_json = ligand_dir / f'{pdb_code}_{ligand}_results.json'
                scores = get_docking_scores(results_json)
                if not scores:
                    scores = [(i, 0.0) for i in range(20)]
                print(f"\n  Processing {ligand} - using top {min(args.n_poses, len(scores))} by docking affinity")

            for pose_idx, affinity in scores[:args.n_poses]:
                pose_name = f"pose_{pose_idx + 1}"

                # Output paths for this pose
                pose_params = params_dir / ligand
                pose_setup = setup_dir / kinase / ligand / pose_name
                pose_stability = stability_dir / kinase / ligand / pose_name

                jobs.append({
                    'kinase': kinase,
                    'ligand': ligand,
                    'pose_idx': pose_idx,
                    'pose_name': pose_name,
                    'affinity': affinity,
                    'poses_sdf': poses_sdf,
                    'receptor_pdb': receptor_pdb,
                    'params_dir': pose_params,
                    'setup_dir': pose_setup,
                    'stability_dir': pose_stability,
                })

    print(f"\n{'='*70}")
    print(f"Total jobs: {len(jobs)}")
    print("="*70)

    # Process each job
    success_count = 0
    for job in jobs:
        print(f"\n{'='*70}")
        print(f"{job['kinase']}/{job['ligand']}/{job['pose_name']} (affinity: {job['affinity']:.2f})")
        print("="*70)

        # Step 1: Extract pose
        print("\n[Step 1] Extracting pose from SDF...")
        job['setup_dir'].mkdir(parents=True, exist_ok=True)
        pose_sdf = job['setup_dir'] / 'ligand_pose.sdf'

        if not extract_pose_from_sdf(job['poses_sdf'], job['pose_idx'], pose_sdf):
            print("  FAILED to extract pose")
            continue

        # Step 2: Parameterize ligand (cached per ligand)
        ligand_itp = job['params_dir'] / 'LIG.itp'
        if not args.skip_parameterize and not ligand_itp.exists():
            print("\n[Step 2] Parameterizing ligand...")
            job['params_dir'].mkdir(parents=True, exist_ok=True)

            script_path = script_dir / '04_parameterize_ligand.py'
            run_script(
                script_path,
                f"--ligand {pose_sdf} --output_dir {job['params_dir']} --resname LIG"
            )
        else:
            print(f"\n[Step 2] Using cached parameters: {ligand_itp}")

        if not ligand_itp.exists():
            print("  ERROR: Parameterization failed")
            continue

        # Step 3: Setup complex
        complex_gro = job['setup_dir'] / 'complex_em.gro'
        if not args.skip_setup and not complex_gro.exists():
            print("\n[Step 3] Setting up GROMACS complex...")
            script_path = script_dir / '05_setup_complex.py'

            # Use --complex mode with docked pose
            complex_pdb = job['setup_dir'].parent.parent.parent.parent / job['kinase'] / job['ligand'] / f"{pdb_codes[job['kinase']]}_{job['ligand']}_complex.pdb"

            # Alternative: Use receptor + docked ligand
            run_script(
                script_path,
                f"--receptor {job['receptor_pdb']} "
                f"--ligand_itp {ligand_itp} "
                f"--ligand_gro {job['params_dir'] / 'LIG.gro'} "
                f"--docked_pose {pose_sdf} "
                f"--output_dir {job['setup_dir']}"
            )
        else:
            print(f"\n[Step 3] Using existing setup: {complex_gro}")

        if not complex_gro.exists():
            # Try finding em.gro in setup_dir
            em_gro = job['setup_dir'] / 'em.gro'
            if em_gro.exists():
                complex_gro = em_gro

        if not complex_gro.exists():
            print("  ERROR: Complex setup failed")
            continue

        # Step 4: Run stability test
        print("\n[Step 4] Setting up stability test...")
        job['stability_dir'].mkdir(parents=True, exist_ok=True)

        script_path = script_dir / '06_test_binding_stability.py'
        script_args = (
            f"--complex_gro {complex_gro} "
            f"--topology {job['setup_dir'] / 'topol.top'} "
            f"--output_dir {job['stability_dir']} "
            f"--prod_time {args.prod_time}"
        )

        if args.scripts_only:
            script_args += " --scripts_only"

        run_script(script_path, script_args)

        # Save metadata
        metadata = {
            'kinase': job['kinase'],
            'ligand': job['ligand'],
            'pose_index': job['pose_idx'],
            'pose_name': job['pose_name'],
            'docking_affinity': job['affinity'],
            'production_time_ns': args.prod_time,
        }
        with open(job['stability_dir'] / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        success_count += 1
        print(f"\n  Setup complete: {job['stability_dir']}")

    # Summary
    print("\n" + "="*70)
    print("WORKFLOW SUMMARY")
    print("="*70)
    print(f"\nTotal jobs:  {len(jobs)}")
    print(f"Successful:  {success_count}")
    print(f"Failed:      {len(jobs) - success_count}")

    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── 01_params/      # Parameterized ligands")
    print(f"  ├── 02_setup/       # GROMACS setup files")
    print(f"  └── 03_stability/   # MD simulations")

    if args.scripts_only:
        print(f"\nScripts generated. To run simulations:")
        print(f"  cd {stability_dir}/<kinase>/<ligand>/<pose>")
        print(f"  ./run.sh")
        print(f"\nOr submit all to SLURM:")
        print(f"  find {stability_dir} -name 'run.sh' -execdir bash {{}} \\;")
    else:
        print(f"\nSimulations should now be running or complete.")
        print(f"To analyze results:")
        print(f"  cd {stability_dir}/<kinase>/<ligand>/<pose>")
        print(f"  ./analyze.sh")
        print(f"  python {script_dir}/analyze_stability.py --dir analysis/")


if __name__ == '__main__':
    main()
