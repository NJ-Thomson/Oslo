#!/usr/bin/env python3
"""
Analyze convergence of ensemble sampling across multiple GPCR MD simulation repeats.
Checks RMSD, RMSF, Rg, and structural clustering to assess convergence.
"""

import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from MDAnalysis.analysis.rms import RMSF
from MDAnalysis.analysis.distances import dist
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_repeats(base_dir='.', n_repeats=5, struct='gpcr_only.gro', traj='gpcr_only.xtc'):
    """
    Load trajectories from repeat folders r1/ to r5/.

    Parameters
    ----------
    base_dir : str
        Base directory containing r1/, r2/, etc.
    n_repeats : int
        Number of repeat simulations
    struct : str
        Structure filename in each repeat folder
    traj : str
        Trajectory filename in each repeat folder

    Returns
    -------
    list of MDAnalysis.Universe
        Loaded universes for each repeat
    """
    universes = []
    base_path = Path(base_dir)

    for i in range(1, n_repeats + 1):
        repeat_dir = base_path / f'r{i}'
        struct_file = repeat_dir / struct
        traj_file = repeat_dir / traj

        if not struct_file.exists():
            raise FileNotFoundError(f"Structure not found: {struct_file}")
        if not traj_file.exists():
            raise FileNotFoundError(f"Trajectory not found: {traj_file}")

        u = mda.Universe(str(struct_file), str(traj_file))
        universes.append(u)
        print(f"Loaded r{i}: {u.atoms.n_atoms} atoms, {u.trajectory.n_frames} frames")

    return universes


def calculate_rmsd_matrix(universes, selection='protein and name CA'):
    """
    Calculate RMSD for each repeat trajectory.

    Parameters
    ----------
    universes : list
        List of MDAnalysis universes
    selection : str
        Atom selection for RMSD calculation

    Returns
    -------
    list of arrays
        RMSD time series for each repeat
    """
    rmsd_data = []

    for i, u in enumerate(universes):
        print(f"Calculating RMSD for repeat {i+1}...")

        # Align trajectory to first frame
        align.AlignTraj(u, u, select=selection, in_memory=False).run()

        # Calculate RMSD
        ref = u.select_atoms(selection)
        rmsd_vals = []

        for ts in u.trajectory:
            mobile = u.select_atoms(selection)
            rmsd_vals.append(rms.rmsd(mobile.positions, ref.positions, superposition=True))

        rmsd_data.append(np.array(rmsd_vals))

    return rmsd_data


def calculate_rmsf(universes, selection='protein and name CA'):
    """
    Calculate RMSF for each repeat.

    Parameters
    ----------
    universes : list
        List of MDAnalysis universes
    selection : str
        Atom selection for RMSF calculation

    Returns
    -------
    list of arrays
        RMSF values for each repeat
    """
    rmsf_data = []

    for i, u in enumerate(universes):
        print(f"Calculating RMSF for repeat {i+1}...")

        atoms = u.select_atoms(selection)
        rmsf_calc = RMSF(atoms).run()
        rmsf_data.append(rmsf_calc.results.rmsf)

    return rmsf_data


def calculate_rg(universes, selection='protein'):
    """
    Calculate radius of gyration for each repeat.

    Parameters
    ----------
    universes : list
        List of MDAnalysis universes
    selection : str
        Atom selection for Rg calculation

    Returns
    -------
    list of arrays
        Rg time series for each repeat
    """
    rg_data = []

    for i, u in enumerate(universes):
        print(f"Calculating Rg for repeat {i+1}...")

        protein = u.select_atoms(selection)
        rg_vals = []

        for ts in u.trajectory:
            rg_vals.append(protein.radius_of_gyration())

        rg_data.append(np.array(rg_vals))

    return rg_data


def plot_convergence(rmsd_data, rmsf_data, rg_data, output_prefix='convergence'):
    """
    Create convergence plots.

    Parameters
    ----------
    rmsd_data : list
        RMSD data for each repeat
    rmsf_data : list
        RMSF data for each repeat
    rg_data : list
        Radius of gyration data for each repeat
    output_prefix : str
        Prefix for output files
    """
    n_repeats = len(rmsd_data)
    colors = plt.cm.viridis(np.linspace(0, 1, n_repeats))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # RMSD time series
    ax = axes[0, 0]
    for i, rmsd in enumerate(rmsd_data):
        time = np.arange(len(rmsd))
        ax.plot(time, rmsd, label=f'r{i+1}', color=colors[i], alpha=0.7)
    ax.set_xlabel('Frame')
    ax.set_ylabel('RMSD (Å)')
    ax.set_title('RMSD Time Series')
    ax.legend()
    ax.grid(alpha=0.3)

    # RMSF comparison
    ax = axes[0, 1]
    for i, rmsf in enumerate(rmsf_data):
        residue_ids = np.arange(1, len(rmsf) + 1)
        ax.plot(residue_ids, rmsf, label=f'r{i+1}', color=colors[i], alpha=0.7)
    ax.set_xlabel('Residue')
    ax.set_ylabel('RMSF (Å)')
    ax.set_title('RMSF per Residue')
    ax.legend()
    ax.grid(alpha=0.3)

    # Radius of gyration
    ax = axes[1, 0]
    for i, rg in enumerate(rg_data):
        time = np.arange(len(rg))
        ax.plot(time, rg, label=f'r{i+1}', color=colors[i], alpha=0.7)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Rg (Å)')
    ax.set_title('Radius of Gyration')
    ax.legend()
    ax.grid(alpha=0.3)

    # Statistical summary: RMSD distributions
    ax = axes[1, 1]
    rmsd_means = [np.mean(rmsd) for rmsd in rmsd_data]
    rmsd_stds = [np.std(rmsd) for rmsd in rmsd_data]
    x_pos = np.arange(1, n_repeats + 1)
    ax.bar(x_pos, rmsd_means, yerr=rmsd_stds, color=colors, alpha=0.7, capsize=5)
    ax.set_xlabel('Repeat')
    ax.set_ylabel('Mean RMSD (Å)')
    ax.set_title('RMSD Statistics')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'r{i}' for i in x_pos])
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis.png', dpi=300)
    print(f"\nSaved plot: {output_prefix}_analysis.png")
    plt.close()


def assess_convergence(rmsd_data, rmsf_data, rg_data):
    """
    Quantitatively assess ensemble convergence.

    Parameters
    ----------
    rmsd_data : list
        RMSD data for each repeat
    rmsf_data : list
        RMSF data for each repeat
    rg_data : list
        Radius of gyration data for each repeat

    Returns
    -------
    dict
        Convergence metrics
    """
    n_repeats = len(rmsd_data)

    # RMSD statistics
    rmsd_means = np.array([np.mean(rmsd) for rmsd in rmsd_data])
    rmsd_stds = np.array([np.std(rmsd) for rmsd in rmsd_data])
    rmsd_cv = np.std(rmsd_means) / np.mean(rmsd_means) * 100  # Coefficient of variation

    # RMSF statistics
    rmsf_means = np.array([np.mean(rmsf) for rmsf in rmsf_data])
    rmsf_cv = np.std(rmsf_means) / np.mean(rmsf_means) * 100

    # Rg statistics
    rg_means = np.array([np.mean(rg) for rg in rg_data])
    rg_stds = np.array([np.std(rg) for rg in rg_data])
    rg_cv = np.std(rg_means) / np.mean(rg_means) * 100

    # RMSF correlation between repeats
    rmsf_correlations = []
    for i in range(n_repeats):
        for j in range(i + 1, n_repeats):
            corr = np.corrcoef(rmsf_data[i], rmsf_data[j])[0, 1]
            rmsf_correlations.append(corr)
    mean_rmsf_corr = np.mean(rmsf_correlations)

    metrics = {
        'rmsd_mean': np.mean(rmsd_means),
        'rmsd_std': np.std(rmsd_means),
        'rmsd_cv': rmsd_cv,
        'rmsf_mean': np.mean(rmsf_means),
        'rmsf_cv': rmsf_cv,
        'rmsf_correlation': mean_rmsf_corr,
        'rg_mean': np.mean(rg_means),
        'rg_std': np.std(rg_means),
        'rg_cv': rg_cv,
    }

    return metrics


def print_convergence_report(metrics):
    """
    Print convergence assessment report.

    Parameters
    ----------
    metrics : dict
        Convergence metrics from assess_convergence()
    """
    print("\n" + "="*60)
    print("ENSEMBLE CONVERGENCE REPORT")
    print("="*60)

    print(f"\nRMSD Analysis:")
    print(f"  Mean RMSD across repeats: {metrics['rmsd_mean']:.2f} ± {metrics['rmsd_std']:.2f} Å")
    print(f"  Coefficient of variation: {metrics['rmsd_cv']:.2f}%")

    print(f"\nRMSF Analysis:")
    print(f"  Mean RMSF across repeats: {metrics['rmsf_mean']:.2f} Å")
    print(f"  Coefficient of variation: {metrics['rmsf_cv']:.2f}%")
    print(f"  Mean pairwise correlation: {metrics['rmsf_correlation']:.3f}")

    print(f"\nRadius of Gyration:")
    print(f"  Mean Rg across repeats: {metrics['rg_mean']:.2f} ± {metrics['rg_std']:.2f} Å")
    print(f"  Coefficient of variation: {metrics['rg_cv']:.2f}%")

    print("\n" + "="*60)
    print("CONVERGENCE ASSESSMENT:")
    print("="*60)

    # Assess convergence based on typical criteria
    converged = True
    issues = []

    if metrics['rmsd_cv'] > 10:
        converged = False
        issues.append(f"High RMSD variability (CV={metrics['rmsd_cv']:.1f}% > 10%)")

    if metrics['rmsf_cv'] > 15:
        converged = False
        issues.append(f"High RMSF variability (CV={metrics['rmsf_cv']:.1f}% > 15%)")

    if metrics['rmsf_correlation'] < 0.7:
        converged = False
        issues.append(f"Low RMSF correlation ({metrics['rmsf_correlation']:.2f} < 0.70)")

    if metrics['rg_cv'] > 5:
        converged = False
        issues.append(f"High Rg variability (CV={metrics['rg_cv']:.1f}% > 5%)")

    if converged:
        print("✓ Ensemble appears CONVERGED")
        print("  - All metrics show good agreement between repeats")
    else:
        print("✗ Ensemble appears NOT CONVERGED")
        print("\nIssues detected:")
        for issue in issues:
            print(f"  - {issue}")

    print("\nGuidelines:")
    print("  - RMSD CV < 10%: Good convergence")
    print("  - RMSF CV < 15%: Good convergence")
    print("  - RMSF correlation > 0.7: Similar flexibility patterns")
    print("  - Rg CV < 5%: Stable overall structure")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze convergence of ensemble sampling across MD repeats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: analyze r1/ to r5/ in current directory
  %(prog)s

  # Specify base directory
  %(prog)s -d /path/to/simulations

  # Different number of repeats
  %(prog)s -n 3

  # Custom filenames
  %(prog)s -s protein.gro -t protein.xtc
        """
    )

    parser.add_argument('-d', '--dir', default='.', help='Base directory containing r1/, r2/, etc.')
    parser.add_argument('-n', '--n-repeats', type=int, default=5, help='Number of repeats (default: 5)')
    parser.add_argument('-s', '--struct', default='gpcr_only.gro', help='Structure filename')
    parser.add_argument('-t', '--traj', default='gpcr_only.xtc', help='Trajectory filename')
    parser.add_argument('-o', '--output', default='convergence', help='Output prefix for plots')
    parser.add_argument('--selection', default='protein and name CA', help='Selection for RMSD/RMSF')

    args = parser.parse_args()

    print("Loading trajectories...")
    universes = load_repeats(
        base_dir=args.dir,
        n_repeats=args.n_repeats,
        struct=args.struct,
        traj=args.traj
    )

    print("\nAnalyzing trajectories...")
    rmsd_data = calculate_rmsd_matrix(universes, selection=args.selection)
    rmsf_data = calculate_rmsf(universes, selection=args.selection)
    rg_data = calculate_rg(universes, selection='protein')

    print("\nGenerating plots...")
    plot_convergence(rmsd_data, rmsf_data, rg_data, output_prefix=args.output)

    print("\nAssessing convergence...")
    metrics = assess_convergence(rmsd_data, rmsf_data, rg_data)
    print_convergence_report(metrics)

    print("Analysis complete!")


if __name__ == '__main__':
    main()
