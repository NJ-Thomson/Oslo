#!/usr/bin/env python3
"""
Analyze stability test results from GROMACS XVG files.

Usage:
    python analyze_stability.py --dir analysis/
    python analyze_stability.py --dir /path/to/analysis --output report.txt
"""

import argparse
import numpy as np
from pathlib import Path
import sys


def parse_xvg(filepath):
    """Parse GROMACS XVG file, return time and data columns."""
    times = []
    data = []

    with open(filepath) as f:
        for line in f:
            if line.startswith(('#', '@')):
                continue
            parts = line.split()
            if len(parts) >= 2:
                times.append(float(parts[0]))
                data.append([float(x) for x in parts[1:]])

    return np.array(times), np.array(data)


def analyze_rmsd(filepath, name="RMSD"):
    """Analyze RMSD for stability."""
    times, data = parse_xvg(filepath)
    rmsd = data[:, 0] if data.ndim > 1 else data

    # Convert to nm if needed (GROMACS outputs in nm)
    total_time = times[-1] - times[0]

    # Split into first and second half
    mid = len(rmsd) // 2
    first_half = rmsd[:mid]
    second_half = rmsd[mid:]

    # Statistics
    mean_rmsd = np.mean(rmsd)
    std_rmsd = np.std(rmsd)
    mean_first = np.mean(first_half)
    mean_second = np.mean(second_half)
    drift = mean_second - mean_first
    max_rmsd = np.max(rmsd)

    # Stability criteria
    # - RMSD should plateau (drift < 0.05 nm)
    # - Protein RMSD typically < 0.3 nm for stable system
    # - Ligand RMSD < 0.2 nm indicates stable binding pose

    is_stable = abs(drift) < 0.05 and std_rmsd < 0.1

    return {
        'name': name,
        'mean': mean_rmsd,
        'std': std_rmsd,
        'max': max_rmsd,
        'drift': drift,
        'is_stable': is_stable,
        'total_time': total_time
    }


def analyze_energy(filepath):
    """Analyze energy for stability."""
    times, data = parse_xvg(filepath)

    # Energy file may have multiple columns (potential, kinetic, total)
    # Usually first column after time is total/potential energy
    energy = data[:, 0] if data.ndim > 1 else data

    total_time = times[-1] - times[0]
    mid = len(energy) // 2

    mean_energy = np.mean(energy)
    std_energy = np.std(energy)
    drift = np.mean(energy[mid:]) - np.mean(energy[:mid])

    # Energy should be stable (fluctuations are normal)
    # Drift should be small relative to mean
    relative_drift = abs(drift / mean_energy) if mean_energy != 0 else 0
    is_stable = relative_drift < 0.01  # < 1% drift

    return {
        'name': 'Energy',
        'mean': mean_energy,
        'std': std_energy,
        'drift': drift,
        'relative_drift': relative_drift,
        'is_stable': is_stable,
        'total_time': total_time
    }


def analyze_gyration(filepath):
    """Analyze radius of gyration."""
    times, data = parse_xvg(filepath)
    rg = data[:, 0] if data.ndim > 1 else data

    total_time = times[-1] - times[0]
    mid = len(rg) // 2

    mean_rg = np.mean(rg)
    std_rg = np.std(rg)
    drift = np.mean(rg[mid:]) - np.mean(rg[:mid])

    # Rg should be stable - large changes indicate unfolding
    is_stable = abs(drift) < 0.1 and std_rg < 0.15

    return {
        'name': 'Radius of Gyration',
        'mean': mean_rg,
        'std': std_rg,
        'drift': drift,
        'is_stable': is_stable,
        'total_time': total_time
    }


def analyze_contacts(filepath):
    """Analyze protein-ligand contacts."""
    times, data = parse_xvg(filepath)
    contacts = data[:, 0] if data.ndim > 1 else data

    total_time = times[-1] - times[0]
    mid = len(contacts) // 2

    mean_contacts = np.mean(contacts)
    std_contacts = np.std(contacts)
    min_contacts = np.min(contacts)
    drift = np.mean(contacts[mid:]) - np.mean(contacts[:mid])

    # Contacts should remain stable - large decrease indicates unbinding
    relative_drift = drift / mean_contacts if mean_contacts > 0 else 0
    is_stable = relative_drift > -0.2 and min_contacts > 0  # No more than 20% loss

    return {
        'name': 'Protein-Ligand Contacts',
        'mean': mean_contacts,
        'std': std_contacts,
        'min': min_contacts,
        'drift': drift,
        'relative_drift': relative_drift,
        'is_stable': is_stable,
        'total_time': total_time
    }


def analyze_mindist(filepath):
    """Analyze minimum distance between protein and ligand."""
    times, data = parse_xvg(filepath)
    mindist = data[:, 0] if data.ndim > 1 else data

    total_time = times[-1] - times[0]

    mean_dist = np.mean(mindist)
    std_dist = np.std(mindist)
    max_dist = np.max(mindist)

    # If min distance increases significantly, ligand may be leaving
    # Typical contact distance < 0.6 nm
    is_stable = max_dist < 0.8 and mean_dist < 0.5

    return {
        'name': 'Min Protein-Ligand Distance',
        'mean': mean_dist,
        'std': std_dist,
        'max': max_dist,
        'is_stable': is_stable,
        'total_time': total_time
    }


def analyze_rmsf(filepath):
    """Analyze RMSF (per-residue fluctuations)."""
    times, data = parse_xvg(filepath)  # times here is actually residue number
    rmsf = data[:, 0] if data.ndim > 1 else data

    mean_rmsf = np.mean(rmsf)
    max_rmsf = np.max(rmsf)
    high_flex_count = np.sum(rmsf > 0.3)  # Residues with RMSF > 0.3 nm

    return {
        'name': 'RMSF',
        'mean': mean_rmsf,
        'max': max_rmsf,
        'high_flexibility_residues': high_flex_count,
        'total_residues': len(rmsf)
    }


def print_report(results, output_file=None):
    """Print stability analysis report."""
    lines = []
    lines.append("=" * 70)
    lines.append("STABILITY ANALYSIS REPORT")
    lines.append("=" * 70)

    all_stable = True

    # RMSD Analysis
    if 'rmsd_protein' in results:
        r = results['rmsd_protein']
        lines.append(f"\n[PROTEIN RMSD]")
        lines.append(f"  Mean:  {r['mean']:.3f} nm")
        lines.append(f"  Std:   {r['std']:.3f} nm")
        lines.append(f"  Max:   {r['max']:.3f} nm")
        lines.append(f"  Drift: {r['drift']:+.3f} nm (2nd half - 1st half)")
        status = "STABLE" if r['is_stable'] else "UNSTABLE"
        lines.append(f"  Status: {status}")
        if r['mean'] > 0.3:
            lines.append(f"  Note: Mean RMSD > 0.3 nm suggests significant conformational change")
        all_stable &= r['is_stable']

    if 'rmsd_ligand' in results:
        r = results['rmsd_ligand']
        lines.append(f"\n[LIGAND RMSD]")
        lines.append(f"  Mean:  {r['mean']:.3f} nm")
        lines.append(f"  Std:   {r['std']:.3f} nm")
        lines.append(f"  Max:   {r['max']:.3f} nm")
        lines.append(f"  Drift: {r['drift']:+.3f} nm")
        status = "STABLE" if r['is_stable'] else "UNSTABLE"
        lines.append(f"  Status: {status}")
        if r['mean'] > 0.2:
            lines.append(f"  Note: Mean RMSD > 0.2 nm suggests ligand pose instability")
        all_stable &= r['is_stable']

    # Energy
    if 'energy' in results:
        r = results['energy']
        lines.append(f"\n[ENERGY]")
        lines.append(f"  Mean:  {r['mean']:.1f} kJ/mol")
        lines.append(f"  Std:   {r['std']:.1f} kJ/mol")
        lines.append(f"  Drift: {r['drift']:+.1f} kJ/mol ({r['relative_drift']*100:.2f}%)")
        status = "STABLE" if r['is_stable'] else "UNSTABLE"
        lines.append(f"  Status: {status}")
        all_stable &= r['is_stable']

    # Radius of Gyration
    if 'gyration' in results:
        r = results['gyration']
        lines.append(f"\n[RADIUS OF GYRATION]")
        lines.append(f"  Mean:  {r['mean']:.3f} nm")
        lines.append(f"  Std:   {r['std']:.3f} nm")
        lines.append(f"  Drift: {r['drift']:+.3f} nm")
        status = "STABLE" if r['is_stable'] else "UNSTABLE"
        lines.append(f"  Status: {status}")
        if abs(r['drift']) > 0.1:
            lines.append(f"  Note: Significant Rg change may indicate unfolding")
        all_stable &= r['is_stable']

    # Contacts
    if 'contacts' in results:
        r = results['contacts']
        lines.append(f"\n[PROTEIN-LIGAND CONTACTS]")
        lines.append(f"  Mean:  {r['mean']:.1f}")
        lines.append(f"  Std:   {r['std']:.1f}")
        lines.append(f"  Min:   {r['min']:.0f}")
        lines.append(f"  Drift: {r['drift']:+.1f} ({r['relative_drift']*100:+.1f}%)")
        status = "STABLE" if r['is_stable'] else "UNSTABLE"
        lines.append(f"  Status: {status}")
        if r['min'] == 0:
            lines.append(f"  WARNING: Zero contacts detected - ligand may have unbound!")
        all_stable &= r['is_stable']

    # Min Distance
    if 'mindist' in results:
        r = results['mindist']
        lines.append(f"\n[MIN PROTEIN-LIGAND DISTANCE]")
        lines.append(f"  Mean:  {r['mean']:.3f} nm")
        lines.append(f"  Std:   {r['std']:.3f} nm")
        lines.append(f"  Max:   {r['max']:.3f} nm")
        status = "STABLE" if r['is_stable'] else "UNSTABLE"
        lines.append(f"  Status: {status}")
        if r['max'] > 0.8:
            lines.append(f"  WARNING: Max distance > 0.8 nm suggests potential unbinding!")
        all_stable &= r['is_stable']

    # RMSF
    if 'rmsf' in results:
        r = results['rmsf']
        lines.append(f"\n[RMSF (Per-residue Flexibility)]")
        lines.append(f"  Mean:  {r['mean']:.3f} nm")
        lines.append(f"  Max:   {r['max']:.3f} nm")
        lines.append(f"  High flexibility residues (>0.3 nm): {r['high_flexibility_residues']}/{r['total_residues']}")

    # Overall Assessment
    lines.append("\n" + "=" * 70)
    lines.append("OVERALL ASSESSMENT")
    lines.append("=" * 70)

    if all_stable:
        lines.append("\nSYSTEM STATUS: STABLE")
        lines.append("\nThe protein-ligand complex appears stable:")
        lines.append("  - RMSD values have plateaued")
        lines.append("  - Energy is stable")
        lines.append("  - Ligand maintains binding contacts")
        lines.append("\nRecommendation: System is suitable for production MD or free energy calculations.")
    else:
        lines.append("\nSYSTEM STATUS: POTENTIAL INSTABILITY DETECTED")
        lines.append("\nIssues detected:")
        if 'rmsd_protein' in results and not results['rmsd_protein']['is_stable']:
            lines.append("  - Protein RMSD shows drift or high fluctuation")
        if 'rmsd_ligand' in results and not results['rmsd_ligand']['is_stable']:
            lines.append("  - Ligand RMSD shows drift (pose instability)")
        if 'energy' in results and not results['energy']['is_stable']:
            lines.append("  - Energy shows significant drift")
        if 'gyration' in results and not results['gyration']['is_stable']:
            lines.append("  - Radius of gyration unstable (potential unfolding)")
        if 'contacts' in results and not results['contacts']['is_stable']:
            lines.append("  - Protein-ligand contacts decreasing (potential unbinding)")
        if 'mindist' in results and not results['mindist']['is_stable']:
            lines.append("  - Min distance increasing (ligand moving away)")
        lines.append("\nRecommendation: Review trajectory, consider longer equilibration or")
        lines.append("  checking initial structure/parameters.")

    lines.append("\n" + "=" * 70)

    report = "\n".join(lines)
    print(report)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}")

    return all_stable


def main():
    parser = argparse.ArgumentParser(
        description="Analyze stability test results from GROMACS XVG files"
    )
    parser.add_argument('--dir', '-d', required=True,
                        help='Directory containing XVG files')
    parser.add_argument('--output', '-o',
                        help='Output file for report (optional)')

    args = parser.parse_args()
    analysis_dir = Path(args.dir)

    if not analysis_dir.exists():
        print(f"ERROR: Directory not found: {analysis_dir}")
        sys.exit(1)

    results = {}

    # Analyze each file if present
    files = {
        'rmsd_protein': 'rmsd_protein.xvg',
        'rmsd_ligand': 'rmsd_ligand.xvg',
        'energy': 'energy.xvg',
        'gyration': 'gyrate.xvg',
        'contacts': 'numcont.xvg',
        'mindist': 'mindist.xvg',
        'rmsf': 'rmsf_protein.xvg'
    }

    for key, filename in files.items():
        filepath = analysis_dir / filename
        if filepath.exists():
            try:
                if key == 'rmsd_protein':
                    results[key] = analyze_rmsd(filepath, "Protein RMSD")
                elif key == 'rmsd_ligand':
                    results[key] = analyze_rmsd(filepath, "Ligand RMSD")
                elif key == 'energy':
                    results[key] = analyze_energy(filepath)
                elif key == 'gyration':
                    results[key] = analyze_gyration(filepath)
                elif key == 'contacts':
                    results[key] = analyze_contacts(filepath)
                elif key == 'mindist':
                    results[key] = analyze_mindist(filepath)
                elif key == 'rmsf':
                    results[key] = analyze_rmsf(filepath)
            except Exception as e:
                print(f"Warning: Could not analyze {filename}: {e}")

    if not results:
        print("ERROR: No XVG files found to analyze")
        sys.exit(1)

    is_stable = print_report(results, args.output)
    sys.exit(0 if is_stable else 1)


if __name__ == '__main__':
    main()
