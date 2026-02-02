#!/usr/bin/env python3
"""
06_analyze_pmf.py - Calculate PMF and Free Energies from Umbrella Sampling

Performs WHAM (Weighted Histogram Analysis Method) or MBAR analysis on
umbrella sampling results to obtain the potential of mean force (PMF)
along the S-C reaction coordinate.

Usage:
    python 06_analyze_pmf.py --config config.yaml [--method wham|mbar]

Output:
    {system}/
        ├── pmf.dat              - PMF data (distance, energy, error)
        ├── pmf.png              - PMF plot
        ├── histogram.png        - Sampling histogram
        ├── free_energy.json     - Extracted free energies
        └── summary.txt          - Human-readable summary
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_gmx() -> Optional[str]:
    """Find GROMACS executable."""
    for cmd in ['gmx', 'gmx_mpi', 'gmx_mimic']:
        try:
            result = subprocess.run([cmd, '--version'], capture_output=True)
            if result.returncode == 0:
                return cmd
        except FileNotFoundError:
            continue
    return None


def extract_pullx_data(window_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract distance data from GROMACS pull output."""
    
    # Look for pullx.xvg file
    pullx_file = window_dir / 'prod_pullx.xvg'
    if not pullx_file.exists():
        pullx_file = window_dir / 'pullx.xvg'
    if not pullx_file.exists():
        return None
    
    times = []
    distances = []
    
    with open(pullx_file) as f:
        for line in f:
            if line.startswith('#') or line.startswith('@'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                times.append(float(parts[0]))
                distances.append(float(parts[1]))
    
    return np.array(times), np.array(distances)


def run_gmx_wham(
    system_dir: Path,
    config: dict
) -> Optional[Path]:
    """Run GROMACS WHAM analysis."""
    
    gmx = find_gmx()
    if not gmx:
        print("    ERROR: GROMACS not found")
        return None
    
    windows_dir = system_dir / 'windows'
    window_dirs = sorted(windows_dir.glob('window_*'))
    
    # Create file lists for WHAM
    tpr_files = []
    pullf_files = []
    
    for window_dir in window_dirs:
        tpr = window_dir / 'prod.tpr'
        pullf = window_dir / 'prod_pullf.xvg'
        
        if not pullf.exists():
            pullf = window_dir / 'pullf.xvg'
        
        if tpr.exists() and pullf.exists():
            tpr_files.append(str(tpr))
            pullf_files.append(str(pullf))
    
    if len(tpr_files) < 3:
        print(f"    ERROR: Not enough windows with data ({len(tpr_files)} found)")
        return None
    
    # Write file lists
    tpr_list = system_dir / 'tpr_files.dat'
    pullf_list = system_dir / 'pullf_files.dat'
    
    with open(tpr_list, 'w') as f:
        f.write('\n'.join(tpr_files))
    
    with open(pullf_list, 'w') as f:
        f.write('\n'.join(pullf_files))
    
    # Run WHAM
    analysis_cfg = config.get('analysis', {}).get('pmf', {})
    temp = analysis_cfg.get('temperature_K', 300)
    bins = int((5.0 - 1.6) / analysis_cfg.get('bin_width_nm', 0.005))
    tol = analysis_cfg.get('tolerance', 1e-6)
    
    pmf_file = system_dir / 'pmf.xvg'
    hist_file = system_dir / 'histogram.xvg'
    
    cmd = [
        gmx, 'wham',
        '-it', str(tpr_list),
        '-if', str(pullf_list),
        '-o', str(pmf_file),
        '-hist', str(hist_file),
        '-temp', str(temp),
        '-bins', str(bins),
        '-tol', str(tol),
        '-unit', 'kJ',
    ]
    
    # Add bootstrap if configured
    bootstrap_cfg = analysis_cfg.get('bootstrap', {})
    if bootstrap_cfg.get('enabled', False):
        n_boot = bootstrap_cfg.get('n_samples', 200)
        bsprof_file = system_dir / 'bsprof.xvg'
        cmd.extend(['-bs-method', 'b-hist', '-nBootstrap', str(n_boot), '-bsprof', str(bsprof_file)])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=system_dir
        )
        
        if result.returncode != 0:
            print(f"    WHAM error: {result.stderr[:500]}")
            return None
        
        return pmf_file
    
    except Exception as e:
        print(f"    ERROR running WHAM: {e}")
        return None


def parse_xvg(xvg_file: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Parse GROMACS XVG file."""
    x = []
    y = []
    err = []
    
    with open(xvg_file) as f:
        for line in f:
            if line.startswith('#') or line.startswith('@'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                x.append(float(parts[0]))
                y.append(float(parts[1]))
                if len(parts) >= 3:
                    err.append(float(parts[2]))
    
    return (
        np.array(x),
        np.array(y),
        np.array(err) if err else None
    )


def analyze_pmf(
    pmf_data: Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
    config: dict
) -> Dict:
    """Analyze PMF to extract key free energies."""
    
    distances, energies, errors = pmf_data
    
    # Convert distances from nm to Angstroms for reporting
    distances_A = distances * 10
    
    # Find minima and transition state
    
    # Bound state: minimum near covalent bond distance (~1.8 Å)
    bound_mask = distances_A < 2.2
    if np.any(bound_mask):
        bound_idx = np.argmin(energies[bound_mask])
        bound_energy = energies[bound_mask][bound_idx]
        bound_dist = distances_A[bound_mask][bound_idx]
    else:
        bound_idx = 0
        bound_energy = energies[0]
        bound_dist = distances_A[0]
    
    # Unbound state: minimum at large distance (>4.0 Å)
    unbound_mask = distances_A > 4.0
    if np.any(unbound_mask):
        unbound_idx = np.argmin(energies[unbound_mask])
        unbound_energy = energies[unbound_mask][unbound_idx]
        unbound_dist = distances_A[unbound_mask][unbound_idx]
    else:
        unbound_idx = -1
        unbound_energy = energies[-1]
        unbound_dist = distances_A[-1]
    
    # Transition state: maximum between bound and unbound minima
    ts_mask = (distances_A > 2.2) & (distances_A < 4.0)
    if np.any(ts_mask):
        ts_idx = np.argmax(energies[ts_mask])
        ts_energy = energies[ts_mask][ts_idx]
        ts_dist = distances_A[ts_mask][ts_idx]
    else:
        ts_energy = np.max(energies)
        ts_dist = distances_A[np.argmax(energies)]
    
    # Calculate key energies (relative to unbound state)
    # Shift so unbound = 0
    dG_bind = bound_energy - unbound_energy  # kJ/mol (should be negative for favorable)
    dG_barrier = ts_energy - unbound_energy   # Activation barrier from unbound
    dG_barrier_rev = ts_energy - bound_energy  # Reverse barrier (bond breaking)
    
    # Convert to kcal/mol
    kj_to_kcal = 0.239006
    
    results = {
        'bound_state': {
            'distance_A': float(bound_dist),
            'energy_kJ': float(bound_energy),
        },
        'transition_state': {
            'distance_A': float(ts_dist),
            'energy_kJ': float(ts_energy),
        },
        'unbound_state': {
            'distance_A': float(unbound_dist),
            'energy_kJ': float(unbound_energy),
        },
        'free_energies': {
            'dG_covalent_kJ': float(dG_bind),
            'dG_covalent_kcal': float(dG_bind * kj_to_kcal),
            'dG_barrier_kJ': float(dG_barrier),
            'dG_barrier_kcal': float(dG_barrier * kj_to_kcal),
            'dG_barrier_reverse_kJ': float(dG_barrier_rev),
            'dG_barrier_reverse_kcal': float(dG_barrier_rev * kj_to_kcal),
        },
        'units': {
            'distance': 'Angstroms',
            'energy': 'kJ/mol (also kcal/mol provided)',
        }
    }
    
    return results


def plot_pmf(
    pmf_data: Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
    analysis: Dict,
    output_path: Path
) -> None:
    """Create PMF plot."""
    if not HAS_MATPLOTLIB:
        print("    Matplotlib not available, skipping plot")
        return
    
    distances, energies, errors = pmf_data
    distances_A = distances * 10  # Convert to Angstroms
    
    # Shift energies so minimum is at 0
    energies_shifted = energies - np.min(energies)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot PMF
    if errors is not None:
        ax.fill_between(distances_A, energies_shifted - errors, energies_shifted + errors,
                       alpha=0.3, color='blue', label='Error')
    ax.plot(distances_A, energies_shifted, 'b-', linewidth=2, label='PMF')
    
    # Mark key states
    fe = analysis['free_energies']
    bound = analysis['bound_state']
    ts = analysis['transition_state']
    unbound = analysis['unbound_state']
    
    ax.axvline(bound['distance_A'], color='green', linestyle='--', alpha=0.5, label='Bound')
    ax.axvline(ts['distance_A'], color='red', linestyle='--', alpha=0.5, label='TS')
    ax.axvline(unbound['distance_A'], color='orange', linestyle='--', alpha=0.5, label='Unbound')
    
    # Add annotation
    ax.annotate(f"ΔG‡ = {fe['dG_barrier_kcal']:.1f} kcal/mol",
               xy=(ts['distance_A'], ts['energy_kJ'] - np.min(energies)),
               xytext=(ts['distance_A'] + 0.5, ts['energy_kJ'] - np.min(energies) + 5),
               fontsize=10)
    
    ax.annotate(f"ΔG = {fe['dG_covalent_kcal']:.1f} kcal/mol",
               xy=(bound['distance_A'], bound['energy_kJ'] - np.min(energies)),
               xytext=(bound['distance_A'] + 0.3, bound['energy_kJ'] - np.min(energies) - 10),
               fontsize=10)
    
    ax.set_xlabel('S–C Distance (Å)', fontsize=12)
    ax.set_ylabel('Free Energy (kJ/mol)', fontsize=12)
    ax.set_title('QM/MM Umbrella Sampling PMF', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim(distances_A[0] - 0.2, distances_A[-1] + 0.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_summary(
    system_info: Dict,
    analysis: Dict,
    output_path: Path
) -> None:
    """Write human-readable summary."""
    
    fe = analysis['free_energies']
    
    summary = f"""
================================================================================
QM/MM Umbrella Sampling Results
================================================================================

System: {system_info['receptor']} / {system_info['inhibitor']} / {system_info['warhead']}
Warhead: {system_info['warhead_name']}

--------------------------------------------------------------------------------
Key States
--------------------------------------------------------------------------------
                    Distance (Å)    Energy (kJ/mol)
Bound (covalent):   {analysis['bound_state']['distance_A']:.2f}            {analysis['bound_state']['energy_kJ']:.1f}
Transition State:   {analysis['transition_state']['distance_A']:.2f}            {analysis['transition_state']['energy_kJ']:.1f}
Unbound:            {analysis['unbound_state']['distance_A']:.2f}            {analysis['unbound_state']['energy_kJ']:.1f}

--------------------------------------------------------------------------------
Free Energies (relative to unbound state)
--------------------------------------------------------------------------------
ΔG_covalent:     {fe['dG_covalent_kJ']:8.1f} kJ/mol  ({fe['dG_covalent_kcal']:6.1f} kcal/mol)
ΔG‡ (forward):   {fe['dG_barrier_kJ']:8.1f} kJ/mol  ({fe['dG_barrier_kcal']:6.1f} kcal/mol)
ΔG‡ (reverse):   {fe['dG_barrier_reverse_kJ']:8.1f} kJ/mol  ({fe['dG_barrier_reverse_kcal']:6.1f} kcal/mol)

--------------------------------------------------------------------------------
Interpretation
--------------------------------------------------------------------------------
"""
    
    if fe['dG_covalent_kcal'] < -5:
        summary += "• Covalent bond formation is THERMODYNAMICALLY FAVORABLE\n"
    elif fe['dG_covalent_kcal'] > 5:
        summary += "• Covalent bond formation is THERMODYNAMICALLY UNFAVORABLE\n"
    else:
        summary += "• Covalent bond formation is roughly THERMONEUTRAL\n"
    
    if fe['dG_barrier_kcal'] < 15:
        summary += "• Reaction barrier is LOW - fast covalent modification expected\n"
    elif fe['dG_barrier_kcal'] < 25:
        summary += "• Reaction barrier is MODERATE\n"
    else:
        summary += "• Reaction barrier is HIGH - slow covalent modification expected\n"
    
    summary += """
================================================================================
"""
    
    with open(output_path, 'w') as f:
        f.write(summary)


def process_system(system_dir: Path, config: dict) -> bool:
    """Analyze PMF for a single system."""
    
    with open(system_dir / 'system_info.json') as f:
        system_info = json.load(f)
    
    print(f"  Processing: {system_info['receptor']}/{system_info['inhibitor']}/{system_info['warhead']}")
    
    # Run WHAM
    print("    Running WHAM analysis...")
    pmf_file = run_gmx_wham(system_dir, config)
    
    if pmf_file is None or not pmf_file.exists():
        print("    ERROR: WHAM failed")
        return False
    
    # Parse PMF data
    pmf_data = parse_xvg(pmf_file)
    
    # Analyze PMF
    print("    Extracting free energies...")
    analysis = analyze_pmf(pmf_data, config)
    
    # Save results
    with open(system_dir / 'free_energy.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Create plot
    print("    Creating plots...")
    plot_pmf(pmf_data, analysis, system_dir / 'pmf.png')
    
    # Write summary
    write_summary(system_info, analysis, system_dir / 'summary.txt')
    
    # Print key results
    fe = analysis['free_energies']
    print(f"    ✓ ΔG_covalent = {fe['dG_covalent_kcal']:.1f} kcal/mol")
    print(f"    ✓ ΔG‡ = {fe['dG_barrier_kcal']:.1f} kcal/mol")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Analyze PMF from umbrella sampling")
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'config.yaml')
    parser.add_argument('--method', choices=['wham', 'mbar'], default='wham',
                       help='Analysis method')
    parser.add_argument('--receptors', nargs='+', default=None)
    parser.add_argument('--inhibitors', nargs='+', default=None)
    parser.add_argument('--warheads', nargs='+', default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    base_dir = args.config.parent
    
    output_dir = Path(config['paths']['output_dir'])
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()
    
    print("=" * 60)
    print("Step 6: PMF Analysis")
    print("=" * 60)
    print(f"Method: {args.method.upper()}")
    print()
    
    # Get system lists
    receptors = args.receptors or list(config['systems']['receptors'].keys())
    inhibitors = args.inhibitors or config['systems']['inhibitors']
    warheads = args.warheads or list(config['systems']['warheads'].keys())
    
    success_count = 0
    total_count = 0
    
    for receptor in receptors:
        for inhibitor in inhibitors:
            for warhead in warheads:
                system_dir = output_dir / receptor / inhibitor / warhead
                if system_dir.exists() and (system_dir / 'windows').exists():
                    total_count += 1
                    if process_system(system_dir, config):
                        success_count += 1
    
    print()
    print(f"Analyzed {success_count}/{total_count} systems")
    
    return success_count == total_count


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
