#!/usr/bin/env python3
"""
analyze_rbfe_validation.py

Analyze results from 4-way RBFE validation workflow.

Compares the 4 RBFE calculations:
  1. forward_a_aligned: acry → chlo (A-aligned) → +ΔG
  2. forward_b_aligned: acry → chlo (B-aligned) → +ΔG
  3. reverse_b_aligned: chlo → acry (B-aligned) → -ΔG
  4. reverse_a_aligned: chlo → acry (A-aligned) → -ΔG

Consistency checks:
  - Run 1 ≈ Run 2 (forward from both structures)
  - Run 3 ≈ Run 4 (reverse from both structures)
  - Run 1 ≈ -Run 4 (cycle closure A-aligned)
  - Run 2 ≈ -Run 3 (cycle closure B-aligned)

If all agree within error bars → high confidence in ΔΔG

Usage:
    python analyze_rbfe_validation.py Outputs/Covalent/RBFE_validation
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Try to import analysis libraries
try:
    HAS_ALCHEMLYB = True
    from alchemlyb.parsing.gmx import extract_dHdl, extract_u_nk
    from alchemlyb.estimators import BAR, MBAR, TI
    import pandas as pd
except ImportError:
    HAS_ALCHEMLYB = False
    print("Note: alchemlyb not found, will use GROMACS bar for analysis")


def find_gmx() -> Optional[str]:
    """Find GROMACS executable."""
    for cmd in ['gmx', 'gmx_mpi']:
        try:
            result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                return cmd
        except FileNotFoundError:
            continue
    return None


def analyze_with_gmx_bar(rbfe_dir: Path, gmx: str = 'gmx') -> Dict:
    """
    Analyze RBFE using GROMACS BAR.

    Returns:
        Dict with 'dG', 'dG_err', 'method'
    """
    # Find all dhdl files
    dhdl_files = sorted(rbfe_dir.glob('lambda*/prod.xvg'))

    if len(dhdl_files) < 2:
        return {'dG': None, 'dG_err': None, 'method': 'gmx_bar', 'error': 'Not enough dhdl files'}

    # Run gmx bar
    file_list = ' '.join(str(f) for f in dhdl_files)
    output_file = rbfe_dir / 'bar_results.xvg'

    cmd = f"{gmx} bar -f {file_list} -o {output_file}"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=rbfe_dir)

    if result.returncode != 0:
        return {'dG': None, 'dG_err': None, 'method': 'gmx_bar', 'error': result.stderr}

    # Parse output for total dG
    dG = None
    dG_err = None

    for line in result.stdout.split('\n'):
        if 'total' in line.lower() and 'kj/mol' in line.lower():
            # Parse line like "total ... XX.XX +/- YY.YY kJ/mol"
            parts = line.split()
            for i, part in enumerate(parts):
                if part == '+/-' and i > 0 and i < len(parts) - 1:
                    try:
                        dG = float(parts[i-1])
                        dG_err = float(parts[i+1])
                    except (ValueError, IndexError):
                        pass
                    break

    return {
        'dG': dG,
        'dG_err': dG_err,
        'method': 'gmx_bar',
        'n_windows': len(dhdl_files)
    }


def analyze_with_alchemlyb(rbfe_dir: Path, temperature: float = 300) -> Dict:
    """
    Analyze RBFE using alchemlyb (BAR and MBAR estimators).

    Returns:
        Dict with 'dG', 'dG_err', 'method', plus MBAR results if available
    """
    if not HAS_ALCHEMLYB:
        return {'dG': None, 'dG_err': None, 'method': 'alchemlyb', 'error': 'alchemlyb not installed'}

    # Find all xvg files
    xvg_files = sorted(rbfe_dir.glob('lambda*/prod.xvg'))

    if len(xvg_files) < 2:
        return {'dG': None, 'dG_err': None, 'method': 'alchemlyb', 'error': 'Not enough xvg files'}

    try:
        # Extract u_nk for BAR/MBAR
        u_nk_list = []
        for f in xvg_files:
            try:
                u_nk = extract_u_nk(str(f), T=temperature)
                u_nk_list.append(u_nk)
            except Exception as e:
                print(f"    Warning: Could not parse {f}: {e}")

        if not u_nk_list:
            return {'dG': None, 'dG_err': None, 'method': 'alchemlyb', 'error': 'No valid xvg files'}

        u_nk_all = pd.concat(u_nk_list)

        # BAR estimator
        bar = BAR()
        bar.fit(u_nk_all)

        # Convert kT to kJ/mol (at 300K, kT ≈ 2.494 kJ/mol)
        kT = 8.314462618 * temperature / 1000  # kJ/mol

        dG_bar = bar.delta_f_.iloc[0, -1] * kT
        dG_err_bar = bar.d_delta_f_.iloc[0, -1] * kT

        result = {
            'dG': dG_bar,
            'dG_err': dG_err_bar,
            'method': 'alchemlyb_BAR',
            'n_windows': len(xvg_files)
        }

        # Try MBAR as well
        try:
            mbar = MBAR()
            mbar.fit(u_nk_all)
            result['dG_mbar'] = mbar.delta_f_.iloc[0, -1] * kT
            result['dG_err_mbar'] = mbar.d_delta_f_.iloc[0, -1] * kT
        except Exception as e:
            result['mbar_error'] = str(e)

        return result

    except Exception as e:
        return {'dG': None, 'dG_err': None, 'method': 'alchemlyb', 'error': str(e)}


def check_completion(rbfe_dir: Path) -> Dict:
    """
    Check completion status of RBFE calculation.

    Returns:
        Dict with 'complete', 'n_complete', 'n_total', 'status'
    """
    lambda_dirs = sorted(rbfe_dir.glob('lambda*/'))
    n_total = len(lambda_dirs)

    if n_total == 0:
        return {'complete': False, 'n_complete': 0, 'n_total': 0, 'status': 'not_started'}

    n_complete = 0
    for lambda_dir in lambda_dirs:
        if (lambda_dir / 'prod.gro').exists():
            n_complete += 1

    return {
        'complete': n_complete == n_total,
        'n_complete': n_complete,
        'n_total': n_total,
        'status': 'complete' if n_complete == n_total else f'{n_complete}/{n_total}'
    }


def analyze_rbfe_validation(validation_dir: Path, temperature: float = 300) -> Dict:
    """
    Analyze all 4 RBFE calculations and check consistency.

    Args:
        validation_dir: Directory containing RBFE validation setup
        temperature: Temperature in Kelvin

    Returns:
        Dict with analysis results for all 4 calculations and consistency metrics
    """
    validation_dir = Path(validation_dir)
    rbfe_dir = validation_dir / 'rbfe'

    if not rbfe_dir.exists():
        print(f"ERROR: RBFE directory not found: {rbfe_dir}")
        return None

    print("=" * 70)
    print("RBFE VALIDATION ANALYSIS")
    print("=" * 70)

    # Define expected calculations
    calculations = [
        {'name': '1_forward_a_aligned', 'direction': 'forward', 'structure': 'A-aligned', 'expected_sign': '+'},
        {'name': '2_forward_b_aligned', 'direction': 'forward', 'structure': 'B-aligned', 'expected_sign': '+'},
        {'name': '3_reverse_b_aligned', 'direction': 'reverse', 'structure': 'B-aligned', 'expected_sign': '-'},
        {'name': '4_reverse_a_aligned', 'direction': 'reverse', 'structure': 'A-aligned', 'expected_sign': '-'},
    ]

    gmx = find_gmx()
    results = []

    print("\n[1] Checking completion status...")
    for calc in calculations:
        calc_dir = rbfe_dir / calc['name']
        status = check_completion(calc_dir)
        calc['status'] = status
        print(f"  {calc['name']}: {status['status']}")

    print("\n[2] Analyzing free energies...")
    for calc in calculations:
        calc_dir = rbfe_dir / calc['name']

        if not calc['status']['complete']:
            calc['analysis'] = {'dG': None, 'dG_err': None, 'error': 'incomplete'}
            print(f"  {calc['name']}: INCOMPLETE - skipping analysis")
            continue

        # Try alchemlyb first, fall back to gmx bar
        if HAS_ALCHEMLYB:
            analysis = analyze_with_alchemlyb(calc_dir, temperature)
        else:
            analysis = analyze_with_gmx_bar(calc_dir, gmx)

        calc['analysis'] = analysis

        if analysis['dG'] is not None:
            sign = '+' if analysis['dG'] > 0 else ''
            print(f"  {calc['name']}: {sign}{analysis['dG']:.2f} +/- {analysis['dG_err']:.2f} kJ/mol")
        else:
            print(f"  {calc['name']}: FAILED - {analysis.get('error', 'unknown error')}")

        results.append(calc)

    # Consistency analysis
    print("\n[3] Consistency checks...")

    # Extract dG values
    dG_values = {}
    for calc in results:
        if calc['analysis']['dG'] is not None:
            dG_values[calc['name']] = (calc['analysis']['dG'], calc['analysis']['dG_err'])

    consistency = {}

    # Check 1: Forward calculations should agree
    if '1_forward_a_aligned' in dG_values and '2_forward_b_aligned' in dG_values:
        dG1, err1 = dG_values['1_forward_a_aligned']
        dG2, err2 = dG_values['2_forward_b_aligned']
        diff = abs(dG1 - dG2)
        combined_err = np.sqrt(err1**2 + err2**2)
        agreement = diff <= 2 * combined_err  # Within 2 sigma

        consistency['forward_agreement'] = {
            'dG1': dG1, 'dG2': dG2,
            'difference': diff,
            'combined_error': combined_err,
            'agreement': agreement
        }
        status = "AGREE" if agreement else "DISAGREE"
        print(f"  Forward (Run1 vs Run2): diff = {diff:.2f} kJ/mol, 2σ = {2*combined_err:.2f} → {status}")

    # Check 2: Reverse calculations should agree
    if '3_reverse_b_aligned' in dG_values and '4_reverse_a_aligned' in dG_values:
        dG3, err3 = dG_values['3_reverse_b_aligned']
        dG4, err4 = dG_values['4_reverse_a_aligned']
        diff = abs(dG3 - dG4)
        combined_err = np.sqrt(err3**2 + err4**2)
        agreement = diff <= 2 * combined_err

        consistency['reverse_agreement'] = {
            'dG3': dG3, 'dG4': dG4,
            'difference': diff,
            'combined_error': combined_err,
            'agreement': agreement
        }
        status = "AGREE" if agreement else "DISAGREE"
        print(f"  Reverse (Run3 vs Run4): diff = {diff:.2f} kJ/mol, 2σ = {2*combined_err:.2f} → {status}")

    # Check 3: Cycle closure A-aligned (Run1 ≈ -Run4)
    if '1_forward_a_aligned' in dG_values and '4_reverse_a_aligned' in dG_values:
        dG1, err1 = dG_values['1_forward_a_aligned']
        dG4, err4 = dG_values['4_reverse_a_aligned']
        diff = abs(dG1 + dG4)  # Should sum to ~0
        combined_err = np.sqrt(err1**2 + err4**2)
        agreement = diff <= 2 * combined_err

        consistency['cycle_a_aligned'] = {
            'dG_forward': dG1, 'dG_reverse': dG4,
            'sum': dG1 + dG4,
            'combined_error': combined_err,
            'agreement': agreement
        }
        status = "CLOSED" if agreement else "NOT CLOSED"
        print(f"  Cycle A-aligned (Run1 + Run4): sum = {dG1 + dG4:.2f} kJ/mol, 2σ = {2*combined_err:.2f} → {status}")

    # Check 4: Cycle closure B-aligned (Run2 ≈ -Run3)
    if '2_forward_b_aligned' in dG_values and '3_reverse_b_aligned' in dG_values:
        dG2, err2 = dG_values['2_forward_b_aligned']
        dG3, err3 = dG_values['3_reverse_b_aligned']
        diff = abs(dG2 + dG3)
        combined_err = np.sqrt(err2**2 + err3**2)
        agreement = diff <= 2 * combined_err

        consistency['cycle_b_aligned'] = {
            'dG_forward': dG2, 'dG_reverse': dG3,
            'sum': dG2 + dG3,
            'combined_error': combined_err,
            'agreement': agreement
        }
        status = "CLOSED" if agreement else "NOT CLOSED"
        print(f"  Cycle B-aligned (Run2 + Run3): sum = {dG2 + dG3:.2f} kJ/mol, 2σ = {2*combined_err:.2f} → {status}")

    # Calculate best estimate
    print("\n[4] Final ΔΔG estimate...")

    if dG_values:
        # Average of forward calculations
        forward_dGs = []
        forward_errs = []
        if '1_forward_a_aligned' in dG_values:
            forward_dGs.append(dG_values['1_forward_a_aligned'][0])
            forward_errs.append(dG_values['1_forward_a_aligned'][1])
        if '2_forward_b_aligned' in dG_values:
            forward_dGs.append(dG_values['2_forward_b_aligned'][0])
            forward_errs.append(dG_values['2_forward_b_aligned'][1])

        # Negated reverse calculations
        if '3_reverse_b_aligned' in dG_values:
            forward_dGs.append(-dG_values['3_reverse_b_aligned'][0])
            forward_errs.append(dG_values['3_reverse_b_aligned'][1])
        if '4_reverse_a_aligned' in dG_values:
            forward_dGs.append(-dG_values['4_reverse_a_aligned'][0])
            forward_errs.append(dG_values['4_reverse_a_aligned'][1])

        if forward_dGs:
            # Weighted average by inverse variance
            weights = [1/(e**2) for e in forward_errs]
            dG_avg = np.average(forward_dGs, weights=weights)
            dG_err_avg = 1 / np.sqrt(sum(weights))

            # Also calculate simple average and std
            dG_simple_avg = np.mean(forward_dGs)
            dG_simple_std = np.std(forward_dGs)

            print(f"\n  Weighted average: {dG_avg:.2f} +/- {dG_err_avg:.2f} kJ/mol")
            print(f"  Simple average:   {dG_simple_avg:.2f} +/- {dG_simple_std:.2f} kJ/mol")
            print(f"  (acry → chlo transformation)")

            # Convert to kcal/mol
            print(f"\n  In kcal/mol: {dG_avg/4.184:.2f} +/- {dG_err_avg/4.184:.2f} kcal/mol")

            consistency['final_estimate'] = {
                'dG_weighted': dG_avg,
                'dG_err_weighted': dG_err_avg,
                'dG_simple': dG_simple_avg,
                'dG_std': dG_simple_std,
                'n_calculations': len(forward_dGs)
            }

    # Overall assessment
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_agree = all([
        consistency.get('forward_agreement', {}).get('agreement', False),
        consistency.get('reverse_agreement', {}).get('agreement', False),
        consistency.get('cycle_a_aligned', {}).get('agreement', False),
        consistency.get('cycle_b_aligned', {}).get('agreement', False)
    ])

    if all_agree:
        print("\n  All consistency checks PASSED")
        print("  High confidence in ΔΔG estimate")
    else:
        print("\n  Some consistency checks FAILED")
        print("  Consider:")
        print("    - Running longer simulations")
        print("    - Adding more lambda windows")
        print("    - Using replica exchange (HREX)")

    # Save results
    output = {
        'calculations': [{
            'name': c['name'],
            'direction': c['direction'],
            'structure': c['structure'],
            'status': c['status'],
            'analysis': c['analysis']
        } for c in results],
        'consistency': consistency,
        'temperature': temperature
    }

    output_file = validation_dir / 'rbfe_validation_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_file}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description='Analyze 4-way RBFE validation results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('validation_dir',
                        help='Directory containing RBFE validation setup')
    parser.add_argument('--temperature', type=float, default=300,
                        help='Temperature in Kelvin (default: 300)')

    args = parser.parse_args()

    results = analyze_rbfe_validation(
        validation_dir=Path(args.validation_dir),
        temperature=args.temperature
    )

    if results is None:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
