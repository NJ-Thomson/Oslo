#!/usr/bin/env python3
"""
ABFE analysis using alchemlyb (MBAR)

This provides more accurate free energy estimates than BAR by using
all lambda states simultaneously via the Multistate Bennett Acceptance Ratio.

Supports both:
- Biggin Lab stage-based workflow (restraints.X, coul.X, vdw.X directories)
- Legacy combined lambda workflow (lambdaXX directories)

Requirements:
    pip install alchemlyb pandas matplotlib

Usage:
    python analyze_alchemlyb.py
"""

import os
import glob
import re
import pandas as pd
import numpy as np

try:
    from alchemlyb.parsing.gmx import extract_u_nk
    from alchemlyb.estimators import MBAR
    from alchemlyb.preprocessing import statistical_inefficiency
    from alchemlyb.visualisation import plot_mbar_overlap_matrix
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_ALCHEMLYB = True
except ImportError:
    print("ERROR: alchemlyb not installed")
    print("Install with: pip install alchemlyb pandas matplotlib")
    HAS_ALCHEMLYB = False
    exit(1)


def natural_sort_key(s):
    """Sort strings with numbers naturally (stage.0, stage.1, ..., stage.10)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def analyze_stage(base_dir, stage_name, temperature=298.15):
    """
    Analyze one stage (restraints, coul, or vdw) of ABFE calculation.

    Args:
        base_dir: Base simulation directory (e.g., fep/simulation)
        stage_name: 'restraints', 'coul', or 'vdw'
        temperature: Simulation temperature in K

    Returns:
        tuple: (dG, dG_err) or (None, None) if failed
    """
    print(f"\n  Analyzing {stage_name} stage...")

    # Find all XVG files for this stage (stage.0, stage.1, etc.)
    pattern = os.path.join(base_dir, f"{stage_name}.*/prod/prod.xvg")
    xvg_files = sorted(glob.glob(pattern), key=natural_sort_key)

    if not xvg_files:
        # Try alternative pattern (dhdl.xvg)
        pattern = os.path.join(base_dir, f"{stage_name}.*/prod/dhdl.xvg")
        xvg_files = sorted(glob.glob(pattern), key=natural_sort_key)

    if not xvg_files:
        print(f"    No XVG files found for {stage_name} stage")
        return None, None

    print(f"    Found {len(xvg_files)} windows")

    # Extract u_nk data
    u_nk_list = []
    for xvg in xvg_files:
        window_name = os.path.basename(os.path.dirname(os.path.dirname(xvg)))
        try:
            data = extract_u_nk(xvg, T=temperature)
            # Apply statistical inefficiency
            try:
                data = statistical_inefficiency(data, series=data.iloc[:, 0])
            except Exception as subsample_err:
                if "covariance" in str(subsample_err).lower() or "variance" in str(subsample_err).lower():
                    print(f"    Note: Skipping subsampling for {window_name} (zero variance)")
                else:
                    raise
            u_nk_list.append(data)
        except Exception as e:
            print(f"    Warning: Could not parse {window_name}: {e}")

    if not u_nk_list:
        return None, None

    # Concatenate and run MBAR
    u_nk = pd.concat(u_nk_list)

    print(f"    Running MBAR estimator...")
    mbar = MBAR()
    mbar.fit(u_nk)

    dG = mbar.delta_f_.iloc[0, -1]
    dG_err = mbar.d_delta_f_.iloc[0, -1]

    print(f"    MBAR result: {dG:.2f} +/- {dG_err:.2f} kJ/mol")

    # Plot overlap matrix
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_mbar_overlap_matrix(mbar.overlap_matrix, ax=ax)
        fig.savefig(os.path.join(base_dir, f"{stage_name}_overlap.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved overlap matrix: {stage_name}_overlap.png")
    except Exception as e:
        print(f"    Warning: Could not plot overlap matrix: {e}")

    return dG, dG_err


def analyze_leg_stages(leg_dir, leg_name, temperature=298.15):
    """
    Analyze one leg using Biggin Lab stage-based structure.

    Directory structure expected:
      leg_dir/fep/simulation/restraints.0/, restraints.1/, ...
      leg_dir/fep/simulation/coul.0/, coul.1/, ...
      leg_dir/fep/simulation/vdw.0/, vdw.1/, ...

    Returns:
        tuple: (total_dG, total_err) or (None, None) if failed
    """
    print(f"\nAnalyzing {leg_name} leg (stage-based)...")

    sim_dir = os.path.join(leg_dir, "fep", "simulation")

    if not os.path.exists(sim_dir):
        print(f"  ERROR: Simulation directory not found: {sim_dir}")
        return None, None

    # Determine which stages to analyze
    if leg_name == 'complex':
        stages = ['restraints', 'coul', 'vdw']
    else:
        stages = ['coul', 'vdw']

    total_dG = 0.0
    total_var = 0.0
    stage_results = {}

    for stage in stages:
        dG, dG_err = analyze_stage(sim_dir, stage, temperature)
        if dG is None:
            print(f"  ERROR: Failed to analyze {stage} stage")
            return None, None

        stage_results[stage] = (dG, dG_err)
        total_dG += dG
        total_var += dG_err ** 2

    total_err = np.sqrt(total_var)

    print(f"\n  {leg_name.upper()} LEG SUMMARY:")
    for stage, (dG, err) in stage_results.items():
        print(f"    dG_{stage}: {dG:8.2f} +/- {err:.2f} kJ/mol")
    print(f"    -----------------------------------")
    print(f"    Total:    {total_dG:8.2f} +/- {total_err:.2f} kJ/mol")

    return total_dG, total_err


def analyze_leg_legacy(leg_dir, leg_name, temperature=298.15):
    """
    Analyze one leg using legacy combined lambda structure.

    Directory structure expected:
      leg_dir/lambda00/, lambda01/, ...

    Returns:
        tuple: (dG, dG_err) or (None, None) if failed
    """
    print(f"\nAnalyzing {leg_name} leg (legacy)...")

    # Find all xvg files
    xvg_files = sorted(glob.glob(os.path.join(leg_dir, "lambda*/prod.xvg")))

    if not xvg_files:
        print(f"  No XVG files found in {leg_dir}/lambda*/prod.xvg")
        return None, None

    print(f"  Found {len(xvg_files)} lambda windows")

    u_nk_list = []
    for xvg in xvg_files:
        try:
            data = extract_u_nk(xvg, T=temperature)
            try:
                data = statistical_inefficiency(data, series=data.iloc[:, 0])
            except Exception as subsample_err:
                if "covariance" in str(subsample_err).lower() or "variance" in str(subsample_err).lower():
                    print(f"  Note: Skipping subsampling for {os.path.basename(os.path.dirname(xvg))} (zero variance)")
                else:
                    raise
            u_nk_list.append(data)
        except Exception as e:
            print(f"  Warning: Could not parse {xvg}: {e}")

    if not u_nk_list:
        return None, None

    u_nk = pd.concat(u_nk_list)

    print("  Running MBAR estimator...")
    mbar = MBAR()
    mbar.fit(u_nk)

    dG = mbar.delta_f_.iloc[0, -1]
    dG_err = mbar.d_delta_f_.iloc[0, -1]

    print(f"  MBAR result: {dG:.2f} +/- {dG_err:.2f} kJ/mol")

    # Plot overlap matrix
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_mbar_overlap_matrix(mbar.overlap_matrix, ax=ax)
        fig.savefig(os.path.join(leg_dir, f"{leg_name}_overlap.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved overlap matrix: {leg_name}_overlap.png")
    except Exception as e:
        print(f"  Warning: Could not plot overlap matrix: {e}")

    return dG, dG_err


def detect_workflow_type(leg_dir):
    """Detect whether stage-based or legacy workflow is used."""
    stage_based = os.path.exists(os.path.join(leg_dir, "fep", "simulation"))
    legacy = len(glob.glob(os.path.join(leg_dir, "lambda*"))) > 0
    return "stage_based" if stage_based else ("legacy" if legacy else None)


def main():
    print("=" * 60)
    print("ABFE ANALYSIS (alchemlyb/MBAR)")
    print("=" * 60)

    # Restraint correction (from MDRestraintsGenerator if available, otherwise analytical)
    dG_restr = {{DG_RESTR}}  # kJ/mol
    print(f"\nRestraint correction: {dG_restr:.2f} kJ/mol")

    # Detect workflow type
    complex_type = detect_workflow_type("complex")
    solvent_type = detect_workflow_type("solvent")

    print(f"\nDetected workflow types:")
    print(f"  Complex: {complex_type}")
    print(f"  Solvent: {solvent_type}")

    if complex_type is None or solvent_type is None:
        print("\nERROR: Could not detect workflow type for one or more legs")
        print("       Expected either fep/simulation/ (stage-based) or lambda*/ (legacy)")
        return

    # Analyze complex leg
    if complex_type == "stage_based":
        dG_complex, err_complex = analyze_leg_stages("complex", "complex")
    else:
        dG_complex, err_complex = analyze_leg_legacy("complex", "complex")

    # Analyze solvent leg
    if solvent_type == "stage_based":
        dG_solvent, err_solvent = analyze_leg_stages("solvent", "solvent")
    else:
        dG_solvent, err_solvent = analyze_leg_legacy("solvent", "solvent")

    if dG_complex is None or dG_solvent is None:
        print("\nERROR: Analysis failed for one or more legs")
        return

    # Calculate binding free energy
    # dG_bind = dG_complex - dG_solvent + dG_restr
    dG_bind = dG_complex - dG_solvent + dG_restr

    # Propagate errors
    err_bind = np.sqrt(err_complex**2 + err_solvent**2)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\ndG_complex:  {dG_complex:8.2f} +/- {err_complex:.2f} kJ/mol")
    print(f"dG_solvent:  {dG_solvent:8.2f} +/- {err_solvent:.2f} kJ/mol")
    print(f"dG_restraint:{dG_restr:8.2f} kJ/mol")
    print("-" * 40)
    print(f"dG_bind:     {dG_bind:8.2f} +/- {err_bind:.2f} kJ/mol")
    print(f"             {dG_bind/4.184:8.2f} +/- {err_bind/4.184:.2f} kcal/mol")
    print("\n" + "=" * 60)

    # Save results
    with open("abfe_results.txt", "w") as f:
        f.write("ABFE Results (alchemlyb/MBAR)\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"dG_complex:   {dG_complex:.4f} +/- {err_complex:.4f} kJ/mol\n")
        f.write(f"dG_solvent:   {dG_solvent:.4f} +/- {err_solvent:.4f} kJ/mol\n")
        f.write(f"dG_restraint: {dG_restr:.4f} kJ/mol\n\n")
        f.write(f"dG_bind:      {dG_bind:.4f} +/- {err_bind:.4f} kJ/mol\n")
        f.write(f"              {dG_bind/4.184:.4f} +/- {err_bind/4.184:.4f} kcal/mol\n")

    print("\nResults saved to: abfe_results.txt")


if __name__ == "__main__":
    main()
