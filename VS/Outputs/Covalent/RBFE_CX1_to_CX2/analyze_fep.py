#!/usr/bin/env python3
"""
FEP Analysis Script

Analyzes free energy results from all lambda windows using GROMACS BAR
or alchemlyb (if available).
"""

import os
import subprocess
import glob

# Use the directory where this script is located
output_dir = os.path.dirname(os.path.abspath(__file__)) or "."
n_windows = 14
gmx = "gmx"

print("=" * 60)
print("FEP Analysis")
print("=" * 60)

# Collect all dhdl files
dhdl_files = []
for i in range(n_windows):
    dhdl = os.path.join(output_dir, f"lambda{i:02d}", "prod.xvg")
    if os.path.exists(dhdl):
        dhdl_files.append(dhdl)
    else:
        print(f"WARNING: Missing dhdl file for lambda {i}")

if len(dhdl_files) < n_windows:
    print(f"ERROR: Only found {len(dhdl_files)}/{n_windows} dhdl files")
    exit(1)

print(f"Found {len(dhdl_files)} dhdl files")

# Try alchemlyb first
try:
    from alchemlyb.parsing.gmx import extract_dHdl, extract_u_nk
    from alchemlyb.estimators import BAR, MBAR, TI
    import pandas as pd

    print("\nUsing alchemlyb for analysis...")

    # Extract data
    dhdl_data = pd.concat([extract_dHdl(f, T=300) for f in sorted(dhdl_files)])

    # BAR estimator
    bar = BAR()
    bar.fit(extract_u_nk(sorted(dhdl_files)[0], T=300))  # This needs u_nk

    print(f"\nBAR estimate: {bar.delta_f_.iloc[0, -1]:.3f} +/- {bar.d_delta_f_.iloc[0, -1]:.3f} kT")

except ImportError:
    print("\nalchemlyb not found, using GROMACS BAR...")

    # Use gmx bar
    xvg_list = " ".join(sorted(dhdl_files))
    cmd = f"{gmx} bar -f {xvg_list} -o bar.xvg -oi barint.xvg"

    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           cwd=output_dir)

    if result.returncode == 0:
        print("\nBAR analysis complete!")
        print("Results saved to bar.xvg and barint.xvg")
        print(result.stdout)
    else:
        print("ERROR:", result.stderr)

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)
