#!/usr/bin/env python3
"""
ABFE Results Analysis and Comparison

Parses results from the forked Biggin ABFE Snakemake workflow, compares
to experimental values, flags convergence issues, and generates a summary.

Output format from Biggin pipeline:
  <results_dir>/<ligand>/<replica>/dG_results.tsv
  <results_dir>/abfe_results.tsv
  <results_dir>/abfe_single_detailed_results.tsv

Convergence diagnostics:
  - MBAR vs TI disagreement (>0.5 kcal/mol = warning)
  - Inter-replica standard deviation (>1.0 kcal/mol = warning)
  - Per-stage Coulomb magnitude flagging

Dependencies:
    pip install pandas numpy pyyaml
    pip install matplotlib  (optional, for plots)

Usage:
    python 10_analyze_abfe.py \\
        --results_dir Outputs/non_covalent/ABFE/<receptor>/abfe_output \\
        --config abfe_config.yml

    # With experimental reference values
    python 10_analyze_abfe.py \\
        --results_dir Outputs/non_covalent/ABFE/<receptor>/abfe_output \\
        --config abfe_config.yml \\
        --experimental '{"ligand_1": -8.38, "ligand_2": -7.20}'

    # Compare two receptors
    python 10_analyze_abfe.py \\
        --results_dir Outputs/non_covalent/ABFE/<receptor_1>/abfe_output \\
        --results_dir2 Outputs/non_covalent/ABFE/<receptor_2>/abfe_output \\
        --labels receptor_1 receptor_2
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Thresholds for convergence diagnostics
MBAR_TI_THRESHOLD = 0.5    # kcal/mol — flag if MBAR-TI difference exceeds this
REPLICA_STD_WARN = 1.0     # kcal/mol — flag if inter-replica std exceeds this
REPLICA_STD_FAIL = 2.0     # kcal/mol — flag as likely unconverged
COUL_MAGNITUDE_WARN = 80   # kcal/mol — flag large Coulomb contributions


def collect_replica_results(results_dir):
    """
    Collect all per-replica dG_results.tsv files.

    Returns:
        dict: {ligand_name: {replica: DataFrame}}
    """
    results_dir = Path(results_dir)
    pattern = str(results_dir / "*" / "*" / "dG_results.tsv")
    files = sorted(glob.glob(pattern))

    data = {}
    for f in files:
        parts = Path(f).parts
        replica = parts[-2]
        ligand = parts[-3].replace("ligand-", "")

        try:
            df = pd.read_csv(f, index_col=0)
            if ligand not in data:
                data[ligand] = {}
            data[ligand][replica] = df
        except Exception as e:
            print(f"  WARNING: Could not read {f}: {e}")

    return data


def diagnose_convergence(ligand_name, replica_dfs):
    """
    Run convergence diagnostics on per-replica results.

    Returns:
        list of diagnostic messages (warnings/errors)
    """
    diagnostics = []

    abfe_mbar = []
    abfe_ti = []

    for rep_id, df in sorted(replica_dfs.items()):
        abfe_row = df[df["step"] == "ABFE"]
        if abfe_row.empty:
            diagnostics.append(f"  WARN: Replica {rep_id} — no ABFE row found")
            continue

        mbar_val = float(abfe_row["MBAR"].iloc[0])
        ti_val = float(abfe_row["TI"].iloc[0])
        abfe_mbar.append(mbar_val)
        abfe_ti.append(ti_val)

        # Check MBAR vs TI agreement per replica
        diff = abs(mbar_val - ti_val)
        if diff > MBAR_TI_THRESHOLD:
            diagnostics.append(
                f"  WARN: Replica {rep_id} — MBAR/TI disagree by "
                f"{diff:.2f} kcal/mol (MBAR={mbar_val:.2f}, TI={ti_val:.2f})"
            )

        # Check individual stage Coulomb magnitudes
        for _, row in df.iterrows():
            if "coul" in str(row.get("step", "")).lower():
                coul_mag = abs(float(row["MBAR"]))
                if coul_mag > COUL_MAGNITUDE_WARN:
                    diagnostics.append(
                        f"  NOTE: Replica {rep_id} — {row['step']} Coulomb "
                        f"|dG| = {coul_mag:.1f} kcal/mol (large)"
                    )

    # Inter-replica consistency
    if len(abfe_mbar) >= 2:
        std_mbar = np.std(abfe_mbar, ddof=1)
        if std_mbar > REPLICA_STD_FAIL:
            diagnostics.append(
                f"  ERROR: Inter-replica std = {std_mbar:.2f} kcal/mol — "
                f"likely unconverged"
            )
        elif std_mbar > REPLICA_STD_WARN:
            diagnostics.append(
                f"  WARN: Inter-replica std = {std_mbar:.2f} kcal/mol — "
                f"consider more sampling"
            )
    elif len(abfe_mbar) == 1:
        diagnostics.append("  NOTE: Only 1 replica — cannot assess convergence")

    return diagnostics


def build_summary_table(all_data, experimental=None):
    """
    Build a summary DataFrame with ABFE results and diagnostics.

    Args:
        all_data: {ligand: {replica: DataFrame}}
        experimental: optional {ligand: dG_exp} in kcal/mol

    Returns:
        pd.DataFrame with columns: ligand, dG_MBAR, dG_TI, std, n_rep,
                                   dG_exp, error, converged
    """
    rows = []
    for ligand, replica_dfs in sorted(all_data.items()):
        mbar_vals = []
        ti_vals = []

        for rep_id, df in sorted(replica_dfs.items()):
            abfe_row = df[df["step"] == "ABFE"]
            if not abfe_row.empty:
                mbar_vals.append(float(abfe_row["MBAR"].iloc[0]))
                ti_vals.append(float(abfe_row["TI"].iloc[0]))

        if not mbar_vals:
            continue

        mean_mbar = np.mean(mbar_vals)
        mean_ti = np.mean(ti_vals)
        std_mbar = np.std(mbar_vals, ddof=1) if len(mbar_vals) > 1 else np.nan

        row = {
            "ligand": ligand,
            "dG_MBAR": round(mean_mbar, 2),
            "dG_TI": round(mean_ti, 2),
            "std": round(std_mbar, 2) if not np.isnan(std_mbar) else "-",
            "n_rep": len(mbar_vals),
            "MBAR_TI_diff": round(abs(mean_mbar - mean_ti), 2),
        }

        if experimental and ligand in experimental:
            dg_exp = experimental[ligand]
            row["dG_exp"] = dg_exp
            row["error"] = round(mean_mbar - dg_exp, 2)
        else:
            row["dG_exp"] = "-"
            row["error"] = "-"

        # Convergence flag
        converged = True
        if isinstance(std_mbar, float) and not np.isnan(std_mbar):
            if std_mbar > REPLICA_STD_FAIL:
                converged = False
        if abs(mean_mbar - mean_ti) > MBAR_TI_THRESHOLD:
            converged = False
        row["converged"] = "YES" if converged else "NO"

        rows.append(row)

    return pd.DataFrame(rows)


def build_stage_table(all_data):
    """
    Build a per-stage breakdown table showing individual leg contributions.

    Returns:
        pd.DataFrame
    """
    rows = []
    for ligand, replica_dfs in sorted(all_data.items()):
        # Average across replicas per step
        step_mbar = {}
        step_ti = {}
        step_counts = {}

        for rep_id, df in sorted(replica_dfs.items()):
            for _, row in df.iterrows():
                step = row["step"]
                if step in ("total", "ABFE"):
                    continue
                if step not in step_mbar:
                    step_mbar[step] = []
                    step_ti[step] = []
                step_mbar[step].append(float(row["MBAR"]))
                step_ti[step].append(float(row["TI"]))

        for step in sorted(step_mbar.keys()):
            rows.append({
                "ligand": ligand,
                "step": step,
                "sys": "complex" if "restraint" in step or "boresch" in step
                       else step_mbar[step][0],  # approximate
                "MBAR_mean": round(np.mean(step_mbar[step]), 2),
                "TI_mean": round(np.mean(step_ti[step]), 2),
                "std": round(np.std(step_mbar[step], ddof=1), 2)
                       if len(step_mbar[step]) > 1 else "-",
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def plot_correlation(summary_df, output_path, label=""):
    """Plot computed vs experimental dG correlation."""
    if not HAS_MPL:
        print("  matplotlib not available — skipping correlation plot")
        return

    df = summary_df[summary_df["dG_exp"] != "-"].copy()
    if df.empty:
        print("  No experimental data — skipping correlation plot")
        return

    df["dG_exp"] = df["dG_exp"].astype(float)
    df["dG_MBAR"] = df["dG_MBAR"].astype(float)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(df["dG_exp"], df["dG_MBAR"], s=80, zorder=5, edgecolors='black')

    for _, row in df.iterrows():
        std = row["std"]
        yerr = float(std) if std != "-" else 0
        ax.errorbar(row["dG_exp"], row["dG_MBAR"], yerr=yerr,
                     fmt='none', color='gray', capsize=3)
        ax.annotate(row["ligand"], (row["dG_exp"], row["dG_MBAR"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Identity line
    all_vals = list(df["dG_exp"]) + list(df["dG_MBAR"])
    lo, hi = min(all_vals) - 1, max(all_vals) + 1
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, label='y=x')
    ax.plot([lo, hi], [lo + 1, hi + 1], 'k:', alpha=0.3)
    ax.plot([lo, hi], [lo - 1, hi - 1], 'k:', alpha=0.3)

    ax.set_xlabel("Experimental dG (kcal/mol)")
    ax.set_ylabel("Computed dG MBAR (kcal/mol)")
    title = "ABFE: Computed vs Experimental"
    if label:
        title += f" ({label})"
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend()

    # Statistics
    r = np.corrcoef(df["dG_exp"], df["dG_MBAR"])[0, 1]
    mae = np.mean(np.abs(df["dG_MBAR"] - df["dG_exp"]))
    rmse = np.sqrt(np.mean((df["dG_MBAR"] - df["dG_exp"])**2))
    ax.text(0.05, 0.95, f"R = {r:.2f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}",
            transform=ax.transAxes, verticalalignment='top',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved correlation plot: {output_path}")


def plot_replica_spread(all_data, output_path, label=""):
    """Plot per-ligand ABFE values showing replica spread."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(max(4, len(all_data) * 1.2), 5))

    ligands = sorted(all_data.keys())
    for i, lig in enumerate(ligands):
        vals = []
        for rep_id, df in sorted(all_data[lig].items()):
            abfe_row = df[df["step"] == "ABFE"]
            if not abfe_row.empty:
                vals.append(float(abfe_row["MBAR"].iloc[0]))
        if vals:
            ax.scatter([i] * len(vals), vals, alpha=0.6, s=50, zorder=5)
            ax.errorbar(i, np.mean(vals),
                        yerr=np.std(vals, ddof=1) if len(vals) > 1 else 0,
                        fmt='_', color='black', markersize=15, capsize=5,
                        linewidth=2, zorder=6)

    ax.set_xticks(range(len(ligands)))
    ax.set_xticklabels(ligands, rotation=45, ha='right')
    ax.set_ylabel("dG_bind MBAR (kcal/mol)")
    title = "ABFE Replica Spread"
    if label:
        title += f" ({label})"
    ax.set_title(title)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved replica spread plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ABFE results from forked Biggin pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--results_dir', '-r', required=True,
                        help='Root directory of ABFE output (contains <ligand>/<rep>/dG_results.tsv)')
    parser.add_argument('--results_dir2', '-r2', default=None,
                        help='Optional second results directory for comparison')
    parser.add_argument('--labels', nargs=2, default=None,
                        help='Labels for the two result sets (e.g., receptor_1 receptor_2)')
    parser.add_argument('--config', '-c', default=None,
                        help='Per-ligand YAML config (for metadata display)')
    parser.add_argument('--experimental', '-e', default=None,
                        help='Experimental dG values as JSON string or path to JSON file')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory for tables and plots (default: results_dir)')

    args = parser.parse_args()

    output_dir = Path(args.output or args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse experimental values
    experimental = None
    if args.experimental:
        if os.path.isfile(args.experimental):
            with open(args.experimental) as f:
                experimental = json.load(f)
        else:
            experimental = json.loads(args.experimental)

    # Load config for metadata
    config_info = {}
    if args.config and HAS_YAML:
        with open(args.config) as f:
            raw = yaml.safe_load(f)
        config_info = raw.get("ligands", {})

    print("=" * 70)
    print("ABFE RESULTS ANALYSIS")
    print("=" * 70)

    # --- Primary results ---
    label1 = args.labels[0] if args.labels else ""
    print(f"\nCollecting results from: {args.results_dir}")
    all_data = collect_replica_results(args.results_dir)

    if not all_data:
        print("ERROR: No results found")
        sys.exit(1)

    print(f"Found {len(all_data)} ligands: {', '.join(sorted(all_data.keys()))}")

    # Convergence diagnostics
    print("\n" + "-" * 70)
    print("CONVERGENCE DIAGNOSTICS")
    print("-" * 70)
    all_pass = True
    for ligand in sorted(all_data.keys()):
        diags = diagnose_convergence(ligand, all_data[ligand])
        n_reps = len(all_data[ligand])
        if diags:
            all_pass = False
            print(f"\n{ligand} ({n_reps} replicas):")
            for d in diags:
                print(d)
        else:
            print(f"\n{ligand} ({n_reps} replicas): OK")

    if all_pass:
        print("\nAll ligands passed convergence checks.")

    # Summary table
    print("\n" + "-" * 70)
    print("SUMMARY TABLE")
    print("-" * 70)
    summary_df = build_summary_table(all_data, experimental)
    print()
    print(summary_df.to_string(index=False))

    # Save summary
    summary_path = output_dir / "abfe_analysis_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")

    # Per-ligand config notes
    if config_info:
        print("\n" + "-" * 70)
        print("CONFIG NOTES")
        print("-" * 70)
        for lig in sorted(all_data.keys()):
            info = config_info.get(lig, {})
            if info:
                notes = info.get("notes", "")
                charge = info.get("formal_charge", 0)
                method = info.get("charge_method", "")
                extras = []
                if charge != 0:
                    extras.append(f"charge={charge}")
                if method:
                    extras.append(f"method={method}")
                extra_str = f" [{', '.join(extras)}]" if extras else ""
                print(f"  {lig}{extra_str}: {notes}")

    # Statistics vs experimental
    if experimental:
        exp_df = summary_df[summary_df["dG_exp"] != "-"].copy()
        if not exp_df.empty:
            exp_df["dG_exp"] = exp_df["dG_exp"].astype(float)
            exp_df["dG_MBAR"] = exp_df["dG_MBAR"].astype(float)
            errors = exp_df["dG_MBAR"] - exp_df["dG_exp"]

            print("\n" + "-" * 70)
            print("COMPARISON TO EXPERIMENT")
            print("-" * 70)
            print(f"  N ligands:  {len(exp_df)}")
            print(f"  R:          {np.corrcoef(exp_df['dG_exp'], exp_df['dG_MBAR'])[0,1]:.3f}")
            print(f"  MUE:        {np.mean(np.abs(errors)):.2f} kcal/mol")
            print(f"  RMSE:       {np.sqrt(np.mean(errors**2)):.2f} kcal/mol")
            print(f"  Max error:  {np.max(np.abs(errors)):.2f} kcal/mol "
                  f"({exp_df.iloc[np.argmax(np.abs(errors))]['ligand']})")

    # Plots
    if HAS_MPL:
        print("\n" + "-" * 70)
        print("GENERATING PLOTS")
        print("-" * 70)
        plot_correlation(summary_df,
                         output_dir / "abfe_correlation.png", label=label1)
        plot_replica_spread(all_data,
                            output_dir / "abfe_replica_spread.png", label=label1)

    # --- Optional second results set for comparison ---
    if args.results_dir2:
        label2 = args.labels[1] if args.labels else "set2"
        print(f"\n\n{'=' * 70}")
        print(f"COMPARISON: {label1} vs {label2}")
        print("=" * 70)

        all_data2 = collect_replica_results(args.results_dir2)
        summary_df2 = build_summary_table(all_data2, experimental)

        # Merge for comparison
        merged = summary_df.merge(summary_df2, on="ligand", suffixes=(f"_{label1}", f"_{label2}"))
        if not merged.empty:
            print()
            print(merged.to_string(index=False))
            merged_path = output_dir / "abfe_comparison.csv"
            merged.to_csv(merged_path, index=False)
            print(f"\nSaved: {merged_path}")

            # Selectivity plot
            if HAS_MPL and len(merged) >= 2:
                fig, ax = plt.subplots(figsize=(max(4, len(merged) * 1.2), 5))
                x = range(len(merged))
                width = 0.35
                ax.bar([i - width/2 for i in x],
                       merged[f"dG_MBAR_{label1}"].astype(float),
                       width, label=label1, alpha=0.8)
                ax.bar([i + width/2 for i in x],
                       merged[f"dG_MBAR_{label2}"].astype(float),
                       width, label=label2, alpha=0.8)
                ax.set_xticks(list(x))
                ax.set_xticklabels(merged["ligand"], rotation=45, ha='right')
                ax.set_ylabel("dG_bind MBAR (kcal/mol)")
                ax.set_title(f"ABFE Comparison: {label1} vs {label2}")
                ax.legend()
                ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
                fig.tight_layout()
                comp_plot = output_dir / "abfe_selectivity_comparison.png"
                fig.savefig(comp_plot, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved comparison plot: {comp_plot}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Files generated:")
    for f in sorted(output_dir.glob("abfe_*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
