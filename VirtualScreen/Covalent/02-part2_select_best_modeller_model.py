#!/usr/bin/env python3
"""
Select the best MODELLER model (lowest DOPE score) and copy to predock file.

Usage:
    python select_best_model.py 4CXA_modeller_results/
    python select_best_model.py 4CXA_modeller_results/ 5ACB_modeller_results/ 5EFQ_modeller_results/
    
    # Or automatically find all *_modeller_results directories:
    python select_best_model.py --auto

Output:
    4CXA_predock.pdb (copied from best model)
"""

import argparse
import re
import shutil
from pathlib import Path


def parse_dope_from_filename(pdb_file):
    """
    Try to get DOPE score from MODELLER output.
    MODELLER writes scores to .V files or we can parse the log.
    """
    # Look for corresponding .V file (violation file with scores)
    v_file = pdb_file.with_suffix('.V99990001') if '99990001' in pdb_file.name else None
    
    # Try multiple .V file patterns
    base = pdb_file.stem.rsplit('.', 1)[0]  # e.g., 4CXA_fill
    parent = pdb_file.parent
    
    # MODELLER creates files like: 4CXA_fill.B99990001.pdb
    # Model number is in the filename
    match = re.search(r'\.B(\d+)\.pdb$', pdb_file.name)
    if not match:
        return None, None
    
    model_num = match.group(1)
    
    return model_num, None  # We'll get DOPE from log file instead


def parse_dope_from_log(results_dir):
    """Parse DOPE scores from MODELLER log file."""
    scores = {}
    
    # Look for log files
    log_files = list(results_dir.glob("*.log")) + list(results_dir.glob("model_loops.log"))
    
    # Also check parent directory
    if not log_files:
        log_files = list(results_dir.parent.glob("*.log"))
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Parse MODELLER output format:
            # filename                          DOPE score    GA341 score
            # 4CXA_fill.B99990001.pdb          -45678.123      0.987
            for line in content.split('\n'):
                if '.B9999' in line and '.pdb' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        filename = parts[0]
                        try:
                            dope = float(parts[1])
                            scores[filename] = dope
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            continue
    
    return scores


def get_dope_from_pdb_remarks(pdb_file):
    """Try to extract DOPE score from PDB REMARK records."""
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('REMARK') and 'DOPE' in line.upper():
                    # Try to extract score
                    match = re.search(r'[-+]?\d*\.?\d+', line.split('DOPE')[-1])
                    if match:
                        return float(match.group())
    except:
        pass
    return None


def find_best_model(results_dir):
    """Find the model with the lowest DOPE score."""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        return None, None
    
    # Find all model PDB files
    model_files = list(results_dir.glob("*_fill.B*.pdb"))
    
    if not model_files:
        print(f"Error: No model files found in {results_dir}")
        return None, None
    
    print(f"Found {len(model_files)} models in {results_dir}")
    
    # Try to get DOPE scores from log
    scores = parse_dope_from_log(results_dir)
    
    # If no scores from log, try from PDB remarks
    if not scores:
        print("  No scores in log file, checking PDB remarks...")
        for pdb_file in model_files:
            dope = get_dope_from_pdb_remarks(pdb_file)
            if dope is not None:
                scores[pdb_file.name] = dope
    
    # If still no scores, just pick the first model
    if not scores:
        print("  Warning: Could not find DOPE scores, selecting first model")
        best_model = sorted(model_files)[0]
        return best_model, None
    
    # Find best (lowest) DOPE score
    best_file = None
    best_dope = float('inf')
    
    print("\n  Model scores:")
    for filename, dope in sorted(scores.items(), key=lambda x: x[1]):
        print(f"    {filename}: DOPE = {dope:.2f}")
        if dope < best_dope:
            best_dope = dope
            best_file = filename
    
    # Find the actual file path
    for pdb_file in model_files:
        if pdb_file.name == best_file:
            return pdb_file, best_dope
    
    # If exact match not found, try partial match
    for pdb_file in model_files:
        if best_file in pdb_file.name or pdb_file.name in best_file:
            return pdb_file, best_dope
    
    # Fallback: return first model
    print(f"  Warning: Could not match {best_file}, using first model")
    return sorted(model_files)[0], best_dope


def extract_pdb_id(results_dir):
    """Extract PDB ID from directory name like '4CXA_modeller_results'."""
    dirname = Path(results_dir).name
    match = re.match(r'^([A-Za-z0-9]{4})', dirname)
    if match:
        return match.group(1).upper()
    return dirname.split('_')[0].upper()


def process_directory(results_dir, output_dir=None):
    """Process a single modeller results directory."""
    results_dir = Path(results_dir)
    pdb_id = extract_pdb_id(results_dir)
    
    print(f"\n{'='*60}")
    print(f"Processing: {pdb_id}")
    print(f"{'='*60}")
    
    best_model, dope_score = find_best_model(results_dir)
    
    if best_model is None:
        print(f"Error: Could not find best model for {pdb_id}")
        return None
    
    # Determine output path
    if output_dir:
        output_path = Path(output_dir) / f"{pdb_id}_predock.pdb"
    else:
        output_path = Path(f"{pdb_id}_predock.pdb")
    
    # Copy best model to predock file
    shutil.copy(best_model, output_path)
    
    if dope_score is not None:
        print(f"\n  Best model: {best_model.name}")
        print(f"  DOPE score: {dope_score:.2f}")
    else:
        print(f"\n  Selected model: {best_model.name}")
    
    print(f"  Output: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Select best MODELLER model and copy to predock file"
    )
    parser.add_argument(
        "directories",
        nargs="*",
        help="MODELLER results directories (e.g., 4CXA_modeller_results/)"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically find all *_modeller_results directories"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory for predock files (default: current directory)"
    )
    args = parser.parse_args()
    
    # Find directories to process
    if args.auto:
        directories = sorted(Path(".").glob("*_modeller_results"))
        if not directories:
            print("Error: No *_modeller_results directories found")
            return
        print(f"Found {len(directories)} results directories")
    elif args.directories:
        directories = [Path(d) for d in args.directories]
    else:
        parser.print_help()
        print("\nError: Provide directories or use --auto")
        return
    
    # Create output directory if specified
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process each directory
    output_files = []
    for results_dir in directories:
        output_file = process_directory(results_dir, args.output_dir)
        if output_file:
            output_files.append(output_file)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nCreated {len(output_files)} predock files:")
    for f in output_files:
        print(f"  {f}")
    
    print("\nNext step: Run docking with these prepared structures")


if __name__ == "__main__":
    main()
