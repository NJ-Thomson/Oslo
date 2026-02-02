#!/usr/bin/env python3
"""
05_run_simulations.py - Run or Submit QM/MM Simulations

Manages execution of umbrella sampling windows:
- Local execution (sequential or parallel)
- SLURM job submission
- Progress monitoring

Usage:
    python 05_run_simulations.py --config config.yaml [--submit slurm|local]

Options:
    --submit slurm   Submit to SLURM scheduler
    --submit local   Run locally (sequential)
    --parallel N     Run N windows in parallel (local mode)
    --dry-run        Show what would be submitted without running
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def check_window_complete(window_dir: Path) -> bool:
    """Check if a window simulation has completed."""
    prod_gro = window_dir / 'prod.gro'
    prod_log = window_dir / 'prod.log'
    
    if not prod_gro.exists():
        return False
    
    # Check log for successful completion
    if prod_log.exists():
        with open(prod_log) as f:
            content = f.read()
            if 'Finished mdrun' in content or 'finished' in content.lower():
                return True
    
    return False


def run_window_local(window_dir: Path) -> Dict:
    """Run a single window locally."""
    script = window_dir / 'run_mimic.sh'
    
    if not script.exists():
        return {
            'window': window_dir.name,
            'success': False,
            'error': 'run_mimic.sh not found'
        }
    
    try:
        result = subprocess.run(
            ['bash', str(script)],
            cwd=window_dir,
            capture_output=True,
            text=True,
            timeout=86400  # 24 hour timeout
        )
        
        return {
            'window': window_dir.name,
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'stdout': result.stdout[-1000:] if result.stdout else '',
            'stderr': result.stderr[-1000:] if result.stderr else '',
        }
    except subprocess.TimeoutExpired:
        return {
            'window': window_dir.name,
            'success': False,
            'error': 'Timeout (24h)'
        }
    except Exception as e:
        return {
            'window': window_dir.name,
            'success': False,
            'error': str(e)
        }


def submit_window_slurm(window_dir: Path, dry_run: bool = False) -> Dict:
    """Submit a window to SLURM."""
    script = window_dir / 'submit_slurm.sh'
    
    if not script.exists():
        return {
            'window': window_dir.name,
            'success': False,
            'error': 'submit_slurm.sh not found'
        }
    
    if dry_run:
        return {
            'window': window_dir.name,
            'success': True,
            'job_id': 'DRY_RUN',
            'dry_run': True
        }
    
    try:
        result = subprocess.run(
            ['sbatch', str(script)],
            cwd=window_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # Parse job ID from "Submitted batch job XXXXX"
            job_id = result.stdout.strip().split()[-1]
            return {
                'window': window_dir.name,
                'success': True,
                'job_id': job_id
            }
        else:
            return {
                'window': window_dir.name,
                'success': False,
                'error': result.stderr
            }
    except Exception as e:
        return {
            'window': window_dir.name,
            'success': False,
            'error': str(e)
        }


def run_system_local(
    system_dir: Path,
    parallel: int = 1,
    skip_completed: bool = True
) -> Dict:
    """Run all windows for a system locally."""
    
    windows_dir = system_dir / 'windows'
    window_dirs = sorted(windows_dir.glob('window_*'))
    
    # Filter out completed windows
    if skip_completed:
        window_dirs = [w for w in window_dirs if not check_window_complete(w)]
    
    if not window_dirs:
        return {'total': 0, 'success': 0, 'skipped': True}
    
    results = []
    
    if parallel <= 1:
        # Sequential execution
        for window_dir in window_dirs:
            print(f"      Running {window_dir.name}...")
            result = run_window_local(window_dir)
            results.append(result)
            if result['success']:
                print(f"        ✓ Complete")
            else:
                print(f"        ✗ Failed: {result.get('error', 'Unknown')}")
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(run_window_local, w): w.name 
                for w in window_dirs
            }
            
            for future in as_completed(futures):
                window_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = "✓" if result['success'] else "✗"
                    print(f"      {status} {window_name}")
                except Exception as e:
                    results.append({
                        'window': window_name,
                        'success': False,
                        'error': str(e)
                    })
    
    success_count = sum(1 for r in results if r['success'])
    
    return {
        'total': len(results),
        'success': success_count,
        'failed': len(results) - success_count,
        'results': results
    }


def submit_system_slurm(
    system_dir: Path,
    skip_completed: bool = True,
    dry_run: bool = False
) -> Dict:
    """Submit all windows for a system to SLURM."""
    
    windows_dir = system_dir / 'windows'
    window_dirs = sorted(windows_dir.glob('window_*'))
    
    # Filter out completed windows
    if skip_completed:
        window_dirs = [w for w in window_dirs if not check_window_complete(w)]
    
    if not window_dirs:
        return {'total': 0, 'submitted': 0, 'skipped': True}
    
    results = []
    job_ids = []
    
    for window_dir in window_dirs:
        result = submit_window_slurm(window_dir, dry_run=dry_run)
        results.append(result)
        
        if result['success']:
            job_ids.append(result.get('job_id', 'N/A'))
            print(f"      ✓ {window_dir.name} -> Job {result.get('job_id', 'N/A')}")
        else:
            print(f"      ✗ {window_dir.name}: {result.get('error', 'Unknown')}")
    
    submitted = sum(1 for r in results if r['success'])
    
    return {
        'total': len(results),
        'submitted': submitted,
        'failed': len(results) - submitted,
        'job_ids': job_ids,
        'results': results
    }


def process_system(
    system_dir: Path,
    config: dict,
    submit_mode: str,
    parallel: int,
    dry_run: bool
) -> bool:
    """Process all windows for a system."""
    
    with open(system_dir / 'system_info.json') as f:
        system_info = json.load(f)
    
    print(f"  Processing: {system_info['receptor']}/{system_info['inhibitor']}/{system_info['warhead']}")
    
    windows_dir = system_dir / 'windows'
    if not windows_dir.exists():
        print("    ERROR: No windows directory")
        return False
    
    n_windows = len(list(windows_dir.glob('window_*')))
    print(f"    {n_windows} windows")
    
    if submit_mode == 'slurm':
        result = submit_system_slurm(system_dir, dry_run=dry_run)
        if result.get('skipped'):
            print(f"    All windows already complete")
        else:
            print(f"    Submitted {result['submitted']}/{result['total']} jobs")
    else:
        result = run_system_local(system_dir, parallel=parallel)
        if result.get('skipped'):
            print(f"    All windows already complete")
        else:
            print(f"    Completed {result['success']}/{result['total']} windows")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Run QM/MM simulations")
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'config.yaml')
    parser.add_argument('--submit', choices=['slurm', 'local'], default='local',
                       help='Submission mode')
    parser.add_argument('--parallel', type=int, default=1,
                       help='Number of parallel windows (local mode)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done')
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
    print("Step 5: Run QM/MM Simulations")
    print("=" * 60)
    print(f"Mode: {args.submit}")
    if args.submit == 'local':
        print(f"Parallel: {args.parallel}")
    if args.dry_run:
        print("DRY RUN - no jobs will be submitted")
    print()
    
    # Get system lists
    receptors = args.receptors or list(config['systems']['receptors'].keys())
    inhibitors = args.inhibitors or config['systems']['inhibitors']
    warheads = args.warheads or list(config['systems']['warheads'].keys())
    
    for receptor in receptors:
        for inhibitor in inhibitors:
            for warhead in warheads:
                system_dir = output_dir / receptor / inhibitor / warhead
                if system_dir.exists() and (system_dir / 'windows').exists():
                    process_system(
                        system_dir, config,
                        args.submit, args.parallel, args.dry_run
                    )
    
    print()
    print("Done!")


if __name__ == '__main__':
    main()
