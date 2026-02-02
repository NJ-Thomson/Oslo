#!/usr/bin/env python3
"""
QM/MM Workflow for Covalent ABFE Calculations

Complete pipeline from equilibrated covalent structures to free energy profiles.

Usage:
    python run_qmmm_workflow.py [options]

Options:
    --config PATH       Configuration file (default: config.yaml)
    --receptors LIST    Receptor IDs to process (default: all from config)
    --inhibitors LIST   Inhibitor names (default: all from config)
    --warheads LIST     Warheads to include (default: all)
    --steps LIST        Pipeline steps to run (default: all)
    --dry-run           Print what would be done without executing

Examples:
    # Run full workflow
    python run_qmmm_workflow.py

    # Process specific system
    python run_qmmm_workflow.py --receptors 5ACB --inhibitors Inhib_32

    # Run only setup (no simulations)
    python run_qmmm_workflow.py --steps prepare,define_qm,windows,setup_mimic
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_paths(config: dict, base_dir: Path) -> dict:
    """Resolve relative paths in configuration."""
    paths = config.get('paths', {})
    for key in ['input_dir', 'output_dir']:
        if key in paths:
            p = Path(paths[key])
            if not p.is_absolute():
                paths[key] = str((base_dir / p).resolve())
    return config


# ============================================================================
# Workflow Steps
# ============================================================================

PIPELINE_STEPS = [
    ('prepare', '01_prepare_systems.py', 'Prepare system snapshots'),
    ('define_qm', '02_define_qm_region.py', 'Define QM region atoms'),
    ('windows', '03_generate_windows.py', 'Generate umbrella windows'),
    ('setup_mimic', '04_setup_mimic.py', 'Create GROMACS/ORCA/MiMiC input files'),
    ('run', '05_run_simulations.py', 'Run QM/MM simulations'),
    ('analyze', '06_analyze_pmf.py', 'Calculate PMF and free energies'),
]


def run_step(
    script_name: str,
    script_dir: Path,
    config_path: Path,
    receptors: List[str],
    inhibitors: List[str],
    warheads: List[str],
    dry_run: bool = False,
    extra_args: Optional[List[str]] = None
) -> bool:
    """Run a pipeline step script."""
    script_path = script_dir / script_name
    
    if not script_path.exists():
        log.error(f"Script not found: {script_path}")
        return False
    
    cmd = [
        sys.executable, str(script_path),
        '--config', str(config_path),
        '--receptors', *receptors,
        '--inhibitors', *inhibitors,
        '--warheads', *warheads,
    ]
    
    if extra_args:
        cmd.extend(extra_args)
    
    if dry_run:
        log.info(f"DRY RUN: {' '.join(cmd)}")
        return True
    
    log.info(f"Running: {script_name}")
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        log.error(f"Step failed with return code {e.returncode}")
        return False
    except Exception as e:
        log.error(f"Step failed: {e}")
        return False


# ============================================================================
# System Discovery
# ============================================================================

def discover_systems(input_dir: Path, config: dict) -> List[Dict]:
    """Find all equilibrated systems available for QM/MM."""
    systems = []
    
    receptor_config = config['systems']['receptors']
    warhead_config = config['systems']['warheads']
    
    for receptor in input_dir.iterdir():
        if not receptor.is_dir() or receptor.name.startswith('.'):
            continue
        if receptor.name not in receptor_config:
            continue
            
        for inhibitor in receptor.iterdir():
            if not inhibitor.is_dir() or not inhibitor.name.startswith('Inhib_'):
                continue
                
            for warhead in inhibitor.iterdir():
                if not warhead.is_dir() or warhead.name not in warhead_config:
                    continue
                
                # Check for required files
                required = ['npt.gro', 'topol.top']
                if all((warhead / f).exists() for f in required):
                    systems.append({
                        'receptor': receptor.name,
                        'inhibitor': inhibitor.name,
                        'warhead': warhead.name,
                        'path': warhead,
                        'cys_resid': receptor_config[receptor.name]['cys_resid'],
                    })
    
    return systems


def filter_systems(
    systems: List[Dict],
    receptors: Optional[List[str]],
    inhibitors: Optional[List[str]],
    warheads: Optional[List[str]]
) -> List[Dict]:
    """Filter systems based on command line arguments."""
    filtered = systems
    
    if receptors:
        filtered = [s for s in filtered if s['receptor'] in receptors]
    if inhibitors:
        filtered = [s for s in filtered if s['inhibitor'] in inhibitors]
    if warheads:
        filtered = [s for s in filtered if s['warhead'] in warheads]
    
    return filtered


# ============================================================================
# Main Workflow
# ============================================================================

def run_workflow(
    config_path: Path,
    receptors: Optional[List[str]] = None,
    inhibitors: Optional[List[str]] = None,
    warheads: Optional[List[str]] = None,
    steps: Optional[List[str]] = None,
    dry_run: bool = False
):
    """Run the complete QM/MM workflow."""
    
    # Load configuration
    config = load_config(config_path)
    base_dir = config_path.parent
    config = resolve_paths(config, base_dir)
    
    input_dir = Path(config['paths']['input_dir'])
    output_dir = Path(config['paths']['output_dir'])
    
    log.info("=" * 60)
    log.info("QM/MM Workflow for Covalent ABFE")
    log.info("=" * 60)
    log.info(f"Config: {config_path}")
    log.info(f"Input:  {input_dir}")
    log.info(f"Output: {output_dir}")
    
    # Discover available systems
    all_systems = discover_systems(input_dir, config)
    log.info(f"Found {len(all_systems)} equilibrated systems")
    
    # Apply filters
    systems = filter_systems(all_systems, receptors, inhibitors, warheads)
    log.info(f"Processing {len(systems)} systems after filtering")
    
    if not systems:
        log.warning("No systems to process!")
        return
    
    # Show systems
    for s in systems:
        log.info(f"  • {s['receptor']}/{s['inhibitor']}/{s['warhead']}")
    
    # Create output directory
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save manifest
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'config': str(config_path),
        'systems': systems,
    }
    
    if not dry_run:
        with open(output_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
    
    # Determine which steps to run
    if steps:
        selected_steps = [(s, script, desc) for s, script, desc in PIPELINE_STEPS 
                         if s in steps]
    else:
        selected_steps = PIPELINE_STEPS
    
    # Extract unique values for script arguments
    rec_list = list(set(s['receptor'] for s in systems))
    inh_list = list(set(s['inhibitor'] for s in systems))
    war_list = list(set(s['warhead'] for s in systems))
    
    # Run pipeline
    log.info("-" * 60)
    for step_name, script_name, description in selected_steps:
        log.info(f"STEP: {description}")
        
        success = run_step(
            script_name=script_name,
            script_dir=base_dir,
            config_path=config_path,
            receptors=rec_list,
            inhibitors=inh_list,
            warheads=war_list,
            dry_run=dry_run
        )
        
        if not success and not dry_run:
            log.error(f"Workflow failed at step: {step_name}")
            return False
        
        log.info("")
    
    log.info("=" * 60)
    log.info("Workflow completed successfully!")
    log.info("=" * 60)
    return True


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="QM/MM Workflow for Covalent ABFE Calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', type=Path, default=Path(__file__).parent / 'config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--receptors', nargs='+', default=None,
        help='Receptor IDs to process (e.g., 5ACB 7NXJ)'
    )
    parser.add_argument(
        '--inhibitors', nargs='+', default=None,
        help='Inhibitor names (e.g., Inhib_32 Inhib_36)'
    )
    parser.add_argument(
        '--warheads', nargs='+', default=None,
        help='Warheads to include (e.g., acry chlo)'
    )
    parser.add_argument(
        '--steps', nargs='+', default=None,
        choices=[s[0] for s in PIPELINE_STEPS],
        help='Pipeline steps to run'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print what would be done without executing'
    )
    
    args = parser.parse_args()
    
    success = run_workflow(
        config_path=args.config,
        receptors=args.receptors,
        inhibitors=args.inhibitors,
        warheads=args.warheads,
        steps=args.steps,
        dry_run=args.dry_run
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
