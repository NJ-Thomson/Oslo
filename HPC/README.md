# GROMACS MD on LUMI

Scripts for running long GROMACS molecular dynamics simulations on LUMI-G with automatic checkpoint restart.

## Overview

Running MD simulations that exceed the walltime limit requires manual resubmission from checkpoints. These scripts automate this process:

- **runmd.sh** - SLURM submission script that runs GROMACS on compute nodes
- **monitor_md.sh** - Monitoring script that runs on the login node, tracks progress, and resubmits jobs when needed

## Setup

1. Copy both scripts to your simulation directory
2. Edit `runmd.sh`:
   - Set your project account (`--account=project_XXXXXXX`)
   - Adjust GROMACS module if needed
   - Update input file names to match your system
3. Make scripts executable:
   ```bash
   chmod +x runmd.sh monitor_md.sh
   ```

## Required Files

Your simulation directory should contain:
- `step7_production.mdp` - MD parameters (must contain `dt` and `nsteps`)
- `step7_10.gro` - Starting coordinates
- `topol.top` - Topology file
- `index.ndx` - Index file

## Usage

Start the monitor on the login node:
```bash
nohup ./monitor_md.sh -J myjobname &> monitor.log &
```

The monitor will:
1. Calculate target simulation time from `dt * nsteps` in the MDP file
2. Submit the job via `sbatch`
3. Check progress every 15 minutes by reading the log file
4. Resubmit from checkpoint when walltime is reached
5. Exit when simulation is complete

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `-J` | SLURM job name | `gromacs_md` |

## How It Works

1. `runmd.sh` creates a TPR file if one doesn't exist, then runs `gmx mdrun` with `-maxh 23.5` to ensure clean checkpoint writing before the 24h walltime
2. `monitor_md.sh` parses the GROMACS log to find the current simulation time and compares it to the target time
3. When a job ends (walltime reached), the monitor detects this via `squeue` and resubmits
4. On restart, `runmd.sh` detects the checkpoint file and uses `-cpi` to continue

## Customisation

Edit the variables at the top of each script:

**monitor_md.sh:**
```bash
MDP_FILE="step7_production.mdp"
LOG_FILE="md.log"
SLURM_SCRIPT="runmd.sh"
CHECK_INTERVAL=900  # seconds
```

**runmd.sh:**
```bash
TPR="step7_production.tpr"
MDP="step7_production.mdp"
GRO="step7_10.gro"
CPT="step7_10.cpt"
TOP="topol.top"
NDX="index.ndx"
```

## Notes

- The monitor runs indefinitely until the simulation completes - use `nohup` or `screen`/`tmux`
- Check `monitor.log` for progress updates
- The job name passed with `-J` must match between monitor and any manually submitted jobs