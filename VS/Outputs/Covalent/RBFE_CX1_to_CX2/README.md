# RBFE Setup for Covalent Ligand Transformation (RTP Mode)

## Overview
- State A (λ=0): Outputs/Covalent/Inhib_32_acry/md_simulation [CX1]
- State B (λ=1): Outputs/Covalent/Inhib_32_chlo/md_simulation [CX2]
- Lambda windows: 14

## TEST MODE (Current Settings)
**WARNING: These settings are for testing only - not for production!**
- EM: 500 steps
- NVT: 500 steps (1 ps)
- NPT: 500 steps (1 ps)
- Production: 2500 steps (5 ps)
- Total per window: ~7 ps
- Total simulation time: ~100 ps (all 14 windows)

## Production Settings (for real calculations)
- EM: 5000 steps
- NVT: 50,000 steps (100 ps)
- NPT: 250,000 steps (500 ps)
- Production: 1,500,000 steps (3 ns)
- Total simulation time: 42 ns

## Source
Ligand topologies extracted from force field RTP files:
- State A: Outputs/Covalent/Inhib_32_acry/md_simulation/amber99sb-ildn-cx1.ff/aminoacids.rtp [CX1]
- State B: Outputs/Covalent/Inhib_32_chlo/md_simulation/amber99sb-ildn-cx2.ff/aminoacids.rtp [CX2]

## Directory Structure
- `pmx_input/`: pmx intermediate files (atom mapping, hybrid topology)
- `input/`: Input files for simulations (structure, topology, force field)
- `lambda00/` to `lambda13/`: Individual lambda window directories

## Running Simulations

### Run all windows sequentially:
```bash
./run_all.sh
```

### Run a single window:
```bash
cd lambda00
./run.sh
```

### Submit to SLURM (example):
```bash
for i in $(seq 0 13); do
    sbatch --job-name=fep_$i --wrap="cd lambda$(printf '%02d' $i) && ./run.sh"
done
```

## Analysis
After all windows complete:
```bash
python analyze_fep.py
```

Or use GROMACS BAR directly:
```bash
gmx bar -f lambda*/prod.xvg -o bar.xvg
```

## Lambda Schedule
14 windows with λ = 0.000, 0.077, 0.154, 0.231, 0.308, 0.385, 0.462, 0.538, 0.615, 0.692, 0.769, 0.846, 0.923, 1.000
