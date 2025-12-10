#!/bin/bash
#
# Simple MD monitor - runs on login node, checks every 15 min, resubmits if needed
#
# Usage: nohup ./monitor_md.sh &> monitor.log &
#

MDP_FILE="step8_production.mdp"
LOG_FILE="step8_production.log"
SLURM_SCRIPT="runmd.sh"
CHECK_INTERVAL=900  # 15 minutes
JOB_NAME="gmx_md"

while getopts "J:" opt; do
    case $opt in
        J) JOB_NAME="$OPTARG" ;;
    esac
done



# Get target time from mdp (dt * nsteps)
dt=$(grep -E "^\s*dt\s*=" "$MDP_FILE" | sed 's/.*=\s*//' | sed 's/;.*//' | tr -d ' ')
nsteps=$(grep -E "^\s*nsteps\s*=" "$MDP_FILE" | sed 's/.*=\s*//' | sed 's/;.*//' | tr -d ' ')
target_time=$(echo "$dt * $nsteps" | bc -l)

echo "Target time: ${target_time} ps"

while true; do
    # Get current time from log (last occurrence)
    if [[ -f "$LOG_FILE" ]]; then
        current_time=$(awk '/Step.*Time/{getline; time=$2} END{print time}' "$LOG_FILE")
    else
        current_time=0
    fi
    current_time=${current_time:-0}
    
    echo "[$(date)] Current: ${current_time} / ${target_time} ps"
    
    # Check if finished
    if (( $(echo "$current_time >= $target_time" | bc -l) )); then
        echo "Simulation complete!"
        exit 0
    fi
    
    # Check if job is running
    if ! squeue -u "$USER" -n "$JOB_NAME" -h | grep -q .; then
        echo "Job not running, submitting..."
	echo "Job name: $JOB_NAME"
        sbatch -J "$JOB_NAME" "$SLURM_SCRIPT" 
    fi
    
    sleep $CHECK_INTERVAL
done
