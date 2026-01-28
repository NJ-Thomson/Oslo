#!/bin/bash
# Master script to run all lambda windows
# Can be run sequentially or submitted to a queue


echo "=========================================="
echo "RBFE Simulation - 14 Lambda Windows"
echo "=========================================="

for i in $(seq 0 13); do
    lambda_dir=$(printf "lambda%02d" $i)
    echo ""
    echo "Running $lambda_dir..."
    cd $lambda_dir
    ./run.sh
    cd ..
done

echo ""
echo "=========================================="
echo "All lambda windows complete!"
echo "=========================================="
echo ""
echo "To analyze results, run:"
echo "  python analyze_fep.py"
