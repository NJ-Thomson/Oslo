#!/usr/bin/env bash
# GPU job continuation template for Gromacs with proper thread-MPI settings

export OMP_NUM_THREADS="${1}"
STEPNAME="${2}"
TOPOLOGY="${3}"
STRUCTURE="${4}"
CPT="${5}"
#opts
GROMACS_TPR="${STEPNAME}.tpr"
CONFOUT="${STEPNAME}.gro"
#Grompp this:
gmx grompp -f ./${STEPNAME}.mdp -c ${STRUCTURE} -r ${STRUCTURE} -p ${TOPOLOGY} -t ${CPT} -o ${GROMACS_TPR} -maxwarn 3
# run gromacs command with GPU and thread-MPI settings
gmx mdrun  -s $GROMACS_TPR -c $CONFOUT -deffnm ${STEPNAME}
