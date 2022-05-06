#!/bin/bash
#OAR -l gpu_device=1,core=12,walltime=8:00:00


. /etc/profile.d/modules.sh

set -xv

echo "setting up environment"
source ~/.bashrc
stat_env

echo "environment all set up"
nvidia-smi

TMPDIR=$SCRATCHDIR/$OAR_JOB_ID
mkdir -p $TMPDIR
cd $TMPDIR

EXECUTABLE=/srv/tempdd/ktit/stat_reliability_measure/tests/linear_gaussian_tests/linear_gaussian_test_smc_pyt.py
LOGDIR=/srv/tempdd/ktit/stat_reliability_measure/logs/linear_gaussian_tests/
echo "=============== RUN ${OAR_JOB_ID}  ==============="
echo "Running ..."
python -u ${EXECUTABLE} --N_range 32,64,128,256,512 --T_range 2,5,10,20 --L_range 1,5,10 --p_range 1e-6 --update_agg_res True
echo "Done"
echo "==================================="
