#!/bin/bash

proj_dir=/mnt/lustre/working/lab_lucac/sebastiN/projects/OCD_modeling/
redis_ip=`awk '{print $1}' ${proj_dir}traces/.redis_ip`

python $proj_dir/code/OCD_modeling/mcmc/abc_hpc.py

abc-redis-manager stop --port 6379 --host ${redis_ip} --password bayesopt1234321
