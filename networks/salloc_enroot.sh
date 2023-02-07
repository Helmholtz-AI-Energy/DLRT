#!/bin/bash

SCRIPT_DIR="/hkfs/work/workspace/scratch/qv2382-dlrt/"
SINGULARITY_FILE="${SCRIPT_DIR}containers/torch-image.sif"

export DATA_PREFIX=$(sed '4q;d' ../configs/resnet18.yaml | cut -d '"' -f2)

export CUDA_VISIBLE_DEVICES="0,1,2,3"

salloc --partition=accelerated -A haicore-project-scc -N 1 --time 12:00:00 --gres gpu:1 \
  --container-name=torch \
  --container-mounts=/etc/slurm/task_prolog.hk:/etc/slurm/task_prolog.hk,"${DATA_PREFIX}","${SCRIPT_DIR}","/scratch","/tmp" \
  --container-mount-home \
  --container-writable

#export UCX_MEMTYPE_CACHE=0
#export NCCL_IB_TIMEOUT=100
#export SHARP_COLL_LOG_LEVEL=3
#export OMPI_MCA_coll_hcoll_enable=0
#export NCCL_SOCKET_IFNAME="ib0"
#export NCCL_COLLNET_ENABLE=0
