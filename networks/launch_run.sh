#!/usr/bin/env bash

# Slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
### #SBATCH --gpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --job-name=dlrt-ddp
#SBATCH --partition=dev_accelerated
#SBATCH --account=haicore-project-scc
#SBATCH --gres=gpu:1
#SBATCH --output="/hkfs/work/workspace/scratch/qv2382-dlrt/DLRT/logs/slurm-%j"

ml purge

# pmi2 cray_shasta
SRUN_PARAMS=(
  --mpi="pmi2"
  #--ntasks-per-node=1
  --gpus-per-task=1
  --cpus-per-task="19"
  #--cpu-bind="ldoms"
  --gpu-bind="closest"
  --label
)

#export CUDA_AVAILABLE_DEVICES="0,1,2,3"
# /hkfs/work/workspace/scratch/qv2382-dpnn-scratch/dpnn-scratch
SCRIPT_DIR="/hkfs/work/workspace/scratch/qv2382-dlrt/"
SINGULARITY_FILE="${SCRIPT_DIR}containers/torch-image.sif"
#mlperf-torch.sif"
#SINGULARITY_FILE="/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/docker/nvidia-optimized-torch.sif"

export UCX_MEMTYPE_CACHE=0
export NCCL_IB_TIMEOUT=100
export SHARP_COLL_LOG_LEVEL=3
export OMPI_MCA_coll_hcoll_enable=0
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_COLLNET_ENABLE=0

#export DATA_PREFIX="/hkfs/home/dataset/datasets/imagenet-2012/original/imagenet-raw/ILSVRC/Data/CLS-LOC/"
#export TRAIN_FILE="/hkfs/work/workspace/scratch/qv2382-dpnn_scratch/dpnn-scratch/hdfml_train_cifar.sh"

if [ "$DATASET" == "imagenet" ]; then
	export DATA_PREFIX="/hkfs/home/dataset/datasets/imagenet-2012/original/imagenet-raw/ILSVRC/Data/CLS-LOC/"
elif [ "$DATASET" == "cifar10" ]; then
	export DATA_PREFIX="/hkfs/home/dataset/datasets/CIFAR10/"
elif [ "$DATASET" == "cifar100" ]; then
	export DATA_PREFIX="/hkfs/home/dataset/datasets/CIFAR100/"
else
	echo "Defaulting to CIFAR10 training"
	export DATASET="cifar10"
	export DATA_PREFIX="/hkfs/home/dataset/datasets/CIFAR10/"
	#export DATA_PREFIX="/hkfs/home/dataset/datasets/imagenet-2012/original/imagenet-raw/ILSVRC/Data/CLS-LOC/"
fi


#export NN_ARCH="resnet18"

export CUDA_VISIBLE_DEVICES="0,1,2,3"
echo "Loading data from ${DATA_PREFIX}"
#echo "${SCRIPT_DIR}"
pwd
echo "config: ${CONFIG}"
#echo "srun ${SRUN_PARAMS[@]} singularity exec --nv \
#  --bind ${DATA_PREFIX},${SCRIPT_DIR},/scratch,/tmp ${SINGULARITY_FILE} \
#    bash -c python resnet.py --data=${DATA_PREFIX} --world-size=${SLURM_NTASKS}"

srun "${SRUN_PARAMS[@]}" singularity exec --nv \
  --bind "${DATA_PREFIX}","${SCRIPT_DIR}","/scratch","/tmp","/hkfs/work/workspace/scratch/qv2382-dlrt/DLRT/dlrt/":"/opt/conda/lib/python3.8/site-packages/dlrt/" "${SINGULARITY_FILE}" \
    bash -c "python -u resnet.py --data=${DATA_PREFIX} -b 128 -p 10 --lr 0.1 --momentum 0.1 --wd 0.00001 -a resnet18"
