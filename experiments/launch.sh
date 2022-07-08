#!/bin/bash
module purge
module load cuda/10.1
{
	source activate /jmain01/home/JAD035/pkm01/aap21-pkm01/anaconda3/envs/gnn
	wait $!
	printf “sucess”
}||{
	printf “failed”
}
export VISION_DATA="/jmain01/home/JAD035/pkm01/shared/datasets"
export GRAPH_DATA="/jmain01/home/JAD035/pkm01/shared/datasets"
nvidia-smi
nvcc --version
echo "Using CUDA device" $CUDA_VISIBLE_DEVICES
#python reproduce/jade_cifar.py
python reproduce/cifar100.py
python reproduce/tiny.py
python reproduce/cifar10.py
python reproduce/svhn.py
