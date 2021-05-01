#!/bin/tcsh
#SBATCH --partition=hawkgpu
#SBATCH --account=cse498_s2021
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --job-name fp-natprune16to1
#SBATCH --output="fp-nat16prune16to1.out"


cd /home/yud319/cse498_s2021_053121/yud319/ReproducePaper/adversarial_robustness_model_compression/cifar10-test/cifar10/
/home/yud319/anaconda3/bin/python main.py --config_file ./prune/config_w16to1.yaml --stage admm
/home/yud319/anaconda3/bin/python main.py --config_file ./prune/config_w16to1.yaml --stage retrain


# squeue -u yud319
# sbatch admm-prune16to1.slurm