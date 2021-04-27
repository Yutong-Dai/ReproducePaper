#!/bin/tcsh
#SBATCH --partition=hawkgpu
#SBATCH --account=cse498_s2021
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --job-name fp-nat1
#SBATCH --output="fp-nat1.out"

cd /home/yud319/cse498_s2021_053121/yud319/ReproducePaper/adversarial_robustness_model_compression/cifar10-test/cifar10/
/home/yud319/anaconda3/bin/python main.py --config_file ./natural/config_w1.yaml --stage pretrain


# squeue -u yud319
# sbatch natrun1.slurm