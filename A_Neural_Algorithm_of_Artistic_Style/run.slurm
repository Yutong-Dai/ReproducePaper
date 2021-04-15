#!/bin/tcsh
#SBATCH --partition=hawkgpu
#SBATCH —account=cse498_s2021
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --job-name styleTrans
#SBATCH --output="styleTrans.%j.%N.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<username>@lehigh.edu

cd ${SLURM_SUBMIT_DIR}
/home/yud319/anaconda3/bin/python styleTrans.py --c './db/girl.jpg' --s './db/starry.jpg' --savename 'girl-starry' --epochs 20000 
# /home/yud319/anaconda3/bin/python styleTrans.py --c './db/pikachu.jpg' --s './db/starry.jpg' --savename 'pikachu-starry' --epochs 20000 

# squeue -u yud319
# sbatch run.slurm