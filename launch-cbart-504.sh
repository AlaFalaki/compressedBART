#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gres=gpu:v100l:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5    # There are 24 CPU cores on Cedar GPU nodes
#SBATCH --mem=50GB               # Request the full memory of the node
#SBATCH --time=0-03:00
#SBATCH --account=def-rgras
#SBATCH --mail-user=alamfal@uwindsor.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="inf-bart-ae-504"
#SBATCH --output=%x-%j.out

module load gcc/9.3.0 arrow/5.0.0 python/3.8 cuda/11.4 scipy-stack/2022a

source $HOME/lt/bin/activate

python main.py \
        --checkpoint_dir ./cbart-checkpoints/504 \
        --exp_name 504 \
        --batch_size 32 \
        --first 640 \
        --second 576 \
        --third 504
#        --test
