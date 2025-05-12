#!/bin/bash
#SBATCH --job-name=vtnet_training
#SBATCH --output=results/vtnet_%j.out
#SBATCH --error=results/vtnet_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ubcml
#SBATCH --account=ubcml
#SBATCH --mail-user=chhavi.nayyar@ubc.ca
#SBATCH --mail-type=BEGIN,END,FAIL

module load cuda/11.6
source ~/miniconda3/bin/activate vtenv

# ensure results directory exists
mkdir -p /ubc/cs/home/c/cnayyar/hai_work/results

# run your training script
python /ubc/cs/home/c/cnayyar/hai_work/train_vtnet.py \
  --data-dir /ubc/cs/home/c/cnayyar/hai_work/output \
  --save-dir /ubc/cs/home/c/cnayyar/hai_work/trained
