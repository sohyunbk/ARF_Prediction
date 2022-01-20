#!/bin/bash
#SBATCH --job-name=highmemtest        # Job name
#SBATCH --partition=highmem_p         # Partition (queue) name
#SBATCH --ntasks=1                    # Run a single task	
#SBATCH --cpus-per-task=3             # Number of CPU cores per task
#SBATCH --mem=700gb                   # Job memory request
#SBATCH --time=15:00:00               # Time limit hrs:min:sec
#SBATCH --output=./log/5_FS_Up.%j.out   # Standard output log
#SBATCH --error=./log/5_FS_Up.%j.err    # Standard error log

module load Python/3.7.4-GCCcore-8.3.0
module load Anaconda3/2020.02
source activate /home/sb14489/.conda/envs/environmentName

cd /scratch/sb14489/1.ML/4.ChangingSettinginLogisticRegression

python 5_kgrammer_5Fold3Repeat_FS_UpSampling.py Simulated
