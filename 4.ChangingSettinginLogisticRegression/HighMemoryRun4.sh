#!/bin/bash
#SBATCH --job-name=4_DownS        # Job name
#SBATCH --partition=highmem_p         # Partition (queue) name
#SBATCH --ntasks=1                    # Run a single task	
#SBATCH --cpus-per-task=3             # Number of CPU cores per task
#SBATCH --mem=600gb                   # Job memory request
#SBATCH --time=15:00:00               # Time limit hrs:min:sec
#SBATCH --output=./log/4_FS_Down.%j.out   # Standard output log
#SBATCH --error=./log/4_FS_Down.%j.err    # Standard error log

module load Python/3.7.4-GCCcore-8.3.0
module load Anaconda3/2020.02
source activate /home/sb14489/.conda/envs/environmentName

cd /scratch/sb14489/1.ML/4.ChangingSettinginLogisticRegression

python 4_kgrammer_5Fold3Repeat_FS_DownSampling.py Simulated
