#!/bin/bash
#SBATCH --job-name=highmemtest        # Job name
#SBATCH --partition=highmem_p         # Partition (queue) name
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=32             # Number of CPU cores per task
#SBATCH --mem=100gb                   # Job memory request
#SBATCH --time=1:00:00               # Time limit hrs:min:sec
#SBATCH --array=0-13                   # Array range

#SBATCH --output=./log/Statistics_Multi.%j.out   # Standard output log
#SBATCH --error=./log/Statistics_Multi.%j.err    # Standard error log

#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sb14489@uga.edu  # Where to send mail	

cd  /scratch/sb14489/1.ML_ARF/2.ML/1-3.MakeFASTA_Soybean_Redo

module load Anaconda3/2020.02
source activate /home/sb14489/.conda/envs/environmentName

List=(18 25 27 29 34 35 36 39 10 13 14 16 4 7)

#python 2-1_CutAll_AllARFs_Ratio_Statistics.py
python 2-1_CutAll_AllARFs_Ratio_Statistics.py "${List[SLURM_ARRAY_TASK_ID]}"
