#!/bin/bash
#SBATCH --job-name=highmemtest        # Job name
#SBATCH --partition=highmem_p         # Partition (queue) name
#SBATCH --ntasks=1                    # Run a single task	
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --mem=650gb                   # Job memory request
#SBATCH --time=45:00:00               # Time limit hrs:min:sec
#SBATCH --output=./log/3Try_CorrectedScript.%j.out   # Standard output log
#SBATCH --error=./log/3Try_CorrectedScript.%j.err    # Standard error log
#SBATCH --array=0-11                   # Array range


List=(10 14 16 4 7 18 25 27 29 34 36 39)

module load Python/3.7.4-GCCcore-8.3.0

cd /scratch/sb14489/1.ML/5-1.All_ARF_MaizetoMaize
module load Anaconda3/2020.02

source activate /home/sb14489/.conda/envs/environmentName

python kgrammer_FeatureSelection_Upsampling_AlltheARFs_Final_5Fold3Repeat.py "${List[SLURM_ARRAY_TASK_ID]}"
