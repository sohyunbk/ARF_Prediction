#!/bin/bash
#SBATCH --job-name=highmemtest        # Job name
#SBATCH --partition=highmem_p         # Partition (queue) name
#SBATCH --ntasks=1                    # Run a single task	
#SBATCH --cpus-per-task=20             # Number of CPU cores per task
#SBATCH --mem=600gb                   # Job memory request
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --output=./Result_211225/ValidationClassweightNone_lowMemory.%j.out   # Standard output log
#SBATCH --error=./Result_211225/ValidationClassweightNone_lowMemory.%j.err    # Standard error log
#SBATCH --array=0-11                  # Array range


List=(4 16 18 27 29 34 7 10 14 25 36 39)

module load Python/3.7.4-GCCcore-8.3.0

cd /scratch/sb14489/1.ML_ARF/2.ML/5-3.All_ARF_MaizetoSoybean
module load Anaconda3/2020.02

#conda init --all
#conda activate environmentName
source activate /home/sb14489/.conda/envs/environmentName
#conda info --envs
#conda update keras

python kgrammer_FeatureSelection_Upsampling_AlltheARFs.py "${List[SLURM_ARRAY_TASK_ID]}"
