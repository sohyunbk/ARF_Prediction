#!/bin/bash
#SBATCH --job-name=highmemtest        # Job name
#SBATCH --partition=highmem_p         # Partition (queue) name
#SBATCH --ntasks=1                    # Run a single task	
#SBATCH --cpus-per-task=10             # Number of CPU cores per task
#SBATCH --mem=300gb                   # Job memory request
#SBATCH --time=45:00:00               # Time limit hrs:min:sec
#SBATCH --output=Validation.%j.out   # Standard output log
#SBATCH --error=Validation.%j.err    # Standard error log
#SBATCH --array=0-7                   # Array range


List=(18 25 27 29 34 35 36 39)

module load Python/3.7.4-GCCcore-8.3.0

cd /scratch/sb14489/1-2.ML_NewTry/3.ML_FeatureSelection
module load Anaconda3/2020.02
#conda init --all
#conda activate environmentName
#source activate /home/sb14489/.conda/envs/environmentName
#conda info --envs
#conda update keras

python kgrammer_FeatureSelection_Upsampling_AlltheARFs.py "${List[SLURM_ARRAY_TASK_ID]}"
