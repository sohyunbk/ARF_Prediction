#!/bin/bash
#SBATCH --job-name=highmemtest        # Job name
#SBATCH --partition=highmem_p         # Partition (queue) name
#SBATCH --ntasks=1                    # Run a single task	
#SBATCH --cpus-per-task=5             # Number of CPU cores per task
#SBATCH --mem=900gb                   # Job memory request
#SBATCH --time=45:00:00               # Time limit hrs:min:sec
#SBATCH --output=SoybeanToSoybean.%j.out   # Standard output log
#SBATCH --error=SoybeanToSoybean.%j.err    # Standard error log
#SBATCH --array=0-13                   # Array range


List=(10 13 14 16 4 7 18 25 27 29 34 35 36 39)

module load Python/3.7.4-GCCcore-8.3.0

cd /scratch/sb14489/1-2.ML_NewTry/5.SoybeanToSoybean
module load Anaconda3/2020.02

#conda init --all
#conda activate environmentName
source activate /home/sb14489/.conda/envs/environmentName
#conda info --envs
#conda update keras

python kgrammer_FeatureSelection_Upsampling_AlltheARFs_Final.py "${List[SLURM_ARRAY_TASK_ID]}"
