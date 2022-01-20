#!/bin/bash
#SBATCH --job-name=highmemtest        # Job name
#SBATCH --partition=highmem_p         # Partition (queue) name
#SBATCH --ntasks=1                    # Run a single task	
#SBATCH --cpus-per-task=10             # Number of CPU cores per task
#SBATCH --mem=300gb                   # Job memory request
#SBATCH --time=45:00:00               # Time limit hrs:min:sec
#SBATCH --output=MakeFa.%j.out   # Standard output log
#SBATCH --error=MakeFa.%j.err    # Standard error log
#SBATCH --array=0-13                   # Array range


List=(18 25 27 29 34 35 36 39 10 13 14 16 4 7)
module load Python/3.7.4-GCCcore-8.3.0


python 2-1_CutAll_AllARFs_Ratio.py "${List[SLURM_ARRAY_TASK_ID]}"
