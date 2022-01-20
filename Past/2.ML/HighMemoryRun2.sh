#!/bin/bash
#SBATCH --job-name=highmemtest        # Job name
#SBATCH --partition=highmem_p         # Partition (queue) name
#SBATCH --ntasks=1                    # Run a single task	
#SBATCH --cpus-per-task=10             # Number of CPU cores per task
#SBATCH --mem=400gb                   # Job memory request
#SBATCH --time=15:00:00               # Time limit hrs:min:sec
#SBATCH --output=SeqM.%j.out   # Standard output log
#SBATCH --error=SeqM.%j.err    # Standard error log

module load Python/3.7.4-GCCcore-8.3.0

cd /scratch/sb14489/1-2.ML_NewTry/2.ML
module load Anaconda3/2020.02
#conda init --all
#conda activate environmentName
source activate /home/sb14489/.conda/envs/environmentName
python OneHotEncoding_SequentialModel.py   
