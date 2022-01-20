#!/bin/bash
#SBATCH --job-name=highmemtest        # Job name
#SBATCH --partition=highmem_p         # Partition (queue) name
#SBATCH --ntasks=1                    # Run a single task	
#SBATCH --cpus-per-task=10             # Number of CPU cores per task
#SBATCH --mem=400gb                   # Job memory request
#SBATCH --time=15:00:00               # Time limit hrs:min:sec
#SBATCH --output=Simulation.%j.out   # Standard output log
#SBATCH --error=Simulation.%j.err    # Standard error log

cd /scratch/sb14489/1-2.ML_NewTry/6.AllSimulations_Maize
module load Python/3.7.4-GCCcore-8.3.0
module load Anaconda3/2020.02
source activate /home/sb14489/.conda/envs/environmentName


python 3.kgrammer_Multithread_3Repeat.py
