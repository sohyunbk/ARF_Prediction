#!/bin/bash
#SBATCH --job-name=TestResults        # Job name
#SBATCH --partition=batch
#SBATCH --ntasks=1                    # Run a single task	
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=10gb                   # Job memory request
#SBATCH --time=12:00:00               # Time limit hrs:min:sec
#SBATCH --output=CheckResult.%j.out   # Standard output log
#SBATCH --error=CheckResult.%j.err    # Standard error log
#SBATCH --mail-type=END         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sb14489@uga.edu  # Where to send mail	

module load Python/3.7.4-GCCcore-8.3.0
cd /scratch/sb14489/1.ML_DAPSeq/13.AllCombi/2.ML/ARF4

python /scratch/sb14489/1.ML_DAPSeq/13.AllCombi/2.ML/ARF4/kgrammer_Original_AllARFs.py
