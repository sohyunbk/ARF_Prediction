#!/bin/bash
#SBATCH --job-name=testserial         # Job name
#SBATCH --partition=batch             # Partition (queue) name
#SBATCH --ntasks=4                    # Run on a single CPU
#SBATCH --mem=1gb                     # Job memory request
#SBATCH --time=02:00:00               # Time limit hrs:min:sec
#SBATCH --output=NumberofPeaksOverlappedWithUMR.%j.out   # Standard output log
#SBATCH --error=CheckNumber.%j.err    # Standard error log
#SBATCH --mail-type=END         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sb14489@uga.edu  # Where to send mail	

cd /scratch/sb14489/1-2.ML_NewTry/1-2.MakeFASTA_Soybean
module load Python/3.7.4-GCCcore-8.3.0

python 2-1_CutAll_AllARFs_Ratio_print.py
