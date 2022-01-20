#!/bin/bash
#SBATCH --job-name=testserial         # Job name
#SBATCH --partition=batch             # Partition (queue) name
#SBATCH --ntasks=4                    # Run on a single CPU
#SBATCH --mem=1gb                     # Job memory request
#SBATCH --time=04:00:00               # Time limit hrs:min:sec
#SBATCH --output=highmemtest.%j.out   # Standard output log
#SBATCH --error=highmemtest.%j.err    # Standard error log
#SBATCH --mail-type=END         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sb14489@uga.edu  # Where to send mail	

module load Python/3.7.4-GCCcore-8.3.0

python /scratch/sb14489/1.ML_DAPSeq/13.AllCombi/1.MakeFASTA/2-1_CutAll_AllARFs_Ratio_print.py
