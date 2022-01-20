#!/bin/bash
#SBATCH --job-name=highmemtest        # Job name
#SBATCH --partition=highmem_p         # Partition (queue) name
#SBATCH --ntasks=1                    # Run a single task	
#SBATCH --cpus-per-task=5             # Number of CPU cores per task
#SBATCH --mem=50gb                   # Job memory request
#SBATCH --time=45:00:00               # Time limit hrs:min:sec
#SBATCH --output=MakeWg.%j.out   # Standard output log
#SBATCH --error=MakeWg.%j.err    # Standard error log


cd /scratch/sb14489/1-2.ML_NewTry/1-2.MakeFASTA_Soybean/0.WBS/AllC_JBrowser
conda activate /home/sb14489/.conda/envs/ucsc
source activate /home/sb14489/.conda/envs/ucsc

python3 /work/rjslab/bth29393/jbscripts/allc_to_bigwig_pe_v3.py -sort Gmax_508_v4.0_OnlyChr.fa.fai allc_Gmax_OnlyChr.tsv

