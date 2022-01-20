#!/bin/bash
#SBATCH --job-name=highmemtest        # Job name
#SBATCH --partition=highmem_p         # Partition (queue) name
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=10             # Number of CPU cores per task
#SBATCH --mem=400gb                   # Job memory request
#SBATCH --time=62:00:00               # Time limit hrs:min:sec
#SBATCH --output=highmemtest.%j.out   # Standard output log
#SBATCH --error=highmemtest.%j.err    # Standard error log
#SBATCH --mail-type=END         # Mail events (NONE, BEGIN, END, FAIL, ALL)

module load BEDTools/2.29.2-GCC-8.3.0

cd /scratch/sb14489/1-2.ML_NewTry/1-2.MakeFASTA_Soybean/0.WBS/FollowAlex

bedtools makewindows -g Gmax_508_v4.0_OnlyChr.fa.fai -w 100 > Gmax_508_v4.0_OnlyChr_tiles.bed
bedtools intersect -a Gmax_508_v4.0_OnlyChr_tiles.bed -b allc_Gmax_AddOneCol_OnlyChr.tsv -wa -wb > Gmax_508_v4.0_OnlyChr_tiles.allc.bed
