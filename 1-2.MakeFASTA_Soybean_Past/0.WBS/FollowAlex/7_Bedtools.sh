#!/bin/bash
#SBATCH --job-name=highmemtest        # Job name
#SBATCH --partition=highmem_p         # Partition (queue) name
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=32             # Number of CPU cores per task
#SBATCH --mem=400gb                   # Job memory request
#SBATCH --time=02:00:00               # Time limit hrs:min:sec
#SBATCH --output=highmemtest.%j.out   # Standard output log
#SBATCH --error=highmemtest.%j.err    # Standard error log

module load BEDTools/2.29.2-GCC-8.3.0

cd /scratch/sb14489/1-2.ML_NewTry/1-2.MakeFASTA_Soybean/0.WBS/FollowAlex

bedtools merge -i  Gmax_508_v4.0_OnlyChr_tiles.condensed.classified_OnlyUMR_sorted.bed > UMR.txt
#bedtools intersect -a Gmax_508_v4.0_OnlyChr_tiles.condensed.classified_OnlyUMR_sorted.bed -b Gmax_508_v4.0_OnlyChr_tiles.condensed.classified_OnlyUMR_sorted.bed > UMR.txt
