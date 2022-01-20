#!/bin/bash
#SBATCH --job-name=testserial         # Job name
#SBATCH --partition=batch             # Partition (queue) name
#SBATCH --ntasks=4                    # Run on a single CPU
#SBATCH --mem=1gb                     # Job memory request
#SBATCH --time=02:00:00               # Time limit hrs:min:sec
#SBATCH --output=highmemtest.%j.out   # Standard output log
#SBATCH --error=highmemtest.%j.err    # Standard error log

module load BEDTools/2.29.2-GCC-8.3.0

cd /scratch/sb14489/1-2.ML_NewTry/1-2.MakeFASTA_Soybean/0.WBS/FollowAlex

bedtools sort -i  Gmax_508_v4.0_OnlyChr_tiles.condensed.classified_OnlyUMR.bed > Gmax_508_v4.0_OnlyChr_tiles.condensed.classified_OnlyUMR_sorted.bed
#bedtools intersect -a Gmax_508_v4.0_OnlyChr_tiles.condensed.classified_OnlyUMR_sorted.bed -b Gmax_508_v4.0_OnlyChr_tiles.condensed.classified_OnlyUMR_sorted.bed > UMR.txt
