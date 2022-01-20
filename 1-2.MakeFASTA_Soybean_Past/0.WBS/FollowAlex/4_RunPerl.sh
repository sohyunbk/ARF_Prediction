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

module load Anaconda3/2020.02
#conda init --all
#conda activate environmentName
source activate /home/sb14489/.conda/envs/ucsc
conda activate  /home/sb14489/.conda/envs/ucsc

cd /scratch/sb14489/1-2.ML_NewTry/1-2.MakeFASTA_Soybean/0.WBS/FollowAlex

perl callUMRs.pl Gmax_508_v4.0_OnlyChr_tiles.allc.bed  > Gmax_508_v4.0_OnlyChr_tiles.classified.bed
perl condense_bins.pl Gmax_508_v4.0_OnlyChr_tiles.classified.bed > Gmax_508_v4.0_OnlyChr_tiles.condensed.classified.bed
#bedtools makewindows -g Gmax_508_v4.0_OnlyChr.fa.fai -w 100 > Gmax_508_v4.0_OnlyChr_tiles.bed
#bedtools intersect -a Gmax_508_v4.0_OnlyChr_tiles.bed -b allc_Gmax_AddOneCol_OnlyChr.tsv -wa -wb > Gmax_508_v4.0_OnlyChr_tiles.allc.bed
