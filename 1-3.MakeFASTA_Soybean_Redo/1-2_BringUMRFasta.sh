#!/bin/bash
#SBATCH --job-name=highmemtest        # Job name
#SBATCH --partition=highmem_p         # Partition (queue) name
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=32             # Number of CPU cores per task
#SBATCH --mem=200gb                   # Job memory request
#SBATCH --time=1:00:00               # Time limit hrs:min:sec

#SBATCH --output=./log/MakeFasta.%j.out   # Standard output log
#SBATCH --error=./log/MakeFasta.%j.err    # Standard error log

#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sb14489@uga.edu  # Where to send mail	

cd  /scratch/sb14489/1.ML_ARF/2.ML/1-3.MakeFASTA_Soybean_Redo

module load BEDTools/2.29.2-GCC-8.3.0

bedtools getfasta -fi ./1.InputFile/Gmax_508_v4.0_OnlyChr.fa -bed ./1.InputFile/UMR.bed > ./1.InputFile/UMR.fa
