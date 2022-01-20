#!/bin/bash
#SBATCH --job-name=highmemtest        # Job name
#SBATCH --partition=highmem_p         # Partition (queue) name
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=32             # Number of CPU cores per task
#SBATCH --mem=400gb                   # Job memory request
#SBATCH --time=65:00:00               # Time limit hrs:min:sec

cd  /scratch/sb14489/1-2.ML_NewTry/1-2.MakeFASTA_Soybean

module load BEDTools/2.29.2-GCC-8.3.0

bedtools getfasta -fi ./1.InputFile/Gmax_508_v4.0_OnlyChr.fa -bed ./1.InputFile/UMR.txt > ./1.InputFile/UMR.fa
