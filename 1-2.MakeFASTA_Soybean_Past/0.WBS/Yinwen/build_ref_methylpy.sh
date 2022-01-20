#!/bin/bash
#SBATCH --job-name=Gmax.index
#SBATCH --partition=highmem_p
#SBATCH --ntasks=1
#SBATCH --mem=10gb
#SBATCH --time=120:00:00
cd $SLURM_SUBMIT_DIR
ml methylpy
methylpy build-reference --input-files Gmax_508_v4.0.addCt.fa  --output-prefix Gmax_508_v4.0.addCt  --aligner bowtie2 && echo This-Work-is-Completed!
