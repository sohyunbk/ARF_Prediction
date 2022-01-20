#!/bin/bash
#SBATCH --job-name=Gmax.index
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=5gb
#SBATCH --time=120:00:00
cd $SLURM_SUBMIT_DIR
ml methylpy
methylpy build-reference --input-files Gmax_189.addCt.fa  --output-prefix Gmax_189.addCt  --aligner bowtie2 && echo This-Work-is-Completed!
