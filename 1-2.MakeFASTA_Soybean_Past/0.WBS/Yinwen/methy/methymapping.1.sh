#!/bin/bash
#SBATCH --job-name=Gmax.1
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --mem=100gb
#SBATCH --partition=highmem_p

cd $SLURM_SUBMIT_DIR
module load methylpy
module load picard/2.16.0-Java-1.8.0_144
methylpy paired-end-pipeline  --read1-files /scratch/yz46606/12_ibm1/v2.0/1_mintron/data/Glycine_max/v4/methy/data/Gmax_1.fastq --read2-files /scratch/yz46606/12_ibm1/v2.0/1_mintron/data/Glycine_max/v4/methy/data/Gmax_2.fastq --sample Gmax --forward-ref /scratch/yz46606/12_ibm1/v2.0/1_mintron/data/Glycine_max/v4/Gmax_508_v4.0.addCt_f --reverse-ref /scratch/yz46606/12_ibm1/v2.0/1_mintron/data/Glycine_max/v4/Gmax_508_v4.0.addCt_r --ref-fasta  /scratch/yz46606/12_ibm1/v2.0/1_mintron/data/Glycine_max/v4/Gmax_508_v4.0.addCt.fa --binom-test True --min-cov 3   --sort-mem 80G  --unmethylated-control chrCt --num-procs 3  --remove-clonal true --compress-output false --path-to-picard="/apps/eb/picard/2.16.0-Java-1.8.0_144/"
