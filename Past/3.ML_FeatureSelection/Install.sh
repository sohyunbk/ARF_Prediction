#!/bin/bash
#SBATCH --job-name=highmemtest        # Job name
#SBATCH --partition=highmem_p         # Partition (queue) name
#SBATCH --ntasks=1                    # Run a single task	
#SBATCH --cpus-per-task=10             # Number of CPU cores per task
#SBATCH --mem=400gb                   # Job memory request
#SBATCH --time=70:00:00               # Time limit hrs:min:sec
#SBATCH --output=Install.%j.out   # Standard output log
#SBATCH --error=Install.%j.err    # Standard error log

pip install pandas
