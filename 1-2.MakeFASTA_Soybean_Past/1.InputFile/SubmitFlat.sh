#!/bin/bash
#SBATCH --job-name=highmemtest        # Job name
#SBATCH --partition=highmem_p         # Partition (queue) name
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=32             # Number of CPU cores per task
#SBATCH --mem=400gb                   # Job memory request
#SBATCH --time=65:00:00               # Time limit hrs:min:sec
#SBATCH --output=Flat.%j.out   # Standard output log
#SBATCH --error=Flat.%j.err    # Standard error log
#SBATCH --array=0-14                   # Array range

module load  R/4.0.0-foss-2019b

Rscript 5_Socrates_loadRDS_ChangeOptions.R "${List[SLURM_ARRAY_TASK_ID]}"

Data_Path=(/scratch/sb14489/1-2.ML_NewTry/1-2.MakeFASTA_Soybean/1.InputFile)
RawDataFormat=.narrowPeak

cd "$Data_Path"

List=`find "$Data_Path" -name "*""$RawDataFormat" | sed 's|.*/||'`
echo "This is input File: "$List

./flatfile-to-json.pl --bed "${List[SLURM_ARRAY_TASK_ID]}"  --trackLabel "${List[SLURM_ARRAY_TASK_ID]}" --out Track
