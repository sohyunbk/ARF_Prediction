#PBS -S /bin/bash
#PBS -N All
#PBS -q highmem_q 
#PBS -l nodes=1:ppn=24
#PBS -l walltime=400:00:00
#PBS -l mem=200gb

module load Python/3.7.0-foss-2018a

#python /scratch/sb14489/1.ML_DAPSeq/9.100Cut_trial/3.ML/kgrammer_Original.py
python /scratch/sb14489/1.ML_DAPSeq/13.AllCombi/2.ML/ARF4/kgrammer_Original_TwoCases.py
