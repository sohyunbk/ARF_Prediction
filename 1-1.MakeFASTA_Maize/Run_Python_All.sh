#PBS -S /bin/bash
#PBS -N MakeFA
#PBS -q batch 
#PBS -l nodes=1:ppn=24
#PBS -l walltime=400:00:00
#PBS -l mem=20gb

module load Python/2.7.14-foss-2016b
#python /scratch/sb14489/1.ML_DAPSeq/13.AllCombi/1.MakeFASTA/2-1_CutAll.py
python /scratch/sb14489/1.ML_DAPSeq/13.AllCombi/1.MakeFASTA/2-4_MakeRatioData.py
