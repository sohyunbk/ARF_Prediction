import os, glob

ARFNumbList = ["4","16","18","27","29","34","7","10","14","25","36","39"]

outfile = open("Soybean_AllARFs.txt","w")
outfile.write("ARF\tACC\tACC_sd\tFPR\tFPR_sd\tFNR\tFNR_sd\tRatio\n")

Stat_file = open("/scratch/sb14489/1.ML_ARF/2.ML/1-3.MakeFASTA_Soybean_Redo/2.OutputFile/Soybean_CountNumber.txt","r")
Stat_List = Stat_file.readlines()

for i in range(len(ARFNumbList)):
    sARF = ARFNumbList[i]
    Name = sARF+"_5FC_3Re.txt"
    infile = open(Name,"r")
    for sLine in infile:
        outfile.write("\t".join(sLine.strip().split("+-"))+"\t")
    outfile.write(Stat_List[i].split("\t")[3].replace("1:",""))

    infile.close()

Stat_file.close()
outfile.close()   
