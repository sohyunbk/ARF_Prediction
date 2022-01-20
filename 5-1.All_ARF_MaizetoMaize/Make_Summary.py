import os, glob

ARFNumbList = ["4","16","18","27","29","34","7","10","14","25","36","39"]

outfile = open("Maize_AllARFs.txt","w")
outfile.write("ARF\tACC\tACC_sd\tFPR\tFPR_sd\tFNR\tFNR_sd\tRatio\n")

Stat_file = open("/scratch/sb14489/1.ML/1-1.MakeFASTA_Maize/2.Output/CountNumber_Maize.txt","r")
Stat_List = Stat_file.readlines()

for i in range(len(ARFNumbList)):
    sARF = ARFNumbList[i]
    Name = "ARF"+sARF+"_5FC_3Re.txt"
    infile = open(Name,"r")
    for sLine in infile:
        outfile.write("\t".join(sLine.strip().split("+-"))+"\t")
    outfile.write(Stat_List[i].split("\t")[3].replace("1:",""))

    infile.close()

Stat_file.close()
outfile.close()
    
