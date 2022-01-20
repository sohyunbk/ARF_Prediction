import os,sys, glob

for sFiles in glob.glob("*_PredictionInfo"):
    infile = open(sFiles,"r")
    Outfile_FP = open(sFiles.replace("_PredictionInfo","_FP_ActualNonARF_ClassifiedtoARF.bed"),"w")
    Outfile_TP = open(sFiles.replace("_PredictionInfo","_TP_ActualARF_ClassifiedtoARF.bed"),"w")
    Outfile_FN = open(sFiles.replace("_PredictionInfo","_FN_ActualARF_ClassifiedtoNonARF.bed"),"w")
    Outfile_TN = open(sFiles.replace("_PredictionInfo","_TN_ActualNonARF_ClassifiedtoNonARF.bed"),"w")
    
    infile.readline()
    for sLine in infile:
        sList = sLine.strip().split("\t")
        Pos_list = sList[0].split(":")
        sChr = Pos_list[0].replace(">","")
        sStart = Pos_list[1]
        sEnd = Pos_list[2]
        Answer = sList[2]
        Prediction = sList[3]
        if Answer == "0.0" and Prediction == "0.0":
            Outfile_TN.write(sChr+"\t"+sStart+"\t"+sEnd+"\n")
        elif Answer == "1.0" and Prediction == "1.0":
            Outfile_TP.write(sChr+"\t"+sStart+"\t"+sEnd+"\n")
        elif Answer == "1.0" and Prediction == "0.0":
            Outfile_FN.write(sChr+"\t"+sStart+"\t"+sEnd+"\n")
        elif Answer == "0.0" and Prediction == "1.0":        
            Outfile_FP.write(sChr+"\t"+sStart+"\t"+sEnd+"\n")

    infile.close()
    Outfile_FP.close()
    Outfile_TP.close()
    Outfile_FN.close()
    Outfile_TN.close() 
