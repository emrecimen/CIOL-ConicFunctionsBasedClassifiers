# Developer: Emre Cimen
# Date: 06-17-2019


# Include CF.py file for Random Subspace Ensemble Classifier based on Conic Functions
# Datasets' last column should be class information.

 
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import time
import numpy as np
from CF import EnsambleCF
from sklearn import preprocessing


np.random.seed(0)

# 1 if separated test file, 0 is cross validation 
if 1:
    dfTrain = pd.read_csv("/users/path/train.csv", header=None)
    dfTest = pd.read_csv("/users/path/test.csv", header=None)

    start_time = time.time()

    X = dfTrain[dfTrain.columns[0:-1]]
    YD = dfTrain[dfTrain.columns[-1]]


    XTest = dfTest[dfTest.columns[0:-1]]
    YDTest = dfTest[dfTest.columns[-1]]

    YdfFac = pd.factorize(pd.concat([YD, YDTest]))[0]
    Y=YdfFac[:YD.shape[0]]
    YTest = YdfFac[YD.shape[0]:]

    #Grid search for parameters
    MEM = [5, 10, 20]
    RAT = [0.1, 0.2, 0.3]
    ALF = [0.01, 0.05, 0.1, 0.2]
    paracount = 0
    for mem in MEM:
        for rat in RAT:
            for alf in ALF:
                start_time = time.time()
                clf = EnsambleCF(maxRatio=rat, alfa=alf, member=mem)
                mystr = ""
                fld = 0
                paracount = paracount + 1
                print(paracount)


                clf.fit(X, Y)
                test_scores = clf.score(XTest, YTest)
                train_score = clf.score(X, Y)
                mystr = mystr + "*Fold" + str(fld) + "*Member*" + str(mem) + "*Ratio*" + str(rat) + "*Alfa*" + str(
                    alf) + "*Train *" + str(train_score) + "*Test*" + str(test_scores) + "*Time *" + str(
                    time.time() - start_time)

                with open("Ensemble-Predictions.txt", "a") as myfile:
                    mystr = mystr + "\n"
                    myfile.write(mystr)



else:


    df=pd.read_csv("/users/path/dataset.csv",header=None)
    X = df[df.columns[0:-1]]

    YD = df[df.columns[-1]]
    Y=pd.factorize(YD)[0]

    #5 fold cross validation
    kf = StratifiedKFold(n_splits=5)

    #Grid search for parameters
    MEM=[5,10,20]
    RAT=[0.1,0.2,0.3]
    ALF=[0.01, 0.05, 0.1, 0.2]
    paracount=0
    for mem in MEM:
        for rat in RAT:
            for alf in ALF:
                start_time = time.time()
                clf = EnsambleCF(maxRatio=rat, alfa=alf,member=mem)
                mystr = ""
                fld=0
                paracount=paracount+1
                print(paracount)
                for train_index, test_index in kf.split(X,Y):
                    fld=fld+1
                    clf.fit(X.iloc[train_index], Y[train_index])
                    test_scores=clf.score(X.iloc[test_index], Y[test_index])
                    train_score=clf.score(X.iloc[train_index], Y[train_index])
                    mystr = mystr+"*Fold"+str(fld) +"*Member*"+str(mem) +"*Ratio*"+str(rat) +"*Alfa*"+str(alf) +"*Train *"+str(train_score) + "*Test*" + str(test_scores) + "*Time *" + str(time.time() - start_time)

                with open("Ensemble-Predictions.txt", "a") as myfile:
                    mystr = mystr + "\n"
                    myfile.write(mystr)






