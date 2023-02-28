import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json

def Makenpz(dfnum=1,jsonfilename="resclusts.json"):
    with open(jsonfilename) as f:
        resclusts = json.load(f)
    if str(type(resclusts))=="<class 'dict'>":
        resclustsindex = "Sim" + str(dfnum)
    else: 
        resclustsindex = dfnum
    ex = pd.DataFrame(resclusts[resclustsindex]) #'Sim1'
    ex["DiseaseState"] = ex["DiseaseState"].replace("D",'', regex=True)
    ex['DiseaseState'] = pd.to_numeric(ex['DiseaseState']) 
    ex['DiseaseState'] = ex['DiseaseState'] - 1 

    scaler = StandardScaler()
    x = scaler.fit_transform(ex.drop(columns=["DiseaseState","Cohort","DisCoh"]))
    y = ex['DiseaseState'].to_numpy()
    x_train = x[:-200]
    y_train = y[:-200]
    x_test = x[-200:]
    y_test = y[-200:]
    s = (x_train.shape, x_test.shape)
    #np.savez('~/clusteringModule/data/ex1.npz',x=x,y=y,s=s)
    np.savez('./data/EX1.npz',x=x,y=y,s=s)
    try:
        os.mkdir('EX1')   
        os.mkdir('EX1/save')
    except FileExistsError:
        pass

#ex = pd.read_csv("./example_resid.csv", index_col=0)  # removed the row index
