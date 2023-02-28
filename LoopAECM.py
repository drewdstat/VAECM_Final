import os,sys,time,copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]=""   # without GPU

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader

from lib_CM import *

import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import homogeneity_score as homog
# from sklearn.metrics import f1_score
from sklearn.metrics.cluster import adjusted_rand_score

from Module.module_AECM import AECM
from Module.utils import *
from Archi import *
from Makenpz import *

def LoopAECM(dfnum=1,jsonfilename="resclusts.json",FNAME="GeneralAECM.npz",sampargsseed=None,sel=True,selectrand=20,save=False,
             erange=[500,1000,1500],krange=range(2,7),nlayers=range(1,4),batchrange=[50,100,200],
             betarange=[20,50],alpharange=[0.01, 0.05],C1range=[15,30],C2range=[30,60],C3range=[20,40],
             lbdrange=[.1,1.],lrrange=[3,2],optrange=["adam","adam_decay"],seedrange=range(1,6)):
    argsgridsmall = np.array([
            (ep, nl, batch, beta, alpha, c1, c2, c3, lb, lr, opt, sd) for ep in erange for nl in nlayers 
            for batch in batchrange  for beta in betarange for alpha in alpharange for c1 in C1range 
            for c2 in C2range for c3 in C3range for lb in lbdrange for lr in lrrange for opt in optrange
            for sd in seedrange])
    def seed_everything(seed):
            random.seed(seed)
            os.environ['pythonhashseed'] = str(seed)   # set PYTHONHASHSEED env var at fixed value
            np.random.seed(seed) 
            torch.manual_seed(seed)   
            torch.cuda.manual_seed(seed)               # if you use multi-GPU
            # torch.backends.cudnn.deterministic = True  # for DL, random number for CuDNN seed
            # torch.backends.cudnn.benchmark = True
    if not(sampargsseed is None):
        seed_everything(sampargsseed)  
    
    if sel:
        samplerows = np.random.choice(range(argsgridsmall.shape[0]), size=selectrand, replace=False, p=None)
        samplerows = samplerows[np.argsort(samplerows)]
        argsgridsmall = argsgridsmall[samplerows,] 
    
    k = krange[0]
    tt = np.array([k]*argsgridsmall.shape[0])
    tt.shape=(argsgridsmall.shape[0],1)
    newgrid=np.append(np.array(argsgridsmall),tt,1)
    newgrid.shape=(argsgridsmall.shape[0],argsgridsmall.shape[1]+1)
    argsgridmain = newgrid
    for k in krange[1:]:
        tt = np.array([k]*argsgridsmall.shape[0])
        tt.shape=(argsgridsmall.shape[0],1)
        newgrid=np.append(np.array(argsgridsmall),tt,1)
        newgrid.shape=(argsgridsmall.shape[0],argsgridsmall.shape[1]+1)
        argsgridmain = np.append(argsgridmain,newgrid,0)
    summ = pd.DataFrame(argsgridmain)
    summ.columns=["Epochs","Layers","Batches","Beta","Alpha","C1","C2","C3","LBD","LR","OPT","Seed","NZ"]
    # if(seed is None): 
    #     summ['Seed'] = [str("None")] * summ.shape[0]
    # else:
    #     summ['Seed'] = [seed] * summ.shape[0]
    summ['ARI'] = [0] * summ.shape[0]
    summ['ACC'] = [0] * summ.shape[0]
    summ['NMI'] = [0] * summ.shape[0]
    #summ['LLK'] = [0] * summ.shape[0]
    summ['kARI'] = [0] * summ.shape[0]
    summ['kACC'] = [0] * summ.shape[0]
    summ['kNMI'] = [0] * summ.shape[0]
    summ['TotalLoss'] = [0] * summ.shape[0]
    summ['Loss_op1'] = [0] * summ.shape[0]
    summ['Loss_op2'] = [0] * summ.shape[0]
    summ['Loss_op2b'] = [0] * summ.shape[0]
    summ['Loss_op3'] = [0] * summ.shape[0]
    summ['Loss_op4'] = [0] * summ.shape[0]
    summ['Loss_E1'] = [0] * summ.shape[0]
    summ['Loss_E2'] = [0] * summ.shape[0]
    summ['Loss_E3'] = [0] * summ.shape[0]
    summ['Loss_E4'] = [0] * summ.shape[0]
    
    Makenpz(dfnum,jsonfilename)
    INIT  = "rand"
    SAVE  = save
    NAME = "EX1"
    LOAD = np.load('data/'+NAME+'.npz',allow_pickle=True)
    DATA = LOAD['x'].astype('float32')
    TRUE = LOAD['y']
    del LOAD

    N,D  = DATA.shape
    ND = (N * D)
    #K    = int( TRUE.max()+1 ) #only if we have simulated data
    MOD = []
    LLK = []
    CMloss = []
    LBL = []
    kLBL = []
    Zlayer = []
    Reconstr = []
    Gamma = []
    Mu = []
    #ARI,NMI,ACC = [],[],[]
    #kARI,kNMI,kACC = [],[],[]
    #WGT,EPC = [],[]
    #ari_list = []
    
    for i in range(argsgridmain.shape[0]):
        OUT = int(argsgridmain[i,argsgridmain.shape[1]-1])
        epochs = int(argsgridmain[i,0])
        nlayers = int(argsgridmain[i,1])
        BATCH = int(argsgridmain[i,2])
        BETA = float(argsgridmain[i,3])
        ALPHA = float(argsgridmain[i,4])
        C1 = int(argsgridmain[i,5])
        C2 = int(argsgridmain[i,6])
        C3 = int(argsgridmain[i,7])
        LBD = float(argsgridmain[i,8])
        LR = int(argsgridmain[i,9])
        OPT = argsgridmain[i,10]
        OPTNAME = OPT + "|" + str(LR)
        thisseed = int(argsgridmain[i,11])
        seed_everything(thisseed)
        ARCHI = getarchi(nlayers=nlayers,D=D,OUT=OUT,
                         C1=C1,C2=C2,C3=C3)
        
        MODEL = AECM( 
                    architecture=ARCHI, 
                    n_clusters=OUT, 
                    true_labels=TRUE, 
                    beta=BETA, 
                    lbd=LBD
                )

        epc = MODEL.fit( 
                    x=DATA,
                    y=TRUE,
                    alpha=ALPHA, 
                    batch_size=BATCH, 
                    epoch_size=epochs, 
                    optimizer_name=OPTNAME, #if NAME not in ['CIFAR10','10X73K','USPS','PENDIGIT','R10K'] adam|3
                    #else 'adam_decay|3', 
                    optimizer_step=int( 150*(N/BATCH) ),
                    print_interval=epochs, 
                    verbose=False
                )
        
        MOD.append(MODEL)
        thisllk = MODEL.loss(DATA,ALPHA)
        LLK.append(thisllk)
        LBL.append(MODEL.predict(DATA))
        kLBL.append( MODEL.predict_km(DATA) )
        tempz = MODEL.encode(DATA,False)
        Zlayer.append(tempz)
        thiscmloss = MODEL.loss_cm(DATA,ALPHA)
        CMloss.append(thiscmloss)
        Reconstr.append(MODEL.decode(tempz,False))
        tempgamma = MODEL.gamma(tempz,False)
        Gamma.append(tempgamma)
        Mu.append(MODEL.mu(tempgamma,False))
        summ.loc[i,'ARI'] = ari(TRUE, LBL[-1])
        summ.loc[i,'ACC'] = acc(TRUE, LBL[-1])
        summ.loc[i,'NMI'] = nmi(TRUE, LBL[-1])
        #summ.loc[i,'LLK'] = MODEL.loss(DATA,0) #LLK[-1]
        summ.loc[i,'kARI'] = ari(TRUE, kLBL[-1])
        summ.loc[i,'kACC'] = acc(TRUE, kLBL[-1])
        summ.loc[i,'kNMI'] = nmi(TRUE, kLBL[-1])
        summ.loc[i,'TotalLoss'] = tf.reduce_sum(thisllk).numpy()
        summ.loc[i,'Loss_op1'] = thisllk.numpy()[0]
        summ.loc[i,'Loss_op2'] = thisllk.numpy()[1]
        summ.loc[i,'Loss_op2b'] = thisllk.numpy()[2]
        summ.loc[i,'Loss_op3'] = thisllk.numpy()[3]
        summ.loc[i,'Loss_op4'] = thisllk.numpy()[4]
        summ.loc[i,'Loss_E1'] = thiscmloss.numpy()[0] / ND
        summ.loc[i,'Loss_E2'] = thiscmloss.numpy()[1] / ND
        summ.loc[i,'Loss_E3'] = thiscmloss.numpy()[2] / ND
        summ.loc[i,'Loss_E4'] = thiscmloss.numpy()[3]
    if SAVE:
        #FNAME = NAME+'/save/save-loopaecm-'+ INIT + '.npz'

        if not os.path.exists(NAME+'/'):
            os.mkdir(NAME+'/')
        if not os.path.exists(NAME+'/save/'):
            os.mkdir(NAME+'/save/')
        print("*** I will save in ",FNAME)
        np.savez(FNAME,
            ari_list=ari_list,
            llk=LLK,
            cmloss = CMloss,
            lbl=LBL,
            ari=ARI,nmi=NMI,acc=ACC
        )
        
        print("*** I will save in ",FNAME)
        np.savez(FNAME,
            # ari_list=ari_list,
            summ=summ,
            # z=Zlayer,
            llk=LLK,
            cmloss = CMloss,
            lbl=LBL,
            klbl = kLBL,
            reconstr=Reconstr #,
            # ari=ARI,nmi=NMI,acc=ACC
        )
        
        import re
        zfname = re.sub(".npz","_zlayers.npz",FNAME)
        np.savez(zfname,*Zlayer)
        mfname = re.sub(".npz","_mu.npz",FNAME)
        np.savez(mfname,*Mu)
        gfname = re.sub(".npz","_gamma.npz",FNAME)
        np.savez(gfname,*Gamma)
    #        0    1     2      3    4     5      6        7     8   9
    return summ, MOD, Zlayer, LLK, LBL, kLBL, Reconstr, Gamma, Mu, CMloss #ari_list, ARI, NMI, ACC,
