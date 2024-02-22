import torch
import numpy as np
import pandas as pd
from os.path import splitext, basename
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

def get_feature(cancer_type, batch_size, training):

    csv_CN_file = 'test\\'+cancer_type+'\\preprocessed_CNA.csv'
    csv_CN = pd.read_csv(csv_CN_file, header=None, index_col=None, sep=',')

    csv_meth_file = 'test\\'+cancer_type+'\\preprocessed_DNAMeth.csv'
    csv_meth = pd.read_csv(csv_meth_file, header=None, index_col=None, sep=',')

    csv_mirna_file = 'test\\'+cancer_type+'\\preprocessed_MIRNA.csv'
    csv_mirna = pd.read_csv(csv_mirna_file, header=None, index_col=None, sep=',')

    csv_rna_file = 'test\\'+cancer_type+'\\preprocessed_RNASeq.csv'
    csv_rna = pd.read_csv(csv_rna_file, header=None, index_col=None, sep=',')

    csv_psi_file = 'test\\'+cancer_type+'\\preprocessed_PSI.csv'
    csv_psi = pd.read_csv(csv_psi_file, header=None,index_col=None, sep=',')



    feature = np.concatenate((csv_CN, csv_meth, csv_mirna, csv_rna, csv_psi), axis=1)
    ds = feature.shape[1]

    minmaxscaler = MinMaxScaler()
    feature = minmaxscaler.fit_transform(feature)
    feature = torch.tensor(feature)

    dataloader = DataLoader(feature, batch_size=batch_size, shuffle=training, drop_last= True)

    return dataloader,ds

