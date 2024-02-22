import numpy as np
import torch
import torchvision
import argparse
from modules import ae, network, contrastive_loss
from utils import yaml_config_hook, save_model
from torch.utils import data
from dataloader import *
import copy
from tensorboardX import SummaryWriter
import os
import matplotlib as mpl
import lifelines
import sklearn.metrics
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
import pandas as pd
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.pyplot import style
import seaborn as sns
import matplotlib as mpl
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize



# Cox-PH
def cox_ph(pd1):
    pd1.pop("name")
    cph = CoxPHFitter()
    cph.fit(df=pd1, duration_col='days', event_col='status', show_progress=True)

    return cph


# KM
def KM(pd_withcluster,cancer_type):
    kmf = KaplanMeierFitter()
    for name, grouped_df in pd_withcluster.groupby('cluster'):
        kmf.fit(grouped_df["days"], grouped_df["status"], label=name)
        kmf.plot_survival_function()
    results = multivariate_logrank_test(pd_withcluster['days'], pd_withcluster['cluster'], pd_withcluster['status'])
    results.print_summary()
    print(results.p_value)
    m = results.summary
    p = m.iloc[0,1]
    p = round(p,5)
    # plt.text(2000, 0.9, 'p = '+str(p), fontdict={'family': 'serif', 'size': 16, 'color': 'black'})
    # plt.text(800, 0.9, 'p = '+str(p), fontdict={'family': 'serif', 'size': 16, 'color': 'black'})
    plt.text(800, 0.9, 'p = '+str(p), fontdict={'family': 'serif', 'size': 16, 'color': 'black'})
    plt.savefig('out\\'+cancer_type+'_KM.png')
    plt.clf()

    return m



def meanshift(pd_withoutclinial,cancer_type,bw):
    data_tsne = TSNE(n_components=2).fit_transform(pd_withoutclinial)

    clf = MeanShift(bandwidth=bw)
    predicted = clf.fit_predict(pd_withoutclinial)
    cluster_lab = clf.labels_

    n_clusters = np.unique(cluster_lab).size
    print(n_clusters)
    palette = sns.hls_palette(n_clusters)
    sns.palplot(palette)

    colors = [palette[i] for i in predicted]
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=colors, s=10)
    plt.title('Mean Shift')
    plt.savefig('out\\'+cancer_type+'_TSNE.png')
    plt.clf()

    return  cluster_lab


def clinical(pd1 , path):
    clinical_data = pd.read_csv(path)
    print(clinical_data.head())
    pd1['name'] = clinical_data['submitter_id']
    pd1['status'] = clinical_data['status']
    pd1['days'] = clinical_data['days']
    pd1 = pd1.drop(pd1[pd1['days']==0].index)
    return pd1



# def inference(loader, model):
#     model.eval()
#     feature_vector = []
#     for step, x in enumerate(loader):
#         with torch.no_grad():
#             z, x_out = model.ae.forward(x)
#         z = z.detach()
#         feature_vector.extend(z.cpu().detach().numpy())
#     feature_vector = np.array(feature_vector)
#     print("Features shape {}".format(feature_vector.shape))
#     return feature_vector


def inference(DL, model):
    feature_vector = []
    for step, x in enumerate(DL):
        optimizer.zero_grad()
        x_i = (x + torch.normal(0, 1, size=(x.shape[0], x.shape[1]))).float().to(device)
        x_j = (x + torch.normal(0, 1, size=(x.shape[0], x.shape[1]))).float().to(device)
        z_i, z_j, h, h_index = model(x_i, x_j)
        optimizer.step()
        feature_vector.extend(h.cpu().detach().numpy())
    feature_vector = np.array(feature_vector)
    return  feature_vector


def train(DL,model):
    total_loss = 0
    feature_vector = []
    index = []
    for step, x in enumerate(DL):
        optimizer.zero_grad()
        x_i = (x + torch.normal(0, 1, size=(x.shape[0], x.shape[1]))).float().to(device)
        x_j = (x + torch.normal(0, 1, size=(x.shape[0], x.shape[1]))).float().to(device)
        z_i, z_j, h, h_index = model(x_i,x_j)
        batch = x.shape[0]
        loss = criterion_instance(z_i, z_j)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        index.extend(h_index.cpu().detach().numpy())
        feature_vector.extend(h.cpu().detach().numpy())
    epoch_loss = total_loss / len(DL)
    feature_vector = np.array(feature_vector)
    index = np.array(index)
    return epoch_loss, feature_vector, index


def draw_fig(list, name, epoch):
    x1 = range(0, epoch + 1)
    print(x1)
    y1 = list
    save_file = 'out\\' + name + '_Train_loss.png'
    plt.cla()
    plt.title('Train loss vs. epoch', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('Train loss', fontsize=20)
    plt.grid()
    plt.savefig(save_file)
    plt.clf()



if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    parser.add_argument("--cancer_type", '-c', type=str, default="UVM")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lambd', default=0.00005, type=float, metavar='L',
                        help='weight on off-diagonal terms')
    args = parser.parse_args()



    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    cancer = ['CESC']
    for cancer_type in cancer:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        logger = SummaryWriter(log_dir="./log")

        # load data
        dataloader_ge, ds = get_feature(cancer_type, args.batch_size, False)

        ae1 = ae.AE(ds)
        model1 = network.Network(ae1, args.feature_dim)

        loss_device = device
        criterion_instance = contrastive_loss.InstanceLoss(args).to(device)

        model1 = model1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        # optimizer / loss
        optimizer = torch.optim.Adam(model1.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        loss = []
        for epoch in range(args.start_epoch, args.epochs + 1):
            lr = optimizer.param_groups[0]["lr"]
            loss_epoch, f_vector, f_index = train(dataloader_ge, model1)
            loss.append(loss_epoch)
            logger.add_scalar("train loss", loss_epoch)
            if epoch % 100 == 0:
                print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch}")
        ge_vector = f_vector

        model_path = 'model\\' +cancer_type
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        save_model(model_path, model1, optimizer, args.epochs, "ge")
        draw_fig(loss, cancer_type, epoch)

        model = network.Network(ae1, args.feature_dim)
        model_fp = os.path.join(model_path, "model.tar")
        model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
        model.to(device)
        f_vector = inference(dataloader_ge, model)

        encoded_factors = normalize(ge_vector, axis=0, norm='max')

        fea1 = pd.DataFrame(data=encoded_factors, columns=map(lambda x: 'v' + str(x), range(encoded_factors.shape[1])))

        cl_path = 'test\\' + cancer_type +'\\'+cancer_type+ '_clinical.csv'
        fea1 = pd.read_csv('out\\' + cancer_type + '_features.csv', index_col=None)

        zero_percentage = fea1.eq(0).mean()

        zero_cols = zero_percentage.index[zero_percentage > args.zero_percentage]
        fea1.drop(zero_cols, axis=1, inplace=True)

        pd_wothclinical = (clinical(fea1, cl_path)).copy(deep=True)

        cox = cox_ph(pd_wothclinical)
        cox_summary = cox.summary
        cox_summary.sort_values(axis=0, by='p', ascending=True, inplace=True)
        index = cox_summary.index.tolist()
        index = index[0:2]
        print(index)
        m = fea1.loc[:, index]

        # MS
        cluster_lab = meanshift(m,cancer_type,args.bandwidth)

        fea1['cluster'] = cluster_lab.tolist()
        fea1.to_csv('out\\'+cancer_type+'_cluster.csv', index=False)
        col = ['status', 'days', 'cluster']
        km_pd = fea1.loc[:, col]
        km = KM(km_pd,cancer_type)


