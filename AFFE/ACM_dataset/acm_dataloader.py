import numpy as np
import scipy
import pickle
import torch
import dgl
import torch.nn.functional as F
import torch as th
import scipy.sparse as sp


def load_ACM_data(prefix=r'C:\Users\Yufei Zhao\Desktop\Z_IDEA_NEW_0408\ACM_dataset'):

    features_0 = scipy.sparse.load_npz(prefix + '/features_0_p.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1_a.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2_s.npz').toarray()

    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)

    features = [features_0, features_1, features_2]

    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)

    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    num_classes = 3

    nei_a = np.load(prefix + "/nei_a.npy", allow_pickle=True)
    nei_s = np.load(prefix + "/nei_s.npy", allow_pickle=True)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_s = [th.LongTensor(i) for i in nei_s]
    NS = [nei_a, nei_s]


    region11 = scipy.sparse.load_npz(prefix + '/region11.npz').toarray()
    region12 = scipy.sparse.load_npz(prefix + '/region12.npz').toarray()

    region21 = scipy.sparse.load_npz(prefix + '/region21.npz').toarray()
    region21 = (region21 > 0)*1

    region22 = scipy.sparse.load_npz(prefix + '/region22.npz').toarray()
    region22 = (region22 > 0) * 1



    region23 = sp.load_npz(prefix+'/region23.npz').toarray()
    region23 = (region23>0)*1
    region11 = F.normalize(torch.from_numpy(region11).type(torch.FloatTensor), dim=1, p=2)
    region12 = F.normalize(torch.from_numpy(region12).type(torch.FloatTensor), dim=1, p=2)
    region23 = F.normalize(torch.from_numpy(region23).type(torch.FloatTensor),dim=1 ,p=2)
    region21 = F.normalize(torch.from_numpy(region21).type(torch.FloatTensor), dim=1, p=2)
    region22 = F.normalize(torch.from_numpy(region22).type(torch.FloatTensor), dim=1, p=2)

    ADJ = [[region11, region12], [region21, region23, region22]]
    return ADJ, NS, features, labels, num_classes, train_idx, val_idx, test_idx
