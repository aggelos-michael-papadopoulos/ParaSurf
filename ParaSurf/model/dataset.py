import numpy as np
import random, os
from scipy import sparse
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import h5py


class dataset(Dataset):
    def __init__(self, train_file, batch_size, data_path, grid_size, training, feature_vector_lentgh, feature_names=['deepsite']):

        super(dataset, self).__init__()
        self.training = training
        self.feature_vector_lentgh = feature_vector_lentgh
        #  in testing mode training file is not read
        if self.training:
            with open(train_file) as f:
                self.train_lines = f.readlines()
            random.shuffle(self.train_lines)

        else:
            self.train_lines = []



        random.shuffle(self.train_lines)

        self.pointer_tr = 0
        self.pointer_val = 0

        self.batch_size = batch_size
        self.data_path = data_path
        self.grid_size = grid_size
        self.feature_names = feature_names
        #        if added_features is None:         # resolved outside
        self.nAtomTypes = 0

        self.nfeats = {
            'deepsite': 8,
            'kalasanty': feature_vector_lentgh,
            'kalasanty_with_force_fields': feature_vector_lentgh,
            'kalasanty_norotgrid': 18,
            'spat_protr': 1,
            'spat_protr_norotgrid': 1
        }

        for name in feature_names:
            self.nAtomTypes += self.nfeats[name]

    def __len__(self):
        if self.training:
            return len(self.train_lines)


    def __getitem__(self, index):
        if self.training:
            samples = self.train_lines


        label, sample_file = samples[index].split()
        label = int(label)
        base_name, prot, sample = sample_file.split('/')

        feats = np.zeros((self.grid_size, self.grid_size, self.grid_size, self.nAtomTypes))
        feat_cnt = 0

        for name in self.feature_names:
            if 'deepsite' == name:
                data = np.load(os.path.join(self.data_path, base_name + '_' + name, prot, sample), allow_pickle=True)
            elif 'kalasanty' == name:
                data = sparse.load_npz(os.path.join(self.data_path, base_name, prot, sample[:-1] + 'z'))
                data = np.reshape(np.array(data.todense()), (self.grid_size, self.grid_size, self.grid_size, self.nfeats['kalasanty']))
            elif 'kalasanty_with_force_fields' == name:
                data = sparse.load_npz(os.path.join(self.data_path, base_name, prot, sample[:-1] + 'z'))
                data = np.reshape(np.array(data.todense()), (self.grid_size, self.grid_size, self.grid_size, self.nfeats['kalasanty_with_force_fields']))
            elif 'spat_protr' in name:
                data = np.load(os.path.join(self.data_path, base_name + '_' + name, prot, sample), allow_pickle=True)
            else:
                print('unknown feat')

            if len(data) == 3:
                data = data[2]  # prosoxh, mono sto scPDB, gia thn wra (sto kalasanty den exw points, normals)

            feats[:, :, :, feat_cnt:feat_cnt + self.nfeats[name]] = data
            feat_cnt += self.nfeats[name]

        if feat_cnt != self.nAtomTypes:
            print('error !')

        # Modified code with explicit strides: Because pytorch does not handle negative samples
        if self.training:
            rot_axis = random.randint(1, 3)
            feats_copy = feats.copy()
            if rot_axis == 1:
                feats_copy = np.rot90(feats_copy, random.randint(0, 3), axes=(0, 1))
            elif rot_axis == 2:
                feats_copy = np.rot90(feats_copy, random.randint(0, 3), axes=(0, 2))
            elif rot_axis == 3:
                feats_copy = np.rot90(feats_copy, random.randint(0, 3), axes=(1, 2))
            feats = np.ascontiguousarray(feats_copy)

        if np.isnan(np.sum(feats)):
            print('nan input')

        return feats, label

