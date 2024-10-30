from ParaSurf.utils import bio_data_featurizer
from ParaSurf.train.utils import rotation
import numpy as np


class KalasantyFeaturizer:
    def __init__(self, gridSize, voxelSize):
        grid_limit = (gridSize / 2 - 0.5) * voxelSize
        grid_radius = grid_limit * np.sqrt(3)
        self.neigh_radius = 4 + grid_radius  # 4 > 2*R_vdw
        # self.neigh_radius = 2*grid_radius  # 4 > 2*R_vdw
        self.featurizer = bio_data_featurizer.Featurizer(save_molecule_codes=False)
        self.grid_resolution = voxelSize
        self.max_dist = (gridSize - 1) * voxelSize / 2

    def get_channels(self, mol, add_forcefields, add_atom_radius_features=False):
        if not add_forcefields:
            self.coords, self.channels = self.featurizer.get_features(mol)  # returns only heavy atoms
        else:
            self.coords, self.channels = self.featurizer.get_features_with_force_fields(mol, add_atom_radius=add_atom_radius_features)  # returns only heavy atoms


    def get_channels_with_forcefields(self, mol):
        self.coords, self.channels = self.featurizer.get_features_with_force_fields(mol, add_atom_radius=True)  # returns only heavy atoms

    def grid_feats(self, point, normal, mol_coords):
        neigh_atoms = np.sqrt(np.sum((mol_coords - point) ** 2, axis=1)) < self.neigh_radius
        Q = rotation(normal)
        Q_inv = np.linalg.inv(Q)
        transf_coords = np.transpose(mol_coords[neigh_atoms] - point)
        rotated_mol_coords = np.matmul(Q_inv, transf_coords)
        features = \
        bio_data_featurizer.make_grid(np.transpose(rotated_mol_coords), self.channels[neigh_atoms], self.grid_resolution,
                             self.max_dist)[0]

        return features

