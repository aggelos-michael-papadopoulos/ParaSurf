from ParaSurf.utils import bio_data_featurizer
# import pybel
from openbabel import pybel
from .bsite_lib import Grid
from .rotation import rotation_quaternion
import numpy as np
from scipy import ndimage


class Featurizer:
    def __init__(self, gridSize, voxelSize, use_protrusion=False, protr_radius=10, speedup=False, add_atom_radius_features=False):
        self.grid = Grid(gridSize, voxelSize)
        self.protrusion = use_protrusion
        self.protr_radius = protr_radius
        self.speedup = speedup
        self.add_atom_radius_features = add_atom_radius_features
        if use_protrusion:
            self.neigh_radius = protr_radius + self.grid.radius
        else:
            self.neigh_radius = 4 + self.grid.radius  # 4 > 2*R_vdw
            # self.neigh_radius = 2*self.grid.radius  # 4 > 2*R_vdw

    def _get_protrusion(self, centers, coords):
        if self.speedup:
            sampled_centers = np.resize(centers, (self.grid.gridSize, self.grid.gridSize, self.grid.gridSize, 3))
            sampled_centers = sampled_centers[0:self.grid.gridSize:2, 0:self.grid.gridSize:2, 0:self.grid.gridSize:2, :]
            sampled_centers = np.resize(sampled_centers, (
            sampled_centers.shape[0] * sampled_centers.shape[1] * sampled_centers.shape[2], 3))
        else:
            sampled_centers = centers
        close = np.zeros((len(sampled_centers), len(coords)), dtype=bool)
        for c, coord in enumerate(coords):
            close[:, c] = np.sqrt(np.sum((sampled_centers - coord) ** 2, axis=1)) < self.protr_radius
        protr = np.count_nonzero(close, axis=1)
        if self.speedup:
            protr.resize(self.grid.gridSize / 2, self.grid.gridSize / 2, self.grid.gridSize / 2)
            protr = ndimage.zoom(protr, (2, 2, 2), order=3, mode='nearest')
            protr = protr.flatten()
        return np.expand_dims(protr, axis=-1) / 200.0  # normalization


class KalasantyFeaturizer(Featurizer):
    def __init__(self, mol_file, protonate, gridSize, voxelSize, use_protrusion=False, protr_radius=10, speedup=False,
                 add_atom_radius_features=False):
        Featurizer.__init__(self, gridSize, voxelSize, use_protrusion, protr_radius, speedup, add_atom_radius_features)
        featurizer = bio_data_featurizer.Featurizer(save_molecule_codes=False)
        # mol = pybel.readfile(mol_file.rsplit('.', 1)[-1], mol_file).next()
        for mol in pybel.readfile(mol_file.rsplit('.', 1)[-1], mol_file):  # changed by me
            if protonate:
                mol.addh()
            # todo add electrostatic features
            self.coords, self.channels = featurizer.get_features(mol)
            self.grid_resolution = voxelSize
            self.max_dist = (gridSize - 1) * voxelSize / 2

    def grid_feats(self, point, normal, rotate_grid):
        neigh_atoms = np.sqrt(np.sum((self.coords - point) ** 2, axis=1)) < self.neigh_radius
        if rotate_grid:
            Q = rotation_quaternion(normal)
            Q_inv = np.linalg.inv(Q)
            transf_coords = np.transpose(self.coords[neigh_atoms] - point)
            rotated_mol_coords = np.matmul(Q_inv, transf_coords)
            features = \
            bio_data_featurizer.make_grid(np.transpose(rotated_mol_coords), self.channels[neigh_atoms], self.grid_resolution,
                                 self.max_dist)[0]      # here we go to (41,41,41,18)
        else:
            features = \
            bio_data_featurizer.make_grid(self.coords[neigh_atoms] - point, self.channels[neigh_atoms], self.grid_resolution,
                                 self.max_dist)[0]

        if self.protrusion:
            centers = self.grid.make_grid(point, normal, rotate_grid)
            protr = self._get_protrusion(centers, self.coords[neigh_atoms])
            features = np.append(features,
                                 np.resize(protr, (self.grid.gridSize, self.grid.gridSize, self.grid.gridSize, 1)),
                                 axis=-1)

        return features


class KalasantyFeaturizer_with_force_fields(Featurizer):
    def __init__(self, mol_file, protonate, gridSize, voxelSize, use_protrusion=False, protr_radius=10, speedup=False,
                 add_atom_radius_features=False):
        Featurizer.__init__(self, gridSize, voxelSize, use_protrusion, protr_radius, speedup, add_atom_radius_features)
        featurizer = bio_data_featurizer.Featurizer(save_molecule_codes=False, add_force_fields=True)
        for mol in pybel.readfile(mol_file.rsplit('.', 1)[-1], mol_file):
            if protonate:
                mol.addh()
            # todo add electrostatic features
            self.coords, self.channels = featurizer.get_features_with_force_fields(mol, add_atom_radius=add_atom_radius_features)
            self.grid_resolution = voxelSize
            self.max_dist = (gridSize - 1) * voxelSize / 2

    def grid_feats(self, point, normal, rotate_grid):
        neigh_atoms = np.sqrt(np.sum((self.coords - point) ** 2, axis=1)) < self.neigh_radius
        if rotate_grid:
            Q = rotation_quaternion(normal)
            Q_inv = np.linalg.inv(Q)
            transf_coords = np.transpose(self.coords[neigh_atoms] - point)
            rotated_mol_coords = np.matmul(Q_inv, transf_coords)
            features = \
            bio_data_featurizer.make_grid(np.transpose(rotated_mol_coords), self.channels[neigh_atoms], self.grid_resolution,
                                 self.max_dist)[0]      # here we go to (41,41,41,18)
        else:
            features = \
            bio_data_featurizer.make_grid(self.coords[neigh_atoms] - point, self.channels[neigh_atoms], self.grid_resolution,
                                 self.max_dist)[0]

        if self.protrusion:
            centers = self.grid.make_grid(point, normal, rotate_grid)
            protr = self._get_protrusion(centers, self.coords[neigh_atoms])
            features = np.append(features,
                                 np.resize(protr, (self.grid.gridSize, self.grid.gridSize, self.grid.gridSize, 1)),
                                 axis=-1)


        return features
