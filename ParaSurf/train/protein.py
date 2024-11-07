import os, numpy as np
import shutil
# import pybel
from openbabel import pybel
from ParaSurf.train.utils import simplify_dms
from ParaSurf.utils.fix_surfpoints_format_issues import process_surfpoints_directory


class Protein_pred:
    def __init__(self, prot_file, save_path, seed=None, mesh_dense=0.3, atom_points_threshold=5, locate_only_surface=False):

        prot_id = prot_file.split('/')[-1].split('.')[0]
        self.save_path = os.path.join(save_path, prot_id)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.mol = next(pybel.readfile(prot_file.split('.')[-1], prot_file))
        self.atom_points_thresh = atom_points_threshold

        surfpoints_file = os.path.join(self.save_path, prot_id + '.surfpoints')

        # we have all the surfpoints ready from the preprocessing step
        if not os.path.exists(surfpoints_file):
            os.system('dms ' + prot_file + f' -d {mesh_dense} -n -o ' + surfpoints_file)      #set to 0.1 for fast results
            # fix any format issues
            print('\nfixing surfpoints format ...')
            process_surfpoints_directory(self.save_path)
            # raise Exception('probably DMS not installed')

        # locate surface: if we want the final coordinates to have the receptor atoms or we want just the surface atoms
        self.surf_points, self.surf_normals = simplify_dms(surfpoints_file, seed=seed,
                                                           locate_surface=locate_only_surface)

        self.heavy_atom_coords = np.array([atom.coords for atom in self.mol.atoms if atom.atomicnum > 1])

        self.binding_sites = []
        if prot_file.endswith('pdb'):
            with open(prot_file, 'r') as f:
                lines = f.readlines()
            self.heavy_atom_lines = [line for line in lines if line[:4] == 'ATOM' and line.split()[2][0] != 'H']
            if len(self.heavy_atom_lines) != len(self.heavy_atom_coords):
                ligand_in_pdb = len([line for line in lines if line.startswith('HETATM')]) > 0
                if ligand_in_pdb:
                    raise Exception('Ligand found in PDBfile. Please remove it to procede.')
                else:
                    raise Exception('Incosistency between Coords and PDBLines')
        else:
            raise IOError('Protein file should be .pdb')

    def _surfpoints_to_atoms(self, surfpoints):
        close_atoms = np.zeros(len(surfpoints), dtype=int)
        for p, surf_coord in enumerate(surfpoints):
            dist = np.sqrt(np.sum((self.heavy_atom_coords - surf_coord) ** 2, axis=1))
            close_atoms[p] = np.argmin(dist)

        return np.unique(close_atoms)

    def add_bsite(self, cluster):  # cluster -> tuple: (surf_points,scores)
        atom_idxs = self._surfpoints_to_atoms(cluster[0])
        self.binding_sites.append(Bsite(self.heavy_atom_coords, atom_idxs, cluster[1]))

    def sort_bsites(self):
        avg_scores = np.array([bsite.score for bsite in self.binding_sites])
        sorted_idxs = np.flip(np.argsort(avg_scores), axis=0)
        self.binding_sites = [self.binding_sites[idx] for idx in sorted_idxs]

    def write_bsites(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        centers = np.array([bsite.center for bsite in self.binding_sites])
        np.savetxt(os.path.join(self.save_path, 'centers.txt'), centers, delimiter=' ', fmt='%10.3f')

        pocket_count = 0
        for i, bsite in enumerate(self.binding_sites):
            outlines = [self.heavy_atom_lines[idx] for idx in bsite.atom_idxs]
            if len(outlines) > self.atom_points_thresh:
                pocket_count += 1
                with open(os.path.join(self.save_path, 'pocket' + str(pocket_count) + '.pdb'), 'w') as f:
                    f.writelines(outlines)




class Bsite:
    def __init__(self, mol_coords, atom_idxs, scores):
        self.coords = mol_coords[atom_idxs]
        self.center = np.average(self.coords, axis=0)
        self.score = np.average(scores)
        self.atom_idxs = atom_idxs

