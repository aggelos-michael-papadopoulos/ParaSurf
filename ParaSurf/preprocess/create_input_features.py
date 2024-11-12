from multiprocessing import Pool
from multiprocessing import Lock
import time, os
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from ParaSurf.utils.bsite_lib import readSurfpoints, readSurfpoints_with_residues, dist_point_from_lig
from ParaSurf.utils.features import KalasantyFeaturizer, KalasantyFeaturizer_with_force_fields
from scipy import sparse
from tqdm import tqdm
import warnings
from ParaSurf.utils.distance_coords import locate_receptor_binding_site_residues
from check_empty_features import remove_empty_features

# Ignore warnings
warnings.filterwarnings('ignore')


lock = Lock()  # Instantiate a Lock for thread safety.


def balanced_sampling(surf_file, protein_file, lig_files, cutoff=4.5):
    """
    Returns a subset of equal positive and negative samples from surface points in `surf_file`, with positive samples
    selected from residues close to the antigen.

    Parameters:
    - surf_file: Path to the file with protein surface points.
    - protein_file: Path to the protein structure file (e.g., PDB format).
    - lig_files: List of ligand (antigen) structure file paths.
    - cutoff: Distance cutoff in Ångstroms for defining binding residues (default is 4).

    Returns:
    - Balanced samples including features and labels for the selected surface points.
    """
    all_lig_coords = []
    for lig_file in lig_files:
        with lock:  # Locks the parser to ensure thread safety.
            lig = parser.get_structure('antigen', lig_file)
        lig_coords = np.array([atom.get_coord() for atom in lig.get_atoms()])
        all_lig_coords.append(lig_coords)

    points, normals = readSurfpoints(surf_file)  # modified by me
    # create the residue groups for the whole protein
    all_rec_residues = readSurfpoints_with_residues(surf_file)


    # find the bind site residues
    bind_site_rec_residues = locate_receptor_binding_site_residues(protein_file, lig_file, distance_cutoff=cutoff)

    # gather all distances
    # Create an array to store the minimum distance of each point to any ligand atom
    dist_from_lig = np.full(len(points), np.inf)
    near_lig = np.zeros(len(points), dtype=bool)

    # Update distances only for points in bind site residues
    bind_site_indices = [item for i in bind_site_rec_residues for item in all_rec_residues[i]['idx']]
    bind_site_indices_set = set(bind_site_indices)  # Convert to set for fast lookup

    # Loop through ligand coordinates and update distances for binding site points
    # IMPORTANT Step because if a residue belongs to the binding site that DOES NOT mean that all the atom of this
    # residue belongs the binding site (<6 armstrong to the ligand). So here we check from the binding site residues which
    # atoms actually bind (<6 armstrong to the ligand)
    for lig_coords in all_lig_coords:
        for i, p in enumerate(points):
            if i in bind_site_indices_set:
                dist = dist_point_from_lig(p, lig_coords)  # Adjust this function if necessary
                if dist < dist_from_lig[i]:
                    dist_from_lig[i] = dist
                    near_lig[i] = dist < cutoff

    # Filter positive indices to include only those near a ligand
    pos_idxs = np.array([idx for idx in bind_site_indices if near_lig[idx]])

    # If there are more positive indices than allowed, select the best ones based on the distance
    if len(pos_idxs) > maxPosSamples:
        pos_idxs = pos_idxs[np.argsort(dist_from_lig[pos_idxs])[:maxPosSamples]]

    # Select the negative samples
    all_neg_samples = [idx for idx, i in enumerate(points) if idx not in pos_idxs]

    # Calculate number of negative samples to match the number of positive samples
    num_neg_samples = min(len(all_neg_samples), len(pos_idxs))

    neg_idxs = np.array(all_neg_samples)
    if len(neg_idxs) > num_neg_samples:
        neg_downsampled = np.random.choice(neg_idxs, num_neg_samples, replace=False)
    else:
        neg_downsampled = neg_idxs

    # Concatenate positive and negative indices
    sample_idxs = np.concatenate((pos_idxs, neg_downsampled))

    # Shuffle the indices to ensure randomness
    np.random.shuffle(sample_idxs)

    # create the sample labels
    # Convert pos_idxs to a set for faster membership testing
    pos_set = set(pos_idxs)

    # Use list comprehension to create labels
    sample_labels = [i in pos_set for i in sample_idxs]
    if feat_type == 'kalasanty':
        featurizer = KalasantyFeaturizer(protein_file, protonate, gridSize, voxelSize, use_protrusion, protr_radius)
    elif feat_type == 'kalasanty_with_force_fields':
        featurizer = KalasantyFeaturizer_with_force_fields(protein_file, protonate, gridSize, voxelSize, use_protrusion, protr_radius,
                                                           add_atom_radius_features=add_atoms_radius_ff_features)

    feature_vector_length = featurizer.channels.shape[1]
    with open(feature_vector_length_tmp_path, 'w') as file:
        file.write(str(feature_vector_length))

    for i, sample in enumerate(sample_idxs):
        features = featurizer.grid_feats(points[sample], normals[sample], rotate_grid)
        if np.count_nonzero(features) == 0:
            print('Zero features', protein_file.rsplit('/', 1)[1][:-4], i, points[sample], normals[sample])

        yield features, sample_labels[i], points[sample], normals[sample]


def samples_per_prot(prot):
    """
    Generates and saves balanced surface point samples for a given protein.

    Parameters:
    - prot: Protein identifier for which features are being generated.

    Saves each sample as a sparse matrix or NumPy array in `feats_path`.
    """
    prot_path = os.path.join(feats_path, prot)

    # Check if directory exists and has files, if not create it
    if not os.path.exists(prot_path):
        os.makedirs(prot_path)
    elif os.listdir(prot_path):
        return

    surf_file = os.path.join(surf_path, f"{prot}.surfpoints")
    protein_file = os.path.join(pdbs_path, f"{prot}.pdb")
    if not os.path.exists(protein_file):
        protein_file = os.path.join(pdbs_path, f"{prot}.mol2")

    receptor_id = prot_path.split('_')[-1]
    antigen_prefix = prot.split('_')[0]

    # Using set for faster membership checks
    files_set = set(os.listdir(pdbs_path))
    lig_files = [os.path.join(pdbs_path, f) for f in files_set if f"{antigen_prefix}_antigen_{receptor_id}" in f]

    try:
        cnt = 0
        for features, y, point, normal in balanced_sampling(surf_file, protein_file, lig_files, cutoff=cutoff):
            samples_file_name = os.path.join(prot_path, f"sample{cnt}_{int(y)}")

            if feat_type == 'deepsite':
                with open(f"{samples_file_name}.npy", 'w') as f:
                    np.save(f, (point, normal, features.astype(np.float16)))
            elif feat_type == 'kalasanty' or feat_type == 'kalasanty_with_force_fields':
                sparse_mat = sparse.coo_matrix(features.flatten())
                sparse.save_npz(f"{samples_file_name}.npz", sparse_mat)

            cnt += 1

        print(f'Saved "{cnt}" samples for "{prot}".')

    except Exception as e:
        print(f'Exception occurred while processing "{prot}". Error message: "{e}".')


seed = 10  # random seed
num_cores = 6  # Set this to the number of cores you wish to use
maxPosSamples = 800  # maximum number of positive samples per protein
gridSize = 41  # size of grid (16x16x16)
voxelSize = 1  # size of voxel, e.g. 1 angstrom, if 2A we lose details, so leave it to 1
cutoff = 4.5     # cutoff threshold in Armstrong's 6 for general PPIs, 4.5 for antibody antigen databases
feature_vector_length_tmp_path = '/tmp/feature_vector_length.txt'
# feat_type = 'kalasanty'  # select featurizer
feat_type = 'kalasanty_with_force_fields'
add_atoms_radius_ff_features = True    # If you want to add the atom radius features that correspond to the force fields
rotate_grid = True  # whether to rotate the grid (ignore)
use_protrusion = False  # ignore
protr_radius = 10  # ignore
protonate = True  # if protein pdbs are not protonated (do not have Hydrogens) set it to True

user = os.getenv('USER')
pdbs_path = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/pdbs/example/TRAIN'  # input folder with protein pdbs for training
surf_path = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/surfpoints/example/TRAIN'  # input folder with surface points for training
feats_path = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/feats/example'  # training features folder


if not os.path.exists(feats_path):
    os.makedirs(feats_path)

np.random.seed(seed)

all_proteins = [f.rsplit('.', 1)[0] for f in os.listdir(surf_path)]
#
# in case the procedure stacks use the 3 lines below
completed = [f.rsplit('.', 1)[0] for f in os.listdir(feats_path)]
all_proteins = [i for i in all_proteins if i not in completed]
print(len(all_proteins))


parser = PDBParser(PERMISSIVE=1)  # PERMISSIVE=1 allowing more flexibility in handling non-standard or problematic entries in PDB files during parsing.

start = time.time()
with Pool(num_cores) as pool:  # Use a specified number of CPU cores
    list(tqdm(pool.imap(samples_per_prot, all_proteins), total=len(all_proteins)))
print(f'Total preprocess time: {(time.time() - start)/60} mins')

###################################################################################
# Instead of using Pool and imap, iterate through all_proteins with a for loop for easy debugging
# for prot in all_proteins:
#     try:
#         samples_per_prot(prot)
#     except Exception as e:
#         print(f'Error processing protein {prot}: {e}')
#     break


# the last number at the out_path will be the total number of the feature vector
if os.path.exists(feature_vector_length_tmp_path):
    with open(feature_vector_length_tmp_path, 'r') as file:
        feature_vector_length = int(file.read().strip())
        feats_path_new = f'{feats_path}_{feature_vector_length}'
    os.rename(feats_path, feats_path_new)
    os.remove(feature_vector_length_tmp_path)


# remove empty features if found
remove_empty_features(feats_folder=feats_path_new, pdbs_path=pdbs_path, surf_path=surf_path)