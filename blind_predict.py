import os
import numpy as np
import argparse
from ParaSurf.preprocess.clean_dataset import clean_dataset
from ParaSurf.train.protein import Protein_pred
from ParaSurf.train.network import Network
from ParaSurf.train.bsite_extraction import Bsite_extractor
from ParaSurf.train.utils import receptor_info, write_residue_prediction_pdb, write_atom_prediction_pdb
import warnings

warnings.filterwarnings('ignore')

# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description="Run blind prediction using ParaSurf.")
parser.add_argument('--receptor', type=str, required=True, help="Path to the receptor (antibody or paratope) PDB file.")
parser.add_argument('--model_weights', type=str, required=True, help="Path to the model weights file.")
parser.add_argument('--mesh_dense', type=float, default=0.3, help="Density for mesh generation (0.1 to 1.0).")
args = parser.parse_args()

if not (0.1 <= args.mesh_dense <= 1.0):
    raise ValueError("mesh_dense must be between 0.1 and 1")

CFG_blind_pred = {
    'batch_size': 64,
    'Grid_size': 41,  # size of the voxel
    'feature_channels': 22,
    'add_atoms_radius_ff_features': True,
    'device': 'cuda',  # cuda or cpu
}

# Use command-line arguments for receptor, model_weights paths, and mesh_dense
receptor = args.receptor
model_weights = args.model_weights
mesh_dense = args.mesh_dense
results_save_path = f"{os.path.dirname(receptor)}/{os.path.basename(receptor).split('.')[0]}"

# Clean the dataset
clean_dataset(os.path.dirname(receptor))

# Process the protein with specified mesh_dense
prot = Protein_pred(receptor, save_path=os.path.dirname(receptor), mesh_dense=mesh_dense)

# Locate the surfpoints file
surf_file = os.path.join(prot.save_path, [i for i in os.listdir(prot.save_path) if 'surfpoints' in i][0])

only_receptor_atoms_indexes = []
with open(surf_file, 'r') as file:
    for atom_id, line in enumerate(file):
        parts = line.split()
        if parts[6] == 'A':
            only_receptor_atoms_indexes.append(atom_id)

# Load the model
nn = Network(model_weights, gridSize=CFG_blind_pred['Grid_size'], feature_channels=CFG_blind_pred['feature_channels'], device=CFG_blind_pred['device'])

# Make prediction
lig_scores = nn.get_lig_scores(prot, batch_size=CFG_blind_pred['batch_size'], add_forcefields=True, add_atom_radius_features=CFG_blind_pred['add_atoms_radius_ff_features'])
lig_scores_only_receptor_atoms = np.array([lig_scores[i] for i in only_receptor_atoms_indexes])

# Extract residues and residues_best from receptor_info
residues, residues_best = receptor_info(receptor, lig_scores_only_receptor_atoms)

# Write residue-level prediction PDB: Residue-level prediction
write_residue_prediction_pdb(receptor, results_save_path, residues_best)

# Write per-atom prediction PDB: Per atom level prediction
write_atom_prediction_pdb(receptor, results_save_path, lig_scores_only_receptor_atoms)

# Extract Binding Sites: Pocket.pdb file
extractor = Bsite_extractor()
extractor.extract_bsites(prot, lig_scores)

# Remove the surfpoints file
os.remove(surf_file)
