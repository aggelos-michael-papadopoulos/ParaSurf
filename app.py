import gradio as gr
import torch
from ParaSurf.model import ParaSurf_model  # Adjust import based on your repo structure
from ParaSurf.train.protein import Protein_pred  # Import necessary modules

# Load your trained ParaSurf model
model = ParaSurf_model.ResNet3D_Transformer(in_channels=22, block=ParaSurf_model.DilatedBottleneck,
                                            num_blocks=[3, 4, 6, 3], num_classes=1)
model.load_state_dict(torch.load("/home/angepapa/PycharmProjects/DeepSurf2.0/train/Expanded_dataset_Paragraph/Entire_antibody_experiment/model_weights/epoch_3.pth"))
model.eval()

# Define a function for predictions
def predict_paratope(receptor_path):
    CFG_blind_pred = {
        'batch_size': 64,
        'Grid_size': 41,  # size of the voxel
        'feature_channels': 22,
        'add_atoms_radius_ff_features': True,
        'device': 'cuda',  # cuda or cpu
    }

    clean_dataset(os.path.dirname(receptor))

    # Process the protein
    prot = Protein_pred(receptor, save_path=os.path.dirname(receptor))

    # Locate the surfpoints file
    surf_file = os.path.join(prot.save_path, [i for i in os.listdir(prot.save_path) if 'surfpoints' in i][0])

    only_receptor_atoms_indexes = []
    with open(surf_file, 'r') as file:
        for atom_id, line in enumerate(file):
            parts = line.split()
            if parts[6] == 'A':
                only_receptor_atoms_indexes.append(atom_id)

    # Load the model
    nn = Network(model_weights, gridSize=CFG_blind_pred['Grid_size'],
                 feature_channels=CFG_blind_pred['feature_channels'], device=CFG_blind_pred['device'])

    # Make prediction
    lig_scores = nn.get_lig_scores(prot, batch_size=CFG_blind_pred['batch_size'], add_forcefields=True,
                                   add_atom_radius_features=CFG_blind_pred['add_atoms_radius_ff_features'])
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

# Gradio interface
iface = gr.Interface(fn=predict_paratope,
                     inputs=gr.Textbox(label="Receptor PDB Path"),
                     outputs="text",
                     title="ParaSurf Paratope Binding Site Prediction",
                     description="Upload the receptor PDB path and get the binding site predictions.")

iface.launch(share=True)
