import os
import numpy as np
import argparse
import warnings
from tqdm import tqdm

from ParaSurf.preprocess.clean_dataset import clean_dataset
from ParaSurf.train.protein import Protein_pred
from ParaSurf.train.network import Network
from ParaSurf.train.bsite_extraction import Bsite_extractor
from ParaSurf.train.utils import (
    receptor_info,
    write_residue_prediction_pdb,
    write_atom_prediction_pdb,
    antibody_input_recognition
)

warnings.filterwarnings('ignore')


def process_antibody(receptor_path, model_weights, mesh_dense, cfg):
    """
    Run the full ParaSurf blind-prediction pipeline on one antibody PDB.
    Outputs are written into a subfolder named after the PDB prefix.
    """
    prefix = os.path.basename(receptor_path).replace('_antibody.pdb', '') # or antibody
    print(prefix, receptor_path)
    out_dir = os.path.join(os.path.dirname(receptor_path), f'{prefix}_antibody')
    # ✅ Skip if predictions folder already exists (and has something inside)
    if os.path.isdir(out_dir) and any(os.scandir(out_dir)):
        print(f"[{prefix}] Skipping: already predicted ({out_dir})")
        return

    os.makedirs(out_dir, exist_ok=True)

    # 1) Validate input
    ok, msg = antibody_input_recognition(receptor_path)
    if not ok:
        print(f"[{prefix}] Skipped: {msg}")
        return

    # 2) Build surface
    prot = Protein_pred(
        receptor_path,
        save_path=os.path.dirname(receptor_path),
        mesh_dense=mesh_dense
    )

    # 3) Find the generated surfpoints file
    surf_file = os.path.join(
        prot.save_path,
        next(f for f in os.listdir(prot.save_path) if 'surfpoints' in f)
    )

    # 4) Read surfpoints and pick only chain ‘A’ points
    only_receptor_idxs = []
    with open(surf_file, 'r') as sf:
        for idx, line in enumerate(sf):
            if line.split()[6] == 'A':
                only_receptor_idxs.append(idx)

    # 5) Load model & predict
    nn = Network(
        model_weights,
        gridSize=cfg['Grid_size'],
        feature_channels=cfg['feature_channels'],
        device=cfg['device']
    )
    lig_scores = nn.get_lig_scores(
        prot,
        batch_size=cfg['batch_size'],
        add_forcefields=True,
        add_atom_radius_features=cfg['add_atoms_radius_ff_features']
    )

    # 6) Filter scores for chain A only
    lig_scores_A = np.array([lig_scores[i] for i in only_receptor_idxs])

    # 7) Map to residues
    residues, residues_best = receptor_info(receptor_path, lig_scores_A)

    # 8) Write outputs
    write_residue_prediction_pdb(receptor_path, out_dir, residues_best)
    write_atom_prediction_pdb(receptor_path, out_dir, lig_scores_A)

    # 9) Extract pockets
    extractor = Bsite_extractor()
    extractor.extract_bsites(prot, lig_scores)

    # 10) Clean up
    os.remove(surf_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch blind-prediction over a folder of antibody–antigen PDB pairs"
    )
    parser.add_argument(
        '--input_dir', '-i',
        required=True,
        help="Directory containing `_antibody.pdb` and `_antigen.pdb` files."
    )
    parser.add_argument(
        '--model_weights', '-w',
        required=True,
        help="Path to your trained ParaSurf model weights."
    )
    parser.add_argument(
        '--mesh_dense', '-d',
        type=float,
        default=0.3,
        help="Surface-mesh density (0.1–1.0). Lower is faster."
    )
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        help="Torch device to use."
    )
    args = parser.parse_args()

    if not (0.1 <= args.mesh_dense <= 1.0):
        raise ValueError("`mesh_dense` must be between 0.1 and 1.0")

    CFG = {
        'batch_size': 64,
        'Grid_size': 41,
        'feature_channels': 22,
        'add_atoms_radius_ff_features': True,
        'device': args.device,
    }

    # Clean entire folder once
    clean_dataset(args.input_dir)

    # Gather all antibody PDBs
    antibody_files = sorted(
        f for f in os.listdir(args.input_dir)
        if f.endswith('_antibody.pdb')
    )
    if not antibody_files:
        print("No `_antibody.pdb` files found in", args.input_dir)
        exit(1)

    # Process with progress bar
    for ab_file in tqdm(antibody_files, desc="Processing antibodies"):
        receptor_path = os.path.join(args.input_dir, ab_file)
        process_antibody(
            receptor_path,
            model_weights=args.model_weights,
            mesh_dense=args.mesh_dense,
            cfg=CFG
        )
