import warnings, os
from Bio.PDB import PDBParser
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, precision_recall_curve, \
    confusion_matrix, matthews_corrcoef


def mol2_reader(mol_file):  # does not handle H2
    if mol_file[-4:] != 'mol2':
        raise Exception("File's extension is not .mol2")

    with open(mol_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if '@<TRIPOS>ATOM' in line:
            first_atom_idx = i + 1
        if '@<TRIPOS>BOND' in line:
            last_atom_idx = i - 1

    return lines[first_atom_idx:last_atom_idx + 1]


# maybe change to read_surfpoints_new ??
def readSurfPoints(surf_file):
    with open(surf_file, 'r') as f:
        lines = f.readlines()

    lines = [l for l in lines if len(l.split()) > 7]

    if len(lines) > 100000:
        warnings.warn('{} has too many points'.format(surf_file))
        return
    if len(lines) == 0:
        warnings.warn('{} is empty'.format(surf_file))
        return

    coords = np.zeros((len(lines), 3))
    normals = np.zeros((len(lines), 3))
    for i, l in enumerate(lines):
        parts = l.split()

        try:
            coords[i, 0] = float(parts[3])
            coords[i, 1] = float(parts[4])
            coords[i, 2] = float(parts[5])
            normals[i, 0] = float(parts[8])
            normals[i, 1] = float(parts[9])
            normals[i, 2] = float(parts[10])
        except:
            coords[i, 0] = float(parts[2][-8:])
            coords[i, 1] = float(parts[3])
            coords[i, 2] = float(parts[4])
            normals[i, 0] = float(parts[7])
            normals[i, 1] = float(parts[8])
            normals[i, 2] = float(parts[9])

    return coords, normals


def readSurfPoints_with_receptor_atoms(surf_file):
    with open(surf_file, 'r') as f:
        lines = f.readlines()

    # lines = [l for l in lines if len(l.split()) > 7]
    lines = [l for l in lines]
    # if len(lines) > 100000:
    #     warnings.warn('{} has too many points'.format(surf_file))
    #     return
    if len(lines) == 0:
        warnings.warn('{} is empty'.format(surf_file))
        return

    coords = np.zeros((len(lines), 3))
    normals = np.zeros((len(lines), 3))

    # First, ensure each line has at least 11 parts by filling with zeros
    for i in range(len(lines)):
        parts = lines[i].split()
        while len(parts) < 11:
            # Fill with '0' initially
            parts.append('0')
        lines[i] = ' '.join(parts)

    # Modify lines according to the specified rules
    for i in range(len(lines)):
        parts = lines[i].split()
        # Check if there are zeros that need to be replaced
        if '0' in parts:
            if i > 0:  # Use previous line if not the first line
                prev_parts = lines[i - 1].split()
                parts = [prev_parts[j] if part == '0' else part for j, part in enumerate(parts)]
            elif i < len(lines) - 1:  # Use next line if not the last line
                next_parts = lines[i + 1].split()
                parts = [next_parts[j] if part == '0' else part for j, part in enumerate(parts)]
        lines[i] = ' '.join(parts)

        try:
            coords[i, 0] = float(parts[3])
            coords[i, 1] = float(parts[4])
            coords[i, 2] = float(parts[5])
            normals[i, 0] = float(parts[8])
            normals[i, 1] = float(parts[9])
            normals[i, 2] = float(parts[10])
        except:
            coords[i, 0] = float(parts[2][-8:])
            coords[i, 1] = float(parts[3])
            coords[i, 2] = float(parts[4])
            normals[i, 0] = float(parts[7])
            normals[i, 1] = float(parts[8])
            normals[i, 2] = float(parts[9])

    return coords, normals


def simplify_dms(init_surf_file, seed=None, locate_surface=True):
    # Here we decide if we want the final coordinates to have the receptor atoms or we want just
    # the surface atoms
    if locate_surface:
        coords, normals = readSurfPoints(init_surf_file)
    else:
        coords, normals = readSurfPoints_with_receptor_atoms(init_surf_file)  # to also get the receptor points

        return coords, normals

    nCl =  len(coords)

    kmeans = KMeans(n_clusters=nCl, max_iter=300, n_init=1, random_state=seed).fit(coords)
    point_labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    cluster_idx, freq = np.unique(point_labels, return_counts=True)
    if len(cluster_idx) != nCl:
        raise Exception('Number of created clusters should be equal to nCl')

    idxs = []
    for cl in cluster_idx:
        cluster_points_idxs = np.where(point_labels == cl)[0]
        closest_idx_to_center = np.argmin([euclidean(centers[cl], coords[idx]) for idx in cluster_points_idxs])
        idxs.append(cluster_points_idxs[closest_idx_to_center])

    return coords[idxs], normals[idxs]


def rotation(n):
    if n[0] == 0.0 and n[1] == 0.0:
        if n[2] == 1.0:
            return np.identity(3)
        elif n[2] == -1.0:
            Q = np.identity(3)
            Q[0, 0] = -1
            return Q
        else:
            print('not possible')

    rx = -n[1] / np.sqrt(n[0] * n[0] + n[1] * n[1])
    ry = n[0] / np.sqrt(n[0] * n[0] + n[1] * n[1])
    rz = 0
    th = np.arccos(n[2])

    q0 = np.cos(th / 2)
    q1 = np.sin(th / 2) * rx
    q2 = np.sin(th / 2) * ry
    q3 = np.sin(th / 2) * rz

    Q = np.zeros((3, 3))
    Q[0, 0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    Q[0, 1] = 2 * (q1 * q2 - q0 * q3)
    Q[0, 2] = 2 * (q1 * q3 + q0 * q2)
    Q[1, 0] = 2 * (q1 * q2 + q0 * q3)
    Q[1, 1] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
    Q[1, 2] = 2 * (q3 * q2 - q0 * q1)
    Q[2, 0] = 2 * (q1 * q3 - q0 * q2)
    Q[2, 1] = 2 * (q3 * q2 + q0 * q1)
    Q[2, 2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3

    return Q


def TP_TN_FP_FN_visualization2pdb(gt_binding_site_coordinates, lig_scores, to_save_path, gt_indexes):
    '''
    Create dummy PDB files to visualize the results (TP, TN, FP, FN) on the receptor PDB file
    '''
    threshold = 0.5

    # Initialize lists
    TP_coords = []
    FP_coords = []
    TN_coords = []
    FN_coords = []

    for i, score in enumerate(lig_scores):
        # If the atom is a true binding site
        if i in gt_indexes:
            if score > threshold:
                TP_coords.append(gt_binding_site_coordinates[i])
            else:
                FN_coords.append(gt_binding_site_coordinates[i])
        # If the atom is not a binding site
        else:
            if score > threshold:
                FP_coords.append(gt_binding_site_coordinates[i])
            else:
                TN_coords.append(gt_binding_site_coordinates[i])

    def generate_pdb_file(coordinates, file_name):
        """Generate a dummy PDB file using the provided coordinates."""
        with open(os.path.join(to_save_path, file_name), 'w') as pdb_file:
            atom_number = 1
            for coord in coordinates:
                x, y, z = coord
                pdb_file.write(
                    f"ATOM  {atom_number:5}  DUM DUM A{atom_number:4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
                atom_number += 1
                if atom_number == 9999:
                    atom_number = 1
            pdb_file.write("END")

    # Generate PDB files for TP, FP, TN, and FN
    generate_pdb_file(TP_coords, os.path.join(to_save_path, "TP_atoms.pdb"))
    generate_pdb_file(FP_coords, os.path.join(to_save_path, "FP_atoms.pdb"))
    generate_pdb_file(TN_coords, os.path.join(to_save_path, "TN_atoms.pdb"))
    generate_pdb_file(FN_coords, os.path.join(to_save_path, "FN_atoms.pdb"))

    print('TP:', len(TP_coords), 'FP:', len(FP_coords), 'FN:', len(FN_coords), 'TN:', len(TN_coords))


def visualize_TP_TN_FP_FN_residue_level(lig_scores, gt_indexes, residues, receptor_path, tosavepath):
    threshold = 0.5

    # Initialize lists
    tp_list = []
    fp_list = []
    tn_list = []
    fn_list = []

    with open(receptor_path, 'r') as f:
        lines = f.readlines()

    res_atoms = [len(i[1]['atoms']) for i in residues.items()]

    for i, score in enumerate(lig_scores):
        # If the atom is a true binding site
        lines2add = res_atoms[i]
        if i in gt_indexes:
            if score > threshold:
                tp_list.append(lines[:lines2add])
                del lines[:lines2add]
            else:
                fn_list.append(lines[:lines2add])
                del lines[:lines2add]
        # If the atom is not a binding site
        else:
            if score > threshold:
                fp_list.append(lines[:lines2add])
                del lines[:lines2add]
            else:
                tn_list.append(lines[:lines2add])
                del lines[:lines2add]

    # Generate PDB files for TP, FP, TN, and FN
    with open(os.path.join(tosavepath, 'TP_residues.pdb'), 'w') as f:
        for l in tp_list:
            for item in l:
                f.write(item)
    with open(os.path.join(tosavepath, 'FP_residues.pdb'), 'w') as f:
        for l in fp_list:
            for item in l:
                f.write(item)
    with open(os.path.join(tosavepath, 'FN_residues.pdb'), 'w') as f:
        for l in fn_list:
            for item in l:
                f.write(item)
    with open(os.path.join(tosavepath, 'TN_residues.pdb'), 'w') as f:
        for l in tn_list:
            for item in l:
                f.write(item)

    # print('TP:', len(tp_list), 'FP:', len(fp_list), 'FN:', len(fn_list), 'TN:', len(tn_list))


def calculate_TP_TN_FP_FN(lig_scores, gt_indexes):
    threshold = 0.5

    # Initialize lists
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i, score in enumerate(lig_scores):
        # If the atom is a true binding site
        if i in gt_indexes:
            if score > threshold:
                TP += 1
            else:
                FN += 1
        # If the atom is not a binding site
        else:
            if score > threshold:
                FP += 1
            else:
                TN += 1

    print('TP:', TP, 'FP:', FP, 'FN:', FN, 'TN:', TN)


def show_roc_curve(true_labels, lig_scores, auc_roc):
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, lig_scores)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def calculate_metrics(true_labels, predicted_labels, lig_scores, to_save_metrics_path):
    auc_roc = roc_auc_score(true_labels, lig_scores)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    pr, re, _ = precision_recall_curve(true_labels, lig_scores)
    auc_pr = auc(re, pr)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    mcc = matthews_corrcoef(true_labels, predicted_labels)

    tn, fp, fn, tp = conf_matrix.ravel()
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)  # True positive rate == sensitivity == recall
    npv = tn / (tn + fn)
    spc = tn / (fp + tn)  # Specificity or True Negative Rate

    with open(to_save_metrics_path, 'w') as f:
        print(f"AUC-ROC: {auc_roc:.4f}", file=f)
        print(f"Accuracy: {accuracy:.4f}", file=f)
        print(f"Precision: {precision:.4f}", file=f)
        print(f"Recall: {recall:.4f}", file=f)
        print(f"F1 Score: {f1:.4f}", file=f)
        print(f"AUC-PR: {auc_pr:.4f}", file=f)
        print(f"Confusion Matrix:\n {conf_matrix}", file=f)
        print(f"Matthews Correlation Coefficient: {mcc:.4f}", file=f)
        print(f"False Positive Rate (FPR): {fpr:.4f}", file=f)
        print(f"Negative Predictive Value (NPV): {npv:.4f}", file=f)
        print(f"Specificity (SPC): {spc:.4f}", file=f)

    return auc_roc, accuracy, precision, recall, f1, auc_pr, conf_matrix, mcc, fpr, npv, spc


def filter_out_HETATMs(pdb_file_path):
    with open(pdb_file_path, 'r') as infile:
        lines = infile.readlines()

    # Filter out lines starting with 'HETATM'
    filtered_lines = [line for line in lines if not line.startswith('HETATM')]

    # Write the filtered lines back to the file
    with open(pdb_file_path, 'w') as outfile:
        outfile.writelines(filtered_lines)


def write_residue_prediction_pdb(receptor, results_save_path, residues_best):
    """
    :param receptor: original receptor pdb file path
    :param results_save_path: where to save the prediction pdb file residues: the residues dict with scores
    :param residues_best: the residues dict with scores
    :return: Write the prediction PDB file with scores at residue level (replaces B-factor for residues).
    """
    rec_name = receptor.split('/')[-1].split('_')[0]
    output_pdb_path = os.path.join(results_save_path, f'{rec_name}_pred.pdb')

    # Ensure the directory exists
    os.makedirs(results_save_path, exist_ok=True)

    # Open the original receptor PDB file and the output PDB file for writing the predictions
    with open(receptor, 'r') as original_pdb, open(output_pdb_path, 'w') as pred_pdb:
        for line in original_pdb:
            if line.startswith("ATOM") or line.startswith("HETATM"):  # Process only ATOM and HETATM records
                # Extract residue info (residue number, chain ID, and insertion code)
                chain_id = line[21]
                res_num = line[22:26].strip()
                insertion_code = line[26].strip()

                # Create the residue ID in the same format as in residues_best
                res_id = f'{res_num}_{chain_id}'
                if insertion_code:
                    res_id = f'{res_id}_{insertion_code}'

                # Check if the residue exists in residues_best
                if res_id in residues_best:
                    # Extract the prediction score
                    pred_score = residues_best[res_id]['scores']

                    # Modify the line to replace the B-factor (position 61-66) with the prediction score
                    new_b_factor = f'{pred_score:6.3f}'  # Format the prediction score with 3 decimal places
                    new_line = f'{line[:60]}{new_b_factor:>6}{line[66:]}'

                    # Write the modified line to the new PDB file
                    pred_pdb.write(new_line)
                else:
                    # If no prediction score is found, write the original line
                    pred_pdb.write(line)
            else:
                # Write lines that do not start with ATOM or HETATM (like headers and footers) unchanged
                pred_pdb.write(line)

    print(f"Residue-level prediction PDB file saved as {output_pdb_path}")


def write_atom_prediction_pdb(receptor, results_save_path, lig_scores_only_receptor_atoms):
    """
    :param receptor: original receptor pdb file path
    :param results_save_path: where to save the prediction pdb file residues: the residues dict with scores
    :param residues_best: the residues dict with scores
    :return: Write the prediction PDB file with scores at atom level (replaces B-factor for each atom).
    """

    rec_name = receptor.split('/')[-1].split('_')[0]
    output_pdb_path = os.path.join(results_save_path, f'{rec_name}_pred_per_atom.pdb')

    os.makedirs(results_save_path, exist_ok=True)

    # Make sure the length of lig_scores matches the number of atoms in the PDB
    assert len(lig_scores_only_receptor_atoms) == sum(1 for line in open(receptor) if line.startswith("ATOM") or line.startswith("HETATM")), \
        "Number of scores doesn't match the number of atoms in the PDB file"

    # Open the original receptor PDB file and the output PDB file for writing the predictions
    with open(receptor, 'r') as original_pdb, open(output_pdb_path, 'w') as pred_pdb2:
        atom_index = 0  # To track which score corresponds to which atom
        for line in original_pdb:
            if line.startswith("ATOM") or line.startswith("HETATM"):  # Process only ATOM and HETATM records
                # Extract the prediction score for the current atom
                pred_score = lig_scores_only_receptor_atoms[atom_index][0]  # Get the prediction score for this atom

                # Modify the line to replace the B-factor (position 61-66) with the prediction score
                new_b_factor = f'{pred_score:6.3f}'  # Format the prediction score with 3 decimal places
                new_line = f'{line[:60]}{new_b_factor:>6}{line[66:]}'  # Insert the prediction score at the correct position

                # Write the modified line to the new PDB file
                pred_pdb2.write(new_line)

                # Increment the atom index
                atom_index += 1
            else:
                # Write lines that do not start with ATOM or HETATM (like headers and footers) unchanged
                pred_pdb2.write(line)

    print(f"Per-atom prediction PDB file saved as {output_pdb_path}")


def receptor_info(receptor, lig_scores_only_receptor_atoms):
    """
    Extract residue groups and compute the best scores for each residue in the receptor.

    Args:
        receptor (str): The path to the receptor PDB file.
        lig_scores_only_receptor_atoms (ndarray): List of ligandability scores for each atom.

    Returns:
        residues (dict): Dictionary containing atom information and ligand scores for each residue.
        residues_best (dict): Dictionary containing the best ligand score for each residue.
    """
    # Create the residue groups for the whole protein
    residues = {}
    with open(receptor, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                chain_id = line[21]  # Extract chain identifier
                atom_id = line[6:11].strip()
                res_id = f'{line[22:26].strip()}_{chain_id}'  # Concatenate residue ID with chain ID
                insertion_code = line[26].strip()
                if insertion_code:
                    res_id = f'{res_id}_{insertion_code}'
                if res_id not in residues:
                    residues[res_id] = {"atoms": [], 'scores': []}
                residues[res_id]["atoms"].append(atom_id)

                atom2check = int(atom_id) - 1
                residues[res_id]['scores'].append(lig_scores_only_receptor_atoms[atom2check][0])

    # Take the best scores for the whole protein
    residues_best = {}
    for res_id, res_data in residues.items():
        residues_best[res_id] = {'scores': []}
        check_best = res_data['scores']
        best_atom = check_best.index(max(check_best))  # We take the best atom score of the residue
        residues_best[res_id]['scores'] = check_best[best_atom]

    return residues, residues_best

def antibody_input_recognition(pdb_file):
    """
    Checks if the input PDB file corresponds to an antibody-like structure.
    Criteria:
    1. The structure contains two chains.
    2. The 'compound' section mentions "Heavy" or "Light" chains.
    3. If 'compound' section is missing or empty, rely on the two-chain criterion.

    Parameters:
    pdb_file (str): Path to the PDB file.

    Returns:
    bool: True if the input is likely an antibody, False otherwise.
    str: Reason for the decision (if False).
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)

    try:
        # Parse the structure
        structure = parser.get_structure("input", pdb_file)
    except Exception as e:
        return False, f"Failed to parse PDB file: {e}"

    # Get all chains
    chains = list(structure.get_chains())

    # Check for two chains
    if len(chains) != 2:
        return False, f"The structure contains {len(chains)} chain(s), expected 2 for an antibody (Heavy & Light)."

    # Check the 'compound' section for "Heavy" or "Light"
    try:
        compound_info = structure.header.get("compound", {})
        if compound_info:  # Proceed if 'compound' section is not empty
            for _, compound_details in compound_info.items():
                molecule_name = compound_details.get("molecule", "").lower()
                if "heavy" in molecule_name or "light" in molecule_name:
                    return True, "Antibody found based on compound information."
        # If compound_info exists but doesn't mention "Heavy" or "Light"
        # Proceed with two-chain criterion
        return True, "Antibody assumed based on two-chain structure; compound information is missing or does not mention 'Heavy'/'Light'."
    except Exception as e:
        # Handle errors in accessing 'compound' section
        return True, "Antibody assumed based on two-chain structure; compound information could not be read."

    # Default fallback
    return True, "Antibody assumed based on two-chain structure."
