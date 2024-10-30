import numpy as np
import pandas as pd
from utils import calculate_metrics
import os, re


def calculate_Fv_and_cdr_regions(residues_best, gt_true_label_residues, rec_name, output_path, epoch, test_csv=None,
                          thres=0.5):
    # Load the CSV file to identify heavy and light chains, if provided
    heavy_chain_name, light_chain_name = None, None
    calculate_individual_cdrs = False

    if test_csv:
        test_csv = pd.read_csv(test_csv)
        rec_info = test_csv[test_csv['pdb_code'] == rec_name]
        if not rec_info.empty:
            heavy_chain_name = rec_info['Heavy_chain'].iloc[0]
            light_chain_name = rec_info['Light_chain'].iloc[0]

            calculate_individual_cdrs = True
        else:
            print(f"Receptor {rec_name} not found in the test CSV.")

    # Define the CDR+-2 ranges for heavy and light chains
    cdr1 = list(range(25, 41))  # CDR-H/L1: 25-40
    cdr2 = list(range(54, 68))  # CDR-H/L2: 54-67
    cdr3 = list(range(103, 120))  # CDR-H/L3: 103-119
    framework_ranges = list(range(1, 25)) + list(range(41, 54)) + list(range(68, 103)) + list(range(120, 129))

    # Initialize dictionaries
    CDRH1, CDRH2, CDRH3 = {}, {}, {}
    CDRL1, CDRL2, CDRL3 = {}, {}, {}
    FRAMEWORK = {}

    # Loop over the predictions to populate the dictionaries
    for residue, data in residues_best.items():
        # Split the residue into components (e.g., '30_L_C' -> ['30', 'L', 'C'])
        residue_parts = residue.split('_')
        residue_num = int(re.findall(r'\d+', residue_parts[0])[0])
        chain_name = residue_parts[1]

        # Assign residue to the corresponding CDR or FRAMEWORK based on chain and residue number
        if (not heavy_chain_name or chain_name == heavy_chain_name):  # If no csv or matching heavy chain
            if residue_num in cdr1:
                CDRH1[residue] = data
            elif residue_num in cdr2:
                CDRH2[residue] = data
            elif residue_num in cdr3:
                CDRH3[residue] = data
            elif residue_num in framework_ranges:
                FRAMEWORK[residue] = data

        if (not light_chain_name or chain_name == light_chain_name):  # If no csv or matching light chain
            if residue_num in cdr1:
                CDRL1[residue] = data
            elif residue_num in cdr2:
                CDRL2[residue] = data
            elif residue_num in cdr3:
                CDRL3[residue] = data
            elif residue_num in framework_ranges:
                FRAMEWORK[residue] = data

    # Helper function to calculate and save metrics for each CDR and FRAMEWORK
    def calculate_and_save_metrics(cdr_dict, cdr_name, threshold=thres):
        if len(cdr_dict) > 0:  # To check if CDR exists in the antibody
            pred_scores = np.array([[i[1]['scores']] for i in cdr_dict.items()])
            pred_labels = (pred_scores > threshold).astype(int)
            gt_labels = np.array([1 if residue in gt_true_label_residues else 0 for residue in cdr_dict.keys()])

            if len(np.unique(gt_labels)) > 1:  # Ensure both classes are present
                output_results_path = os.path.join(output_path, f'{cdr_name}_results_epoch_{epoch}_{threshold}.txt')
                auc_roc, accuracy, precision, recall, f1, auc_pr, conf_matrix, mcc, _, _, _ = \
                    calculate_metrics(gt_labels, pred_labels, pred_scores, to_save_metrics_path=output_results_path)
                return auc_roc, accuracy, precision, recall, f1, auc_pr, conf_matrix, mcc
        return None

    # Calculate and save metrics for each CDR and FRAMEWORK only if .csv is provided
    if calculate_individual_cdrs:
        calculate_and_save_metrics(CDRH1, 'CDRH1')
        calculate_and_save_metrics(CDRH2, 'CDRH2')
        calculate_and_save_metrics(CDRH3, 'CDRH3')
        calculate_and_save_metrics(CDRL1, 'CDRL1')
        calculate_and_save_metrics(CDRL2, 'CDRL2')
        calculate_and_save_metrics(CDRL3, 'CDRL3')
        calculate_and_save_metrics(FRAMEWORK, 'FRAMEWORK')

    # Calculate the metrics for the CDR+-2 region (CDRH1 + CDRH2 + CDRH3 + CDRL1 + CDRL2 + CDRL3)
    cdr_plus_minus_2 = {**CDRH1, **CDRH2, **CDRH3, **CDRL1, **CDRL2, **CDRL3}
    calculate_and_save_metrics(cdr_plus_minus_2, 'CDR_plus_minus_2')

    # Calculate the metrics for the Fv region (CDRs + FRAMEWORK)
    fv_region = {**CDRH1, **CDRH2, **CDRH3, **CDRL1, **CDRL2, **CDRL3, **FRAMEWORK}
    calculate_and_save_metrics(fv_region, 'Fv')


def calculate_Fv_and_cdr_regions_only_one_chain(residues_best, gt_true_label_residues, rec_name, output_path, epoch, thres=0.5):
    """
    This function calculates metrics for the Fv and CDR+-2 regions, but only for a PDB file with one chain.
    The CSV file is not needed in this case, as there is only one chain.

    Args:
    - residues_best: Dictionary containing residue information.
    - gt_true_label_residues: List of ground truth binding residues.
    - rec_name: Name of the receptor.
    - output_path: Directory to save output results.
    - epoch: Current epoch for model validation.
    - thres: Threshold for classification (default is 0.5).

    Returns:
    - Metrics calculated and saved for CDR+-2 and Fv regions.
    """

    # Define the CDR+-2 and framework ranges for the single chain
    cdr1 = list(range(25, 41))  # CDR1: 25-40
    cdr2 = list(range(54, 68))  # CDR2: 54-67
    cdr3 = list(range(103, 120))  # CDR3: 103-119
    framework_ranges = list(range(1, 25)) + list(range(41, 54)) + list(range(68, 103)) + list(range(120, 129))

    # Initialize dictionaries
    CDR1, CDR2, CDR3 = {}, {}, {}
    FRAMEWORK = {}

    # Loop over the predictions to populate the dictionaries
    for residue, data in residues_best.items():
        # Split the residue into components (e.g., '30_L_C' -> ['30', 'L', 'C'])
        residue_parts = residue.split('_')
        residue_num = int(re.findall(r'\d+', residue_parts[0])[0])

        # Assign residue to the corresponding CDR or FRAMEWORK based on residue number
        if residue_num in cdr1:
            CDR1[residue] = data
        elif residue_num in cdr2:
            CDR2[residue] = data
        elif residue_num in cdr3:
            CDR3[residue] = data
        elif residue_num in framework_ranges:
            FRAMEWORK[residue] = data

    # Helper function to calculate and save metrics for each CDR and FRAMEWORK
    def calculate_and_save_metrics(cdr_dict, cdr_name, threshold=thres):
        if len(cdr_dict) > 0:  # Check if CDR exists in the antibody
            pred_scores = np.array([[i[1]['scores']] for i in cdr_dict.items()])
            pred_labels = (pred_scores > threshold).astype(int)
            gt_labels = np.array([1 if residue in gt_true_label_residues else 0 for residue in cdr_dict.keys()])

            if len(np.unique(gt_labels)) > 1:  # Ensure both classes are present
                output_results_path = os.path.join(output_path, f'{cdr_name}_results_epoch_{epoch}_{threshold}.txt')
                auc_roc, accuracy, precision, recall, f1, auc_pr, conf_matrix, mcc, _, _, _ = \
                    calculate_metrics(gt_labels, pred_labels, pred_scores, to_save_metrics_path=output_results_path)
                return auc_roc, accuracy, precision, recall, f1, auc_pr, conf_matrix, mcc
        return None

    # Calculate the metrics for the CDR+-2 region (CDR1 + CDR2 + CDR3)
    cdr_plus_minus_2 = {**CDR1, **CDR2, **CDR3}
    calculate_and_save_metrics(cdr_plus_minus_2, 'CDR_plus_minus_2')

    # Calculate the metrics for the Fv region (CDR1 + CDR2 + CDR3 + FRAMEWORK)
    fv_region = {**CDR1, **CDR2, **CDR3, **FRAMEWORK}
    calculate_and_save_metrics(fv_region, 'Fv')
