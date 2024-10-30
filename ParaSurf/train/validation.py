import os
from network import Network
from protein import Protein_pred
from bsite_extraction import Bsite_extractor
from distance_coords import locate_receptor_binding_site_atoms_residue_level, coords2pdb_residue_level
from V_domain_results import calculate_Fv_and_cdr_regions
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import warnings
from utils import show_roc_curve, calculate_metrics, calculate_TP_TN_FP_FN
from statistics import mean


# Ignore warnings
warnings.filterwarnings('ignore')


def validate_residue_level(valset, modelweights, test_folder, epoch, feat_type, feature_vector_lentgh, training_scheme_on=True):

    CFG_predict = {
        'TEST_FOLDER': test_folder,             # test folder with receptors-antigens (we need the antigens to calculate the metrics)
        'k_fold_test_txt': valset,
        'MODEL_WEIGHTS_PATH': modelweights,
        'OUTPUT_DIR': ('/').join(modelweights.split('/')[:-2])+'/VAL_results' ,
        'batch_size': 64,
        'dist_cutoff': 4.5,                     # Armstrong threshold
        'prediction_threshold': 0.5,            # 0,734 for Expanded_dataset
        'Grid_size': 41,                        # size of the voxel
        'test_csv': '/home/angepapa/PycharmProjects/github_projects/ParaSurf/training_data/PECAN/test_set.csv',  # ('' or None ) give only the path only if you want to inspect inside each cdr loop CDRH1,H2, H3,L1,L2, L3 and the Framework.
        'feature_channels': feature_vector_lentgh,
        'residue_score_metric': "max",          # "mean" or "max";calculate the max or the average scores of all the atoms of the residue, else take the maximum
        'add_atoms_radius_ff_features': True,   # careful here
        'seed': 42,
        'device': 'cuda',                 # cuda or cpu
        'debug': False
    }

    if not training_scheme_on:
        CFG_predict['OUTPUT_DIR'] = ('/').join(modelweights.split('/')[:-2])+'/TEST_results'

    if feat_type[0] == 'kalasanty':
        add_forcefields = False

    elif feat_type[0] == 'kalasanty_with_force_fields':
        add_forcefields = True


    def check_path_exists(path):
        """Check if a file or directory exists. If not, raise an error."""
        if not os.path.exists(path):
            raise IOError(f'{path} does not exist.')


    def ensure_directory(directory):
        """Ensure the directory exists. If not, create it."""
        if not os.path.exists(directory):
            os.makedirs(directory)


    # Check existence
    check_path_exists(CFG_predict['TEST_FOLDER'])
    check_path_exists(CFG_predict['MODEL_WEIGHTS_PATH'])
    ensure_directory(CFG_predict['OUTPUT_DIR'])

    # Initialize lists to store metrics
    auc_roc_values = []
    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_values = []
    auc_pr_values = []
    mcc_values = []

    all_fpr = []
    all_tpr = []


    to_test_receptors = []
    with open(CFG_predict['k_fold_test_txt'], 'r') as f:
        for line in f:
            to_test_receptors.append(f'{line.strip()}.pdb')

    for rec in tqdm(to_test_receptors, total=len(to_test_receptors)):
        rec_path = os.path.join(CFG_predict['TEST_FOLDER'], rec)

        # Process Protein
        # filter_out_HETATMs(rec_path) # maybe comment was done in the clean_dataset --> so delete this line
        prot = Protein_pred(rec_path,
                       save_path=CFG_predict['OUTPUT_DIR'])

        # generate surfpoints pdb file
        surf_file = os.path.join(prot.save_path, [i for i in os.listdir(prot.save_path) if 'surfpoints' in i][0])

        # gather all corresponding antigens for this receptor
        rec_name = rec_path.split('/')[-1].split('_')[0]
        rec_id = rec_path.split('_receptor_')[-1].split('.')[0]

        matched_antigens = [os.path.join(CFG_predict['TEST_FOLDER'], antigen) for antigen in os.listdir(CFG_predict['TEST_FOLDER'])
                            if rec_name in antigen and f'_{rec_id}_' in antigen]

        print(matched_antigens)
        print(f'\n found {len(matched_antigens)} antigens for receptor {rec_path.split("/")[-1]}')

        print(f'CURRENT ANTIGEN: {matched_antigens[0]}')
        print(f'dealing with surffile: {surf_file}')
        print('##################')

        # find the GT residues of the receptor < 6 arm from antigen
        gt_res_level_bind_site_pdb = os.path.join(prot.save_path, 'real_binding_site_res_level_orig.pdb')
        receptor_binding_atoms_coords, total_receptor_coords, elements = locate_receptor_binding_site_atoms_residue_level(
            rec_path,
            matched_antigens[0],
            distance_cutoff=CFG_predict['dist_cutoff'])

        coords2pdb_residue_level(receptor_binding_atoms_coords,
                                 gt_res_level_bind_site_pdb,
                                 elements)

        # create the residue groups for the binding site. Ground Truth Residues that are < distance threshold
        gt_true_label_residues = []
        with open(gt_res_level_bind_site_pdb, 'r') as file:
            for line in file:
                if line.startswith("ATOM"):
                    chain_id = line[21]
                    res_id = f'{line[22:26].strip()}_{chain_id}'  # Concatenate residue ID with chain ID
                    insertion_code = line[26].strip()
                    if insertion_code:
                        res_id = f'{res_id}_{insertion_code}'
                    if res_id not in gt_true_label_residues:
                        gt_true_label_residues.append(res_id)

        # find the indexes of all the receptor atoms and the true_label atoms from the surfpoints
        # it is the index mapping procedure
        only_receptor_atoms_indexes = []
        # gt_res_indexes = []         # we do not care about these indexes
        atom_id = 0
        with open(surf_file, 'r') as file:
            for line in file:
                parts = line.split()
                cur_res_id = parts[1]
                # x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                # check for gt_residues
                if parts[6] == 'A':
                    only_receptor_atoms_indexes.append(atom_id)
                # if cur_res_id in gt_true_label_residues:
                #     gt_res_indexes.append(atom_id)          # we do not really need this info since we calculating the gt_indexes below

                atom_id += 1

        # Get Ligandability Scores
        nn = Network(CFG_predict['MODEL_WEIGHTS_PATH'],gridSize=CFG_predict['Grid_size'],
                     feature_channels=CFG_predict['feature_channels'], device=CFG_predict['device'])

        lig_scores = nn.get_lig_scores(prot, batch_size=CFG_predict['batch_size'], add_forcefields=add_forcefields,
                                       add_atom_radius_features=CFG_predict['add_atoms_radius_ff_features'])

        lig_scores_only_receptor_atoms = np.array([lig_scores[i] for i in only_receptor_atoms_indexes])

        # create the residue groups for the whole protein
        residues = {}
        with open(rec_path, 'r') as file:
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

        # take the best scores for the whole protein
        residues_best = {}
        for i in residues.items():
            residues_best[i[0]] = {"atoms": [], 'scores': []}
            check_best = residues[i[0]]['scores']
            if CFG_predict['residue_score_metric'] == 'mean':
                score = mean(check_best)
                residues_best[i[0]]['atoms'] = residues[i[0]]['atoms'][0]
                residues_best[i[0]]['scores'] = score

            elif CFG_predict['residue_score_metric'] == 'max':
                best_atom = check_best.index(max(check_best))  # we take the best atom score of the residue
                residues_best[i[0]]['atoms'] = residues[i[0]]['atoms'][best_atom]
                residues_best[i[0]]['scores'] = residues[i[0]]['scores'][best_atom]

        true_labels_residue = np.zeros(len(residues))
        gt_indexes = []
        for j, i in enumerate(residues.items()):
            if i[0] in gt_true_label_residues:
                true_labels_residue[j] = 1
                gt_indexes.append(j)

        ################## RESULTS ##############################################
        # Predicted labels based on a threshold (e.g., 0.5). Adjust as needed.
        predicted_scores = np.array([[i[1]['scores']] for i in residues_best.items()])
        predicted_labels = (predicted_scores > CFG_predict['prediction_threshold']).astype(int)

        # save the metric results for each receptor
        output_results_path = os.path.join(prot.save_path, f'Fab_results_epoch_{epoch}.txt')
        auc_roc, accuracy, precision, recall, f1, auc_pr, conf_matrix, mcc, _, _, _ = \
            (calculate_metrics(true_labels_residue, predicted_labels, predicted_scores,
                               to_save_metrics_path=output_results_path)
             )

        print(f'AUC_ROC score: {auc_roc}')
        auc_roc_values.append(auc_roc)
        accuracy_values.append(accuracy)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        auc_pr_values.append(auc_pr)
        mcc_values.append(mcc)

        # for the final AUC-ROC plot
        fpr, tpr, _ = roc_curve(true_labels_residue, predicted_scores)
        all_fpr.append(fpr)
        all_tpr.append(tpr)

        calculate_metrics(true_labels_residue, predicted_labels, predicted_scores,
                          to_save_metrics_path=output_results_path)

        calculate_TP_TN_FP_FN(predicted_scores, gt_indexes)

        # Extract Binding Sites
        extractor = Bsite_extractor()
        extractor.extract_bsites(prot, lig_scores)


        # Calculate CDR metrics
        if CFG_predict['test_csv'] and os.path.exists(CFG_predict['test_csv']):
            calculate_Fv_and_cdr_regions(residues_best, gt_true_label_residues, rec_name, prot.save_path, epoch,
                                         test_csv=CFG_predict['test_csv'])
        else:
            calculate_Fv_and_cdr_regions(residues_best, gt_true_label_residues, rec_name, prot.save_path, epoch)

        if CFG_predict['debug']:
            break

    #############################################################################################################
    # Calculate average metrics after loop ends
    avg_auc_roc = np.mean(auc_roc_values)
    avg_accuracy = np.mean(accuracy_values)
    avg_precision = np.mean(precision_values)
    avg_recall = np.mean(recall_values)
    avg_f1 = np.mean(f1_values)
    avg_auc_pr = np.mean(auc_pr_values)
    avg_mcc = np.mean(mcc_values)
    cauroc = np.median(auc_roc_values)

    # todo erase the print after testing
    print(f'------------- results for epoch {epoch} -------------\n'
          f'AUC-ROC: {avg_auc_roc}\n'
          f'accuracy: {avg_accuracy}\n'
          f'precision: {avg_precision}\n'
          f'recall: {avg_recall}\n'
          f'f1: {avg_f1}\n'
          f'AUC-Pr: {avg_auc_pr}\n'
          f'MCC: {avg_mcc}\n'
          f'CAUROC: {cauroc}\n')

    return avg_auc_roc, avg_precision, avg_recall, avg_auc_pr, avg_f1

# TEST best epoch weights on the TEST SET to reproduce paper results
if __name__ == "__main__":
    user = os.getenv('USER')
    test_set = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/datasets/eraseme_TEST.proteins'
    model_weights_path = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/ParaSurf/train/eraseme/model_weights/epoch_0.pth'
    test_folder = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/pdbs/eraseme/TEST'
    epoch = int(model_weights_path.split('/')[-1].split('.')[0].split('_')[-1])
    feat_type = ['kalasanty_with_force_fields']
    feature_vector_lentgh = 22

    validate_residue_level(test_set, model_weights_path, test_folder, epoch, feat_type, feature_vector_lentgh, training_scheme_on=False)