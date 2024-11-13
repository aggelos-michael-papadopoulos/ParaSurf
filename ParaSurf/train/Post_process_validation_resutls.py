import os
import numpy as np

user = os.getenv('USER')
results_path = '/home/angepapa/PycharmProjects/github_projects/ParaSurf/ParaSurf/train/TEST_results'

epoch = 8                                       # specify the epoch of your best model, IF YOU HAVE ONLY THE best.pth (e.g. PECAN_best.pth), just type epoch='best'
# epoch = 'best'
thresholds = [0.5, 0.734]
cases = ['CDR_plus_minus_2', 'Fv', 'Fab']
dataset = 'PECAN'                               # choose the dataset; 'PECAN', 'Paragraph_Expanded', 'MIPE'

# Reset lists at the start of each epoch
auc_roc = []
accuracy = []
precision = []
recall = []
f1_score = []
auc_pr = []
mcc = []
fpr = []
npv = []
spc = []
for case in cases:
    for threshold in thresholds:
        for i in os.listdir(results_path):
            # if 'inter' in i:        # not to take any other folders or .txt files
            if 'rec' in i:        # not to take any other folders or .txt files
                cur_my_result_txt = os.path.join(results_path, i, f'{case}_results_epoch_{epoch}_{threshold}.txt')      # cdr
                if case == 'Fab':
                    cur_my_result_txt = os.path.join(results_path, i, f'{case}_results_epoch_{epoch}.txt')  # Fab
                pdb_name = i.split("_")[0]
                if os.path.exists(cur_my_result_txt):
                    with open(cur_my_result_txt, 'r') as file:
                        for line in file:
                            if 'AUC-ROC:' in line:
                                cur_auc_roc = float(line.split(':')[1].strip())
                                auc_roc.append(cur_auc_roc)
                            elif 'Accuracy:' in line:
                                cur_accuracy = float(line.split(':')[1].strip())
                                accuracy.append(cur_accuracy)
                            elif 'Precision:' in line:
                                cur_precision = float(line.split(':')[1].strip())
                                precision.append(cur_precision)
                            elif 'Recall:' in line:
                                cur_recall = float(line.split(':')[1].strip())
                                recall.append(cur_recall)
                            elif 'F1 Score:' in line:
                                cur_f1_score = float(line.split(':')[1].strip())
                                f1_score.append(cur_f1_score)
                            elif 'AUC-PR:' in line:
                                cur_auc_pr = float(line.split(':')[1].strip())
                                auc_pr.append(cur_auc_pr)
                            elif 'Matthews Correlation Coefficient:' in line:
                                cur_mcc = float(line.split(':')[1].strip())
                                mcc.append(cur_mcc)
                            elif 'False Positive Rate (FPR):' in line:
                                cur_fpr = float(line.split(':')[1].strip())
                                fpr.append(cur_fpr)
                            elif 'Negative Predictive Value (NPV):' in line:
                                cur_npv = float(line.split(':')[1].strip())
                                npv.append(cur_npv)
                            elif 'Specificity (SPC):' in line:
                                cur_spc = float(line.split(':')[1].strip())
                                spc.append(cur_spc)

        with open(os.path.join(results_path, f'{dataset}_results_epoch_{epoch}_{case}_{threshold}.txt'), 'w') as file:
            file.write("Average Metrics:\n\n")
            file.write(f"Average AUC-ROC: {np.mean(auc_roc):.4f}\n")
            file.write(f"Average Accuracy: {np.mean(accuracy):.4f}\n")
            file.write(f"Average Precision: {np.mean(precision):.4f}\n")
            file.write(f"Average Recall: {np.mean(recall):.4f}\n")
            file.write(f"Average F1 Score: {2 * np.mean(precision) * np.mean(recall) / (np.mean(precision) + np.mean(recall)):.4f}\n")
            # file.write(f"Average F1 Score: {np.mean(f1_score):.4f}\n")
            file.write(f"Average AUC-PR: {np.mean(auc_pr):.4f}\n")
            file.write(f"Average MCC: {np.mean(mcc):.4f}\n")
            file.write(f"CAUROC: {np.median(auc_roc):.4f}\n")
            file.write(f"Average NPV: {np.mean(npv):.4f}\n")
            file.write(f"Average SPC: {np.mean(spc):.4f}\n")
            file.write(f"Average FPR: {np.mean(fpr):.4f}\n")