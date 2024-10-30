import os
import time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from ParaSurf.model import ParaSurf_model
from ParaSurf.model.dataset import dataset
from validation import validate_residue_level #(CHANGE TO validation_CDR (new!)
import wandb
from tqdm import tqdm


user = os.getenv('USER')
base_dir = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data'
CFG = {
    'name': 'ParaSurf train dummy eraseme folder',
    'initial_lr': 0.0001,
    'epochs': 100,
    'batch_size': 64,
    'grid': 41, # don't change
    'seed': 42,
    'wandb': False,
    'debug': True,
    'model_weights': None,  # if ('' or None )is given then training starts from scratch
    'num_workers': 8,
    'feat_type': ['kalasanty_with_force_fields'],
    'feats_path': os.path.join(base_dir, 'feats'),
    'TRAIN_samples': os.path.join(base_dir, 'datasets/eraseme_TRAIN.samples'),
    'VAL_proteins_list': os.path.join(base_dir, 'datasets/eraseme_VAL.proteins'),
    'VAL_proteins': os.path.join(base_dir, 'pdbs/eraseme/VAL'),
    'save_dir': f'/home/{user}/PycharmProjects/github_projects/ParaSurf/ParaSurf/train/eraseme/model_weights'
}

if CFG['wandb']:
    wandb.init(project='ParaSurf', entity='angepapa', config=CFG, name=CFG['name'])


# Set random seed for repeatability
def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


set_seed(CFG['seed'])

# model
device = 'cuda' if torch.cuda.is_available() else 'cpu'


with open(CFG['TRAIN_samples']) as f:
    lines = f.readlines()
    feature_vector_lentgh = int(lines[0].split()[1].split('/')[0].split('_')[-1])

model = ParaSurf_model.ResNet3D_Transformer(in_channels=feature_vector_lentgh,
                                                  block=ParaSurf_model.DilatedBottleneck,
                                                  num_blocks=[3, 4, 6, 3], num_classes=1).to(device)
print(model)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=CFG['initial_lr'])

scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Load Dataset
train_set = dataset(CFG['TRAIN_samples'], CFG['batch_size'], CFG['feats_path'], CFG['grid'], True,
                    feature_vector_lentgh, CFG['feat_type'])
train_loader = DataLoader(dataset=train_set, batch_size=CFG['batch_size'], shuffle=True,
                          num_workers=CFG['num_workers'])

# Training
if not os.path.exists(CFG['save_dir']):
    os.makedirs(CFG['save_dir'])

# check if pretrain weights are loaded and start the epoch from there
if CFG['model_weights'] and os.path.exists(CFG['model_weights']):
    model.load_state_dict(torch.load(CFG['model_weights']))
    start_epoch = int(CFG['model_weights'].split('/')[-1].split('.')[0].split('_')[1]) + 1
    print(f"\nLoading weights from epoch {start_epoch-1} ...\n")
    print(f"Start training for epoch {start_epoch} ...")
else:
    print('\nStart training from scratch ...')
    start_epoch = 0


train_losses = []  # to keep track of training losses

for epoch in range(start_epoch, CFG['epochs']):
    start = time.time()
    model.train()
    total_loss = 0.0

    correct_train_predictions = 0  # Reset for each epoch
    total_train_samples = 0  # Reset for each epoch

    for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs, labels = inputs.float().to(device), labels.to(device).unsqueeze(1)
        total_train_samples += labels.shape[0]
        optimizer.zero_grad()

        # scaler option
        # with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())

        loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        total_loss += loss.item()

        predicted_train = torch.sigmoid(outputs) > 0.5
        correct_train_predictions += (predicted_train == labels).sum().item()

        # Print the training loss every 100 batches
        if (i + 1) % 100 == 0:
            print(f"Epoch: {epoch + 1} Batch: {i + 1} Train Loss: {loss.item():.3f}")

        # if CFG['wandb']:
        #     wandb.log({'Mini Batch Train Loss': loss.item()})

        if CFG['debug']:
            break


    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = correct_train_predictions / total_train_samples  # Calculate training accuracy

    train_losses.append(avg_train_loss)

    cur_model_weight_path = os.path.join(CFG['save_dir'], f'epoch_{epoch}.pth')
    torch.save(model.state_dict(), cur_model_weight_path)

    avg_auc_roc, avg_precision, avg_recall, avg_auc_pr, avg_f1 = validate_residue_level(valset=CFG['VAL_proteins_list'],
                                                                    modelweights=cur_model_weight_path,
                                                                    test_folder=CFG['VAL_proteins'],
                                                                    epoch=epoch + 1,
                                                                    feat_type=CFG['feat_type'],
                                                                    feature_vector_lentgh=feature_vector_lentgh)


    print(
        f"Epoch {epoch + 1}/{CFG['epochs']} - Train Loss: {avg_train_loss:.3f}, Train Accuracy: {train_accuracy:.3f},"  
        f"Val_AUC-ROC: {avg_auc_roc:.3f}, Val_Precision: {avg_precision:.3f}, Val_Recall: {avg_recall:.3f},"            
        f" Val_AUC_pr: {avg_auc_pr:.3f}, Val_F1: {avg_f1}")

    print(f"Total epoch time: {(time.time() - start) / 60:.3f} mins")

    if CFG['wandb']:
        wandb.log({'Epoch': epoch,
                   'Train Loss': avg_train_loss,
                   'Train Accuracy': train_accuracy,
                   'Valid AUC-ROC': avg_auc_roc,
                   'Valid Precision': avg_precision,
                   'Valid Recall': avg_recall,
                   'Valid AUC-pr': avg_auc_pr,
                   'Valid F1':  avg_f1
                   })


    # Step the scheduler
    scheduler.step()

# # Finish the wandb run at the end of all epochs for the current iteration
if CFG['wandb']:
    wandb.finish()