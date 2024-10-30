import os

# Here we create a .proteins file that has all the proteins==receptors that we are working with.
features_path = r'/home/angepapa/PycharmProjects/DeepSurf2.0/test_data/feats/eraseme_22'
proteins_file = r'/home/angepapa/PycharmProjects/DeepSurf2.0/test_data/datasets/eraseme.proteins'

receptors = []

for prot in os.listdir(features_path):
    prot_name = prot.split('.')[0]
    if 'rec' in prot:
        receptors.append(prot_name + '\n')

with open(proteins_file,'w') as f:
    f.writelines(receptors)