import os, random


user = os.getenv('USER')

feats_path = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/feats/eraseme_22'  # input folder with protein grids (training features)
proteins_file = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/datasets/eraseme_TRAIN.proteins'  # input file with a list of train proteins
samples_file = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/datasets/eraseme.samples'  # output file with respective training samples info (class_label + sample_path)
seed = 1

with open(proteins_file, 'r') as f:
    proteins = f.readlines()

sample_lines = []
feats_prefix = feats_path.rsplit('/')[-1]

for prot in proteins:
    prot = prot[:-1]
    prot_feats_path = os.path.join(feats_path, prot)
    if not os.path.isdir(prot_feats_path):
        print('No features for ', prot)
        continue
    for sample in os.listdir(prot_feats_path):
        cls_idx = sample[-5]
        sample_lines.append(cls_idx + ' ' + feats_prefix + '/' + prot + '/' + sample + '\n')

random.seed(seed)
random.shuffle(sample_lines)

with open(samples_file, 'w') as f:
    f.writelines(sample_lines)