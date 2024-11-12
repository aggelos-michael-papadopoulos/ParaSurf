import os


# Here we create a .proteins file that has all the proteins==receptors that we are working with.
cases = ['TRAIN', 'VAL', 'TEST'] # change to ['TRAIN_VAL', 'TEST'] for MIPE
user = os.getenv('USER')
for case in cases:
    pdbs_path = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/pdbs/example/{case}'
    proteins_file = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/datasets/example_{case}.proteins'  # run for train val and test

    # Create directories if they don't exist
    os.makedirs(pdbs_path, exist_ok=True)
    os.makedirs(os.path.dirname(proteins_file), exist_ok=True)

    receptors = []

    for prot in os.listdir(pdbs_path):
        prot_name = prot.split('.')[0]
        if 'rec' in prot:
            receptors.append(prot_name + '\n')

    with open(proteins_file,'w') as f:
        f.writelines(receptors)