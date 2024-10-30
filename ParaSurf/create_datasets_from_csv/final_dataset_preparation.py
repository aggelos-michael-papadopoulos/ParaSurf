import os
import shutil
import pandas as pd
from split_pdb2chains_only import extract_chains_from_pdb
from tqdm import tqdm


def process_raw_pdb_data(info_df, initial_raw_pdb_files, final_folder):
    """
    Processes the raw PDB files by extracting the specific antibody and antigen chains from the .csv file, merging them,
    and saving the merged files in the final train_val and test folder.

    Parameters:
    info_df (DataFrame): DataFrame containing the PDB codes, antibody chains, and antigen chains.
    initial_raw_pdb_files (str): Path to the initial raw PDB files directory.
    final_folder (str): Path to the folder where the processed files will be saved.
    """
    if not os.path.exists(final_folder):
        os.makedirs(final_folder)

    for i, row in tqdm(info_df.iterrows(), total=len(info_df)):
        pdb_id = row['pdb_code']
        ab_heavy_chain = row['Heavy_chain'] # Use only this line if you want to construct the only heavy chain dataset
        ab_light_chain = row['Light_chain'] # Use only this line if you want to construct the only light chain dataset
        ag_chain = row['ag']

        pdb_file = os.path.join(initial_raw_pdb_files, pdb_id + '.pdb')
        # Extract all the chains from the pdb file and save them to /tmp
        chain_files, all_chains = extract_chains_from_pdb(pdb_file, '/tmp')

        # Assign the correct chains
        ab_heavy_chain_path = f'/tmp/{pdb_id}_chain{ab_heavy_chain}.pdb'
        ab_light_chain_path = f'/tmp/{pdb_id}_chain{ab_light_chain}.pdb'

        # Merge antibody chains into one file
        receptor_output_path = f'{final_folder}/{pdb_id}_receptor_1.pdb'
        with open(receptor_output_path, 'w') as receptor_file:
            for ab_file in [ab_heavy_chain_path, ab_light_chain_path]: # also delete one (ab_heavy_chain_path or ab_light_chain_path) if you construct the only heavy/light chain dataset
                with open(ab_file, 'r') as infile:
                    receptor_file.write(infile.read())

        print(f"Successfully merged {ab_heavy_chain} and {ab_light_chain} into {receptor_output_path}")

        ag_chain_list = ag_chain.split(';')

        if len(ag_chain_list) == 1:
            # If there's only one antigen chain
            ag_chain_1 = ag_chain_list[0].strip()
            ag_chain_1_path = f'/tmp/{pdb_id}_chain{ag_chain_1}.pdb'
            print(f"Handling one antigen chain: {ag_chain_1}")

            # Copy the single antigen chain to the output
            antigen_output_path = f'{final_folder}/{pdb_id}_antigen_1_1.pdb'
            shutil.copyfile(ag_chain_1_path, antigen_output_path)

            print(f"Successfully copied {ag_chain_1} to {antigen_output_path}")

        elif len(ag_chain_list) == 2:
            # If there are two antigen chains
            ag_chain_1, ag_chain_2 = ag_chain_list
            ag_chain_1_path = f'/tmp/{pdb_id}_chain{ag_chain_1}.pdb'
            ag_chain_2_path = f'/tmp/{pdb_id}_chain{ag_chain_2}.pdb'
            print(f"Handling two antigen chains: {ag_chain_1}, {ag_chain_2}")

            # Merge the antigen chains into a single PDB file
            antigen_output_path = f'{final_folder}/{pdb_id}_antigen_1_1.pdb'
            with open(antigen_output_path, 'w') as outfile:
                for ag_file in [ag_chain_1_path, ag_chain_2_path]:
                    with open(ag_file, 'r') as infile:
                        outfile.write(infile.read())

            print(f"Successfully merged {ag_chain_1} and {ag_chain_2} into {antigen_output_path}")

        elif len(ag_chain_list) == 3:
            # If there are three antigen chains
            ag_chain_1, ag_chain_2, ag_chain_3 = ag_chain_list
            ag_chain_1_path = f'/tmp/{pdb_id}_chain{ag_chain_1}.pdb'
            ag_chain_2_path = f'/tmp/{pdb_id}_chain{ag_chain_2}.pdb'
            ag_chain_3_path = f'/tmp/{pdb_id}_chain{ag_chain_3}.pdb'
            print(f"Handling three antigen chains: {ag_chain_1}, {ag_chain_2}, {ag_chain_3}")

            # Merge the antigen chains into a single PDB file
            antigen_output_path = f'{final_folder}/{pdb_id}_antigen_1_1.pdb'
            with open(antigen_output_path, 'w') as outfile:
                for ag_file in [ag_chain_1_path, ag_chain_2_path, ag_chain_3_path]:
                    with open(ag_file, 'r') as infile:
                        outfile.write(infile.read())

            print(f"Successfully merged {ag_chain_1}, {ag_chain_2}, and {ag_chain_3} into {antigen_output_path}")

        # At the end, remove all the chain pdb files from the /tmp folder
        for chain_file in chain_files:
            os.remove(chain_file)


if __name__ == '__main__':
    user = os.getenv('USER')

    datasets = ['PECAN', 'Paragraph_Expanded', 'MIPE']

    for dataset in datasets:
        if dataset == 'MIPE':  # here the split is train-val and test according to the MIPE paper
            # csv path
            train_val_info = pd.read_csv(f'/home/{user}/PycharmProjects/github_projects/ParaSurf/training_data/{dataset}/train_val.csv')
            test_info = pd.read_csv(f'/home/{user}/PycharmProjects/github_projects/ParaSurf/training_data/{dataset}/test_set.csv')

            # path to init raw PDB storage
            init_pdb_files_train_val = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/pdbs/{dataset}/train_val_data_initial_raw_pdb_files'
            init_pdb_files_test = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/pdbs/{dataset}/test_data_initial_raw_pdb_files'

            # final folder
            final_train_val_folder = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/pdbs/{dataset}/TRAIN_VAL'
            final_test_folder = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/pdbs/{dataset}/TEST'

            process_raw_pdb_data(train_val_info, init_pdb_files_train_val, final_train_val_folder)
            process_raw_pdb_data(test_info, init_pdb_files_test, final_test_folder)

            shutil.rmtree(init_pdb_files_train_val)
            shutil.rmtree(init_pdb_files_test)

    else:
        # Paths to dataset csv files
        train_info = pd.read_csv(f'/home/{user}/PycharmProjects/github_projects/ParaSurf/training_data/{dataset}/train_set.csv')
        val_info = pd.read_csv(f'/home/{user}/PycharmProjects/github_projects/ParaSurf/training_data/{dataset}/val_set.csv')
        test_info = pd.read_csv(f'/home/{user}/PycharmProjects/github_projects/ParaSurf/training_data/{dataset}/test_set.csv')


        # Paths to init raw pdb files
        initial_pdb_files_train = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/pdbs/{dataset}/train_data_initial_raw_pdb_files'
        initial_pdb_files_val = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/pdbs/{dataset}/val_data_initial_raw_pdb_files'
        initial_pdb_files_test = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/pdbs/{dataset}/test_data_initial_raw_pdb_files'

        # Final folder for the merged files that contain the final PDB complexes
        final_train_folder = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/pdbs/{dataset}/TRAIN'
        final_val_folder = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/pdbs/{dataset}/VAL'
        final_test_folder = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/pdbs/{dataset}/TEST'

        # Process the train-val-test data
        process_raw_pdb_data(train_info, initial_pdb_files_train, final_train_folder)
        process_raw_pdb_data(val_info, initial_pdb_files_val, final_val_folder)
        process_raw_pdb_data(test_info, initial_pdb_files_test, final_test_folder)

        # REMOVE the init raw pdb files
        shutil.rmtree(initial_pdb_files_train)
        shutil.rmtree(initial_pdb_files_val)
        shutil.rmtree(initial_pdb_files_test)
