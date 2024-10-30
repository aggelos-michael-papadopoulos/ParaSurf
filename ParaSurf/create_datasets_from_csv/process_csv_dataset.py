import os
import pandas as pd
from Bio.PDB import PDBList
from tqdm import tqdm



def add_headers_if_not_present(csv_file, headerlist):
 """
 Add headers to the CSV file if they are not already present.

 Parameters:
 csv_file (str): Path to the CSV file.
 headerlist (list): List of headers to add.
 """
 # Read the first row to check for headers
 first_row = pd.read_csv(csv_file, nrows=1)

 # Check if the first row contains the expected headers
 if list(first_row.columns) != headerlist:
     print(f"Headers not found in {csv_file}. Adding headers...")
     # Load the full data without headers
     data = pd.read_csv(csv_file, header=None)
     # Assign the correct headers
     data.columns = headerlist
     # Save the file with the correct headers
     data.to_csv(csv_file, header=True, index=False)
     print(f"Headers added to {csv_file}")
 else:
     print(f"Headers already present in {csv_file}. No changes made.")


def download_pdb(pdb_code, output_dir):
 pdbl = PDBList()
 pdbl.retrieve_pdb_file(pdb_code, pdir=output_dir, file_format='pdb')


def download_and_rename_pdb_files(pdb_list, folder):

     """
     Downloads PDB files from the provided list and renames them from `.ent` to `{pdb_code}.pdb`.

     Parameters:
     pdb_list (list): List of PDB codes to be downloaded.
     folder (str): Directory where the PDB files will be saved and renamed.
     """
     # Download PDB files
     for pdb_code in pdb_list:
         download_pdb(pdb_code, folder)

     # Rename files to {pdb_code}.pdb
     for pdb_file in os.listdir(folder):
         if pdb_file.endswith('.ent'):
             old_file_path = os.path.join(folder, pdb_file)
             new_file_name = pdb_file.split('.')[0][-4:].upper() + '.pdb' #Capital because the csv gives the pdb names in capital
             new_file_path = os.path.join(folder, new_file_name)
             os.rename(old_file_path, new_file_path)
             print(f"Renamed {old_file_path} to {new_file_path}")


def process_dataset(csv_file, folder):

     """
     Processes a dataset by adding headers, extracting PDB codes, and downloading/renaming PDB files.

     Parameters:
     csv_file (str): Path to the CSV file.
     folder (str): Directory where the PDB files will be saved and renamed.
     """
     # Add headers if not present
     add_headers_if_not_present(csv_file, headerlist)

     # Read the CSV file
     dataset = pd.read_csv(csv_file)

     # Create folder if it doesn't exist
     if not os.path.exists(folder):
         os.makedirs(folder)

     # Initialize the PDB list
     pdb_list = []

     # Process each row
     for i, row in dataset.iterrows():
         pdb_list.append(row['pdb_code'])

     # Download and rename PDB files
     download_and_rename_pdb_files(pdb_list, folder)

if __name__ == '__main__':

    # ALL datasets follow the same process
    user = os.getenv('USER')
    datasets = ['PECAN', 'Paragraph_Expanded', 'MIPE']


    # Define the correct headers
    headerlist = ['pdb_code', 'Light_chain', 'Heavy_chain', 'ag']

    for dataset in datasets:

        if dataset == 'MIPE': # here the split is train-val and test according to the MIPE paper
            # csv path
            train_val = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/training_data/{dataset}/train_val.csv'
            test = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/training_data/{dataset}/test_set.csv'

            # path to init raw PDB storage
            train_val_folder = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/pdbs/{dataset}/train_val_data_initial_raw_pdb_files'
            test_folder = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/pdbs/{dataset}/test_data_initial_raw_pdb_files'

            process_dataset(train_val, train_val_folder)
            process_dataset(test, test_folder)

        else:
            # Paths to your CSV files. Download dataset from here: https://github.com/oxpig/Paragraph/tree/main/training_data/Expanded
            train = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/training_data/{dataset}/train_set.csv'
            val = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/training_data/{dataset}/val_set.csv'
            test = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/training_data/{dataset}/test_set.csv'


            # Paths for init raw PDB file storage
            train_folder = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/pdbs/{dataset}/train_data_initial_raw_pdb_files'
            val_folder = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/pdbs/{dataset}/val_data_initial_raw_pdb_files'
            test_folder = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/pdbs/{dataset}/test_data_initial_raw_pdb_files'


            # Process each dataset
            process_dataset(train, train_folder)
            process_dataset(val, val_folder)
            process_dataset(test, test_folder)
