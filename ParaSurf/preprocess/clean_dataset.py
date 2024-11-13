import os
from ParaSurf.utils.remove_hydrogens_from_pdb import remove_hydrogens_from_pdb_folder
from ParaSurf.utils.remove_HETATMS_from_receptors import remove_hetatm_from_pdb_folder
from ParaSurf.utils.reaarange_atom_id import process_pdb_files_in_folder


def clean_dataset(dataset_path_with_pdbs):
    """
    :param dataset_path_with_pdbs:
    :return: a cleaned dataset ready to be processed for training purposes with 3 steps of filtering
    """
    data_path = dataset_path_with_pdbs

    # step1: remove hydrogens
    remove_hydrogens_from_pdb_folder(input_folder=data_path,
                                     output_folder=data_path)

    # step2: remove HETATMS only from the receptors
    remove_hetatm_from_pdb_folder(input_folder=data_path,
                                  output_folder=data_path)

    # step3: re-arrange the atom_id of each pdb
    process_pdb_files_in_folder(folder_path=data_path)

if __name__ == "__main__":
    user = os.getenv('USER')
    clean_dataset(f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/pdbs/example/TRAIN')                # all train, val and test should be cleaned