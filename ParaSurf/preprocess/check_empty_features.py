import os

def remove_empty_features(feats_folder, pdbs_path, surf_path, log_file_path="removed_complexes_log.txt"):
    """
    Checks each subfolder in the base folder for files. If a subfolder is empty, removes it along with
    associated files from `data_path` and `surf_path` and logs the removals.

    Parameters:
    - feats_folder (str): The main directory containing subfolders with the features to check.
    - data_path (str): Path where receptor and antigen PDB files are located.
    - surf_path (str): Path where surface points files are located.
    - log_file_path (str): Path for the log file to track removed folders and files. Default is 'removed_folders_log.txt'.

    Returns:
    - total_empty_folders (int): Count of empty folders removed.
    """

    # Identify all subfolders in the base folder
    subfolders = [d for d in os.listdir(feats_folder) if os.path.isdir(os.path.join(feats_folder, d))]
    empty_folders = []

    # Open log file to record removed folders
    with open(log_file_path, 'w') as log_file:
        log_file.write("Log of Removed Folders and Files\n")
        log_file.write("=" * 30 + "\n")

        # Check each subfolder and remove if empty
        for folder in subfolders:
            path = os.path.join(feats_folder, folder)
            if not any(os.path.isfile(os.path.join(path, i)) for i in os.listdir(path)):
                empty_folders.append(folder)
                pdb_code = folder.split('_')[0]

                # Define paths to the files to be removed
                rec_file = os.path.join(pdbs_path, pdb_code + '_receptor_1.pdb')
                antigen_file = os.path.join(pdbs_path, pdb_code + '_antigen_1_1.pdb')
                surf_file = os.path.join(surf_path, pdb_code + '_receptor_1.surfpoints')

                # Remove the empty folder and associated files
                os.rmdir(path)
                if os.path.exists(rec_file):
                    os.remove(rec_file)
                if os.path.exists(antigen_file):
                    os.remove(antigen_file)
                if os.path.exists(surf_file):
                    os.remove(surf_file)

                # Log each removal
                log_file.write(f"{pdb_code} complex removed since no features found.\n")

    total_empty_folders = len(empty_folders)
    # Delete the log file if no folders were removed
    if total_empty_folders == 0:
        os.remove(log_file_path)
        print("\nAll complexes have features!!!")
    else:
        print(f"Total empty folders removed: {total_empty_folders}")
        print(f"Details logged in {log_file_path}")

    return total_empty_folders

# Example usage
if __name__ == '__main__':
    user = os.getenv('USER')
    pdbs_path = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/pdbs/eraseme/TRAIN'
    surf_path = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/surf_points/eraseme/TRAIN'
    feats_path = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/feats/eraseme_22'
    remove_empty_features(feats_path, pdbs_path, surf_path)
