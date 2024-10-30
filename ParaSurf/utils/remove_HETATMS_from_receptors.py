import os


def remove_hetatm_from_pdb_folder(input_folder, output_folder):
    """
    Processes all PDB files in a specified directory, removing lines that start with 'HETATM' from each file, and saving the result to a new directory.

    Args:
    input_folder (str): The path to the directory containing the original PDB files.
    output_folder (str): The path to the directory where the modified PDB files will be saved.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all PDB files in the input directory
    for filename in os.listdir(input_folder):
        if 'receptor_1.pdb' in filename:  # delete the HETATMS only on the receptor
            pdb_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)
            remove_hetatm_lines(pdb_file_path, output_file_path)

def remove_hetatm_lines(pdb_filename, output_filename):
    """
    Reads a PDB file and writes a new file excluding all lines that start with 'HETATM'.

    Args:
    pdb_filename (str): The path to the original PDB file.
    output_filename (str): The path to the output file with 'HETATM' lines removed.
    """
    with open(pdb_filename, 'r') as file:
        lines = file.readlines()

    with open(output_filename, 'w') as outfile:
        for line in lines:
            if not line.startswith('HETATM'):
                outfile.write(line)

# Usage example:
if __name__ == "__main__":
    input_dir = '/home/angepapa/PycharmProjects/DeepSurf2.0/test_data/pdbs/PECAN/TRAIN'
    output_dir = '/home/angepapa/PycharmProjects/DeepSurf2.0/test_data/pdbs/PECAN/TRAIN'
    remove_hetatm_from_pdb_folder(input_dir, output_dir)
