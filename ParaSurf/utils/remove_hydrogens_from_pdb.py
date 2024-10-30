import os

def remove_hydrogens_from_pdb_folder(input_folder, output_folder):
    """
    Processes all PDB files in a specified directory, removing hydrogen atoms from each file and saving the result to a new directory.

    Args:
    input_folder (str): The path to the directory containing the original PDB files.
    output_folder (str): The path to the directory where the modified PDB files will be saved.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all PDB files in the input directory
    for filename in os.listdir(input_folder):
        if filename.endswith(".pdb"):  # Check for PDB files
            pdb_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)
            remove_hydrogen_atoms(pdb_file_path, output_file_path)

def remove_hydrogen_atoms(pdb_filename, output_filename):
    """
    Reads a PDB file and writes a new file excluding all hydrogen atoms.

    Args:
    pdb_filename (str): The path to the original PDB file.
    output_filename (str): The path to the output file with hydrogens removed.
    """
    with open(pdb_filename, 'r') as file:
        lines = file.readlines()

    with open(output_filename, 'w') as outfile:
        for line in lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                if line[76:78].strip() != 'H':
                    outfile.write(line)
            else:
                outfile.write(line)

if __name__ == "__main__":
    # Usage example:
    input_dir = '/home/angepapa/PycharmProjects/DeepSurf2.0/test_data/pdbs/PECAN/TRAIN'
    output_dir = '/home/angepapa/PycharmProjects/DeepSurf2.0/test_data/pdbs/PECAN/TRAIN'
    remove_hydrogens_from_pdb_folder(input_dir, output_dir)
