import os, glob


def renumber_pdb_atoms(pdb_file_path, output_file_path):
    with open(pdb_file_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    atom_id = 1
    for line in lines:
        if line.startswith("ATOM"):
            # Construct the new line with the updated atom ID
            new_line = f"{line[:6]}{atom_id:5d}{line[11:]}"
            new_lines.append(new_line)
            atom_id += 1
        else:
            new_lines.append(line)

    with open(output_file_path, 'w') as file:
        file.writelines(new_lines)


def process_pdb_files_in_folder(folder_path):
    # Find all .pdb files in the folder
    pdb_files = glob.glob(os.path.join(folder_path, '*.pdb'))

    for pdb_file in pdb_files:
        # if 'ant' in pdb_file:
        # Define the new file name
        base_name = os.path.basename(pdb_file)
        output_file_name = base_name.replace('.pdb', '.pdb')
        output_file_path = os.path.join(folder_path, output_file_name)

        # Process each file
        renumber_pdb_atoms(pdb_file, output_file_path)
        print(f"Processed {pdb_file} and saved as {output_file_path}")


if __name__ == "__main__":
    # example usage for pdb folder
    folder_path = '/home/angepapa/PycharmProjects/DeepSurf2.0/test_data/pdbs/PECAN/TRAIN'
    process_pdb_files_in_folder(folder_path)