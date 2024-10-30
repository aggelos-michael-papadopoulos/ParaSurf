import time
import os
from tqdm import tqdm
from fix_surfpoints_format_issues import process_surfpoints_directory

def generate_molecular_surface(input_path, out_path):
    """
    Generates the molecular surface for protein structures in PDB files using the DMS tool.

    Parameters:
    - input_path (str): Path to the input directory containing protein PDB files.
    - out_path (str): Path to the output directory where generated surface points files will be saved.

    Process:
    - The function iterates over receptor PDB files in the input path.
    - For each receptor file, it checks if a corresponding surface points file already exists in the output directory.
    - If the surface points file does not exist, it generates the file using the DMS tool with a density of 0.5 Å.

    Outputs:
    - Each receptor file generates a surface points file saved in `out_path`.
    """

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    start = time.time()
    for f in tqdm(os.listdir(input_path), desc="Generating surface points"):
        if 'antigen' in f:
            continue

        surfpoints_file = os.path.join(out_path, f[:-3] + 'surfpoints')
        if os.path.exists(surfpoints_file):
            continue

        print(f"Processing {f}")
        os.system(f'dms {os.path.join(input_path, f)} -d 0.5 -n -o {surfpoints_file}')

    # Calculate and print statistics
    rec_count = sum(1 for receptor in os.listdir(input_path) if 'receptor' in receptor)
    total_time = (time.time() - start) / 60  # Convert time to minutes
    print(f'Total time to create surfpoints for {rec_count} receptors: {total_time:.2f} mins')


# Example usage
if __name__ == '__main__':
    pdbs_path = '/home/angepapa/PycharmProjects/DeepSurf2.0/test_data/pdbs/PECAN/TEST'
    surfpoints_path = '/home/angepapa/PycharmProjects/DeepSurf2.0/test_data/surf_points/PECAN/TEST'

    # create the molecular surface
    generate_molecular_surface(
        input_path= pdbs_path,
        out_path= surfpoints_path
    )

    # fix some format issues with the .surfpoints files
    process_surfpoints_directory(surfpoints_path)