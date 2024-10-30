import os
import math
from tqdm import tqdm



def locate_receptor_binding_site_atoms(receptor_pdb_file, antigen_pdb_file, distance_cutoff=4):
    rec_coordinates = []
    with open(receptor_pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                rec_coordinates.append((x, y, z))

    ant_coordinates = []
    with open(antigen_pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                ant_coordinates.append((x, y, z))

    # Create a list to store the final coordinates
    final_coordinates = []

    # Compare each coordinate from rec_coordinates with each coordinate from ant_coordinates
    for rec_coord in rec_coordinates:
        for ant_coord in ant_coordinates:
            if math.dist(rec_coord, ant_coord) < distance_cutoff:
                final_coordinates.append(rec_coord)
                break  # Break the inner loop if a match is found to avoid duplicate entries

    # sanity check
    for coor in final_coordinates:
        if coor not in rec_coordinates:
            print('BINDING SITE COORDINATE NOT IN RECEPTORs COORDINATES!!!!!!')
    return final_coordinates, rec_coordinates


def check_receptor_antigen_interactions(pdb_dir, distance_cutoff=6, log_file="interaction_issues.txt"):
    """
    :param pdb_dir: directory with receptor and antigen pdb files
    :param distance_cutoff: the distance cutoff for binding site
    :param log_file: the file where issues will be logged
    :return: It checks if the receptor and antigen are in contact with each other
    """
    all_successful = True  # A flag to track if all pairs are correct

    # Open the log file for writing
    with open(log_file, 'w') as log:
        log.write("Receptor-Antigen Interaction Issues Log\n")
        log.write("=====================================\n")

        non_interacting_pdbs = 0
        for pdb_file in tqdm(os.listdir(pdb_dir)):
            pdb_id = pdb_file.split('_')[0]
            cur_rec_pdb = os.path.join(pdb_dir, f'{pdb_id}_receptor_1.pdb')
            cur_ant_pdb = os.path.join(pdb_dir, f'{pdb_id}_antigen_1_1.pdb')

            if os.path.exists(cur_rec_pdb) and os.path.exists(cur_ant_pdb):
                final, rec = locate_receptor_binding_site_atoms(cur_rec_pdb, cur_ant_pdb, distance_cutoff)
                if len(final) == 0:
                    non_interacting_pdbs += 1
                    log.write(f'\nNON-INTERACTING PAIRS!!!: problem with {pdb_id}.pdb. {pdb_id}_receptor_1.pdb and '
                              f' {pdb_id}_antigen_1_1.pdb files are removed.\n')
                    os.remove(cur_rec_pdb)
                    os.remove(cur_ant_pdb)
                    all_successful = False  # Mark as unsuccessful if any issue is found

        # Check if everything was successful
        if all_successful:
            print("Success! All receptors interact with their associated antigens.")
            # since no issue s were found we can remove the log file
            os.remove(log_file)
        else:
            print(f'\n ~~~~~ Total pdbs found with issues: {non_interacting_pdbs} and are removed from the folder ~~~~~\n')
            log.write(f'\n\n ~~~~~ Total pdbs found with issues: {non_interacting_pdbs} ~~~~~')
            print(f"Some receptors do not interact with their antigens. Issues logged in {log_file}.")


# example usage
if __name__ == '__main__':
    user = os.getenv('USER')
    pdb_dir = f'/home/{user}/PycharmProjects/github_projects/ParaSurf/test_data/pdbs/eraseme'
    index = pdb_dir.split('/')[-1]
    check_receptor_antigen_interactions(pdb_dir, distance_cutoff=4.5, log_file=f'{pdb_dir}/{index}_interaction_issues.txt')
