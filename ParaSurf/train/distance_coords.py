import math
import re
import numpy as np

def locate_surface_binding_site_atoms(receptor_surf_file, antigen_pdb_file, distance_cutoff=4):
    rec_coordinates = []
    with open(receptor_surf_file, 'r') as file:
        for line in file:
            parts = line.split()

            # Check for the presence of a numeric value in the 3rd element of parts
            match = re.search(r'([-+]?\d*\.\d+|\d+)(?=\.)', parts[2])
            if match:
                numeric_value = match.group(0)
                non_numeric_value = parts[2].replace(numeric_value, "")

                # Update the 'parts' list
                parts[2:3] = [non_numeric_value, numeric_value]

            if len(parts) >= 7:  # Since we added an extra element to parts, its length increased by 1
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
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


def coords2pdb(coordinates, tosavepath):
    with open(tosavepath, 'w') as pdb_file:
        atom_number = 1
        for coord in coordinates:
            x, y, z = coord
            pdb_file.write(f"ATOM  {atom_number:5}  DUM DUM A{atom_number:4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")

            atom_number += 1
            if atom_number == 9999:
                atom_number = 1
        pdb_file.write("END")


def locate_receptor_binding_site_atoms_residue_level(receptor_file, antigen_pdb_file, distance_cutoff=4):
    rec_atoms = []
    chain_elements = []
    with open(receptor_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                atom_id = line[6:11].strip()
                atom_type = line[12:16].strip()
                res_id = line[22:26].strip()
                # check if there is Code for insertions of residues
                insertion_code = line[26].strip()
                if insertion_code:
                    res_id = res_id + insertion_code
                res_name = line[17:20].strip()
                chain_id = line[21].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                rec_atoms.append((atom_id, atom_type, res_id, res_name, chain_id, x, y, z))
                chain_elements.append((atom_id, atom_type, res_id, chain_id))

    ant_atoms = []
    with open(antigen_pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                atom_id = line[6:11].strip()
                atom_type = line[12:16].strip()
                res_id = line[22:26].strip()
                res_name = line[17:20].strip()
                chain_id = line[21].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                ant_atoms.append((atom_id, atom_type, res_id, res_name, chain_id, x, y, z))

    final_atoms = []

    for rec_atom in rec_atoms:
        for ant_atom in ant_atoms:
            if math.dist(rec_atom[5:], ant_atom[5:]) < distance_cutoff:
                final_atoms.append(rec_atom)
                break

    rec_atoms = np.array([atom[5:] for atom in rec_atoms])
    final_atoms_ = np.array([atom[5:] for atom in final_atoms])
    final_elements = np.array([atom[:5] for atom in final_atoms])

    return final_atoms_, rec_atoms, final_elements


def coords2pdb_residue_level(coordinates, tosavepath, elements):
    with open(tosavepath, 'w') as pdb_file:
        for i, atom in enumerate(coordinates):
            atom_id, atom_type, res_id, res_name, chain_id = elements[i]

            # Separate the numeric part from the insertion code (if any)
            if res_id[-1].isalpha():  # Check if the last character is an insertion code
                res_num = res_id[:-1]  # Numeric part of the residue
                insertion_code = res_id[-1]  # Insertion code (e.g., 'A' in '30A')
            else:
                res_num = res_id
                insertion_code = " "  # No insertion code

            x, y, z = atom

            # Write to the PDB file with the correct formatting
            pdb_file.write(
                f"ATOM  {int(atom_id):5} {atom_type:<4} {res_name} {chain_id}{int(res_num):4}{insertion_code:1}   {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")

        pdb_file.write("END\n")


