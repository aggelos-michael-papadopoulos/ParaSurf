import os
import math
import re
import numpy as np
from Bio.PDB.PDBParser import PDBParser


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


def locate_receptor_binding_site_residues(receptor_file, antigen_pdb_file, distance_cutoff=4):
    rec_atoms = []
    with open(receptor_file, 'r') as file:
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
                rec_atoms.append((atom_id, atom_type, res_id, res_name, chain_id, x, y, z))

        parser = PDBParser(
            PERMISSIVE=1)  # PERMISSIVE=1 allowing more flexibility in handling non-standard or problematic entries in PDB files during parsing.
        lig = parser.get_structure('antigen', antigen_pdb_file)

        ant_atoms = np.array([atom.get_coord() for atom in lig.get_atoms()])

        bind_site_residues = []

        for rec_atom in rec_atoms:

            for ant_atom in ant_atoms:
                if math.dist(rec_atom[5:], ant_atom) < distance_cutoff:
                    bind_site_residues.append(rec_atom[2]+rec_atom[4]) #res_id + chain_id
                    break

        return set(list(bind_site_residues))


def locate_receptor_binding_site_atoms_residue_level(receptor_file, antigen_pdb_file, distance_cutoff=4):
    rec_atoms = []
    with open(receptor_file, 'r') as file:
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
                rec_atoms.append((atom_id, atom_type, res_id, res_name, chain_id, x, y, z))

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
            x, y, z = atom
            pdb_file.write(
                f"ATOM  {int(atom_id):5} {atom_type:<4} {res_name} {chain_id}{int(res_id):4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")

        pdb_file.write("END\n")



def locate_surface_binding_site_atoms_surface_case(receptor_surf_file, antigen_pdb_file, distance_cutoff=4):
    # Assuming receptor surface file format allows extraction of atom and residue names
    rec_coordinates = []
    with open(receptor_surf_file, 'r') as file:
        for line in file:
            parts = line.split()
            # Adjust based on actual receptor surface file format to extract atom and residue names
            if len(parts) > 7:  # Assuming there's a format to follow
                res_id = parts[1][:-1]#.split('')[0]  # Placeholder, adjust as necessary
                atom_type = parts[2]
                chain_id = parts[1][-1]
                residue_name = parts[0]  # Placeholder, adjust as necessary
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
                rec_coordinates.append((atom_type, res_id, residue_name, chain_id, x, y, z))

    ant_coordinates = []
    with open(antigen_pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                ant_coordinates.append((x, y, z, atom_name, residue_name))

    final_coordinates = []
    for rec_atom in rec_coordinates:
        for ant_atom in ant_coordinates:
            if math.dist(rec_atom[4:], ant_atom[:3]) < distance_cutoff:
                final_coordinates.append(rec_atom)
                break

    rec_atoms = np.array([atom[4:] for atom in rec_coordinates])
    final_atoms = np.array([atom[4:] for atom in final_coordinates])
    final_elements = np.array([atom[:4] for atom in final_coordinates])

    return final_atoms, rec_atoms, final_elements


def coords2pdb_surface_case(coordinates, tosavepath, elements):
    with open(tosavepath, 'w') as pdb_file:
        atom_number = 1
        for i, atom in enumerate(coordinates):
            atom_type, res_id, res_name, chain_id = elements[i]
            x, y, z = atom
            pdb_file.write(
                f"ATOM  {int(atom_number):5} {atom_type:<4} {res_name} {chain_id}{int(res_id):4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")

            atom_number += 1
            if atom_number == 9999:
                atom_number = 1
        pdb_file.write("END\n")


def keep_only_surface_atoms(surfpoint_file, min_line_length=60):
    """
    Simple technique to keep only the surface atoms and delete the original receptor points
    Filters out lines from the input file that are shorter than min_line_length
    and writes the remaining lines to the output file.

    Parameters:
    - surfpoint_file: The path to the input-output file to process.
    - min_line_length: The minimum length of lines to keep.
    """
    filtered_lines = []  # A list to hold the filtered lines

    # Open and read the file
    with open(surfpoint_file, "r") as file:
        for line in file:
            if len(line.strip()) >= min_line_length:
                filtered_lines.append(line.strip())

    # Write the filtered lines to the output file
    with open(surfpoint_file, "w") as output_file:
        for line in filtered_lines:
            output_file.write(f"{line}\n")


if __name__ =='__main__':
    receptor_surf_file = '/home/angepapa/PycharmProjects/DeepSurf2.0/eraseme/1Z0K_receptor_1.surfpoints'
    receptor_pdb_file = '/home/angepapa/PycharmProjects/DeepSurf2.0/eraseme/1Z0K_receptor_1.pdb'
    antigen_pdb_file = '/home/angepapa/PycharmProjects/DeepSurf2.0/eraseme/1Z0K_antigen_1_1.pdb'

    # new
    receptor_binding_site_coordinates,  all_atoms, elements = locate_receptor_binding_site_atoms_residue_level(receptor_pdb_file, antigen_pdb_file, distance_cutoff=4)
    coords2pdb_residue_level(receptor_binding_site_coordinates,
                             '/home/angepapa/PycharmProjects/DeepSurf2.0/xxxxxx.pdb', elements)
