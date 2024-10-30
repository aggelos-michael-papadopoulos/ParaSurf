import os


def extract_chains_from_pdb(pdb_file, output_dir):
    """
    Extract and save the chains from a PDB file as separate chain-specific PDB files.

    Args:
        pdb_file (str): Path to the PDB file.
        output_dir (str): Path to the directory where the chain-specific files should be saved.

    Returns:
        list: Paths to the chain-specific PDB files.
    """
    chain_dict = {}

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                chain_id = line[21]
                if chain_id in chain_dict:
                    chain_dict[chain_id].append(line)
                else:
                    chain_dict[chain_id] = [line]

    chain_files = []
    for chain_id, lines in chain_dict.items():
        chain_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pdb_file))[0]}_chain{chain_id}.pdb")
        with open(chain_file, 'w') as f:
            f.writelines(lines)
        # print(f'Chain {chain_id} saved as {chain_file}.')
        chain_files.append(chain_file)

    chain_ids = [chain.split("/")[-1].split(".")[0][-1] for chain in chain_files]

    return chain_files, chain_ids

if __name__ == '__main__':
    pdb_file = '/home/angepapa/PycharmProjects/DeepSurf2.0/3bgf.pdb'
    output_dir = "/".join(pdb_file.split('/')[:-1])
    chain_files, chain_ids = extract_chains_from_pdb(pdb_file, output_dir)
    print(chain_files)
    print(chain_ids)
