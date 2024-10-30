import os
from tqdm import tqdm


def fix_line(line):
    parts = line.split()
    new_parts = []
    for part in parts:
        if '-' in part and any(c.isalpha() for c in part.split('-')[0]) and part.split('-')[1].replace('.', '', 1).isdigit():
            idx = part.find('-')
            new_parts.append(part[:idx])
            new_parts.append(part[idx:])
        else:
            new_parts.append(part)
    return new_parts


def process_surfpoints_directory(directory):
    """
    Processes all .surfpoints files in a directory, fixes the formatting of each line, and saves the corrected file back to the same path.
    e.g. ['ARG', '40H', 'CD-108.775', '22.706', '95.764', 'SR0', '0.347', '-0.118', '0.364', '-0.924'] -->
    ['ARG', '40H', 'CD', '-108.775', '22.706', '95.764', 'SR0', '0.347', '-0.118', '0.364', '-0.924']

    Args:
    directory (str): The path to the directory containing the .surfpoints files.
    """
    # Iterate through all files in the given directory
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".surfpoints"):
            file_path = os.path.join(directory, filename)
            corrected_lines = []

            # Read the file and process each line
            with open(file_path, 'r') as infile:
                for line in infile:
                    corrected_parts = fix_line(line)
                    corrected_lines.append(' '.join(corrected_parts))

            # Write the corrected lines back to the same file
            with open(file_path, 'w') as outfile:
                for line in corrected_lines:
                    outfile.write(line + '\n')

    print('\n All .surfpoints files have been checked and are ready !!!')

# Usage example:
if __name__ == "__main__":
    directory_path = '/home/angepapa/PycharmProjects/DeepSurf2.0/test_data/surf_points/PECAN/TRAIN'
    process_surfpoints_directory(directory_path)
