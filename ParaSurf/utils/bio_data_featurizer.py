import jsonpickle
import os, re
import numpy as np
# import pybel
from openbabel import pybel
from math import ceil, sin, cos, sqrt, pi
from itertools import combinations
import collections
import subprocess



class Featurizer():
    """Calculates atomic features for molecules. Features can encode atom type,
    native pybel properties or any property defined with SMARTS patterns

    Attributes
    ----------
    FEATURE_NAMES: list of strings
        Labels for features (in the same order as features)
    NUM_ATOM_CLASSES: int
        Number of atom codes
    ATOM_CODES: dict
        Dictionary mapping atomic numbers to codes
    NAMED_PROPS: list of string
        Names of atomic properties to retrieve from pybel.Atom object
    CALLABLES: list of callables
        Callables used to calculcate custom atomic properties
    SMARTS: list of SMARTS strings
        SMARTS patterns defining additional atomic properties
    """

    def __init__(self, atom_codes=None, atom_labels=None,
                 named_properties=None, save_molecule_codes=True,
                 custom_properties=None, smarts_properties=None,
                 smarts_labels=None, add_force_fields=False):

        """Creates Featurizer with specified types of features. Elements of a
        feature vector will be in a following order: atom type encoding
        (defined by atom_codes), Pybel atomic properties (defined by
        named_properties), molecule code (if present), custom atomic properties
        (defined `custom_properties`), and additional properties defined with
        SMARTS (defined with `smarts_properties`).

        Parameters
        ----------
        atom_codes: dict, optional
            Dictionary mapping atomic numbers to codes. It will be used for
            one-hot encoging therefore if n different types are used, codes
            shpuld be from 0 to n-1. Multiple atoms can have the same code,
            e.g. you can use {6: 0, 7: 1, 8: 1} to encode carbons with [1, 0]
            and nitrogens and oxygens with [0, 1] vectors. If not provided,
            default encoding is used.
        atom_labels: list of strings, optional
            Labels for atoms codes. It should have the same length as the
            number of used codes, e.g. for `atom_codes={6: 0, 7: 1, 8: 1}` you
            should provide something like ['C', 'O or N']. If not specified
            labels 'atom0', 'atom1' etc are used. If `atom_codes` is not
            specified this argument is ignored.
        named_properties: list of strings, optional
            Names of atomic properties to retrieve from pybel.Atom object. If
            not specified ['hyb', 'heavyvalence', 'heterovalence',
            'partialcharge'] is used.
        save_molecule_codes: bool, optional (default True)
            If set to True, there will be an additional feature to save
            molecule code. It is usefeul when saving molecular complex in a
            single array.
        custom_properties: list of callables, optional
            Custom functions to calculate atomic properties. Each element of
            this list should be a callable that takes pybel.Atom object and
            returns a float. If callable has `__name__` property it is used as
            feature label. Otherwise labels 'func<i>' etc are used, where i is
            the index in `custom_properties` list.
        smarts_properties: list of strings, optional
            Additional atomic properties defined with SMARTS patterns. These
            patterns should match a single atom. If not specified, deafult
            patterns are used.
        smarts_labels: list of strings, optional
            Labels for properties defined with SMARTS. Should have the same
            length as `smarts_properties`. If not specified labels 'smarts0',
            'smarts1' etc are used. If `smarts_properties` is not specified
            this argument is ignored.
        """

        # Remember namse of all features in the correct order
        self.FEATURE_NAMES = []

        if atom_codes is not None:
            if not isinstance(atom_codes, dict):
                raise TypeError('Atom codes should be dict, got %s instead'
                                % type(atom_codes))
            codes = set(atom_codes.values())
            for i in range(len(codes)):
                if i not in codes:
                    raise ValueError('Incorrect atom code %s' % i)

            self.NUM_ATOM_CLASSES = len(codes)
            self.ATOM_CODES = atom_codes
            if atom_labels is not None:
                if len(atom_labels) != self.NUM_ATOM_CLASSES:
                    raise ValueError('Incorrect number of atom labels: '
                                     '%s instead of %s'
                                     % (len(atom_labels), self.NUM_ATOM_CLASSES))
            else:
                atom_labels = ['atom%s' % i for i in range(self.NUM_ATOM_CLASSES)]
            self.FEATURE_NAMES += atom_labels
        else:
            self.ATOM_CODES = {}

            metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                      + list(range(37, 51)) + list(range(55, 84))
                      + list(range(87, 104)))

            # List of tuples (atomic_num, class_name) with atom types to encode.
            atom_classes = [
                (5, 'B'),
                (6, 'C'),
                (7, 'N'),
                (8, 'O'),
                (15, 'P'),
                (16, 'S'),
                (34, 'Se'),
                ([9, 17, 35, 53], 'halogen'),
                (metals, 'metal')
            ]

            for code, (atom, name) in enumerate(atom_classes):
                if type(atom) is list:
                    for a in atom:
                        self.ATOM_CODES[a] = code
                else:
                    self.ATOM_CODES[atom] = code
                self.FEATURE_NAMES.append(name)

            self.NUM_ATOM_CLASSES = len(atom_classes)

        if named_properties is not None:
            if not isinstance(named_properties, (list, tuple, np.ndarray)):
                raise TypeError('named_properties must be a list')
            allowed_props = [prop for prop in dir(pybel.Atom)
                             if not prop.startswith('__')]
            for prop_id, prop in enumerate(named_properties):
                if prop not in allowed_props:
                    raise ValueError(
                        'named_properties must be in pybel.Atom attributes,'
                        ' %s was given at position %s' % (prop_id, prop)
                    )
            self.NAMED_PROPS = named_properties
            self.FORCE_FIELDS = named_properties
        else:
            # # pybel.Atom properties to save
            # self.NAMED_PROPS = ['hyb', 'heavyvalence', 'heterovalence',
            #                     'partialcharge'] # without  , 'implicitvalence', 'exactmass'

            self.NAMED_PROPS = ['hyb', 'heavydegree', 'heterodegree',
                                'partialcharge']# without  , 'implicitvalence', 'exactmass'                 # for openbabel 3.1.1

            self.FORCE_FIELDS = ['AMBER','PARSE'] # 'AMBER', 'CHARMM', 'PARSE', 'PEOEPB', 'SWANSON', 'TYL06'
        self.FEATURE_NAMES += self.NAMED_PROPS

        if add_force_fields:
            self.FEATURE_NAMES += self.FORCE_FIELDS
        if not isinstance(save_molecule_codes, bool):
            raise TypeError('save_molecule_codes should be bool, got %s '
                            'instead' % type(save_molecule_codes))
        self.save_molecule_codes = save_molecule_codes
        if save_molecule_codes:
            # Remember if an atom belongs to the ligand or to the protein
            self.FEATURE_NAMES.append('molcode')

        self.CALLABLES = []
        if custom_properties is not None:
            for i, func in enumerate(custom_properties):
                if not isinstance(func, collections.Callable):
                    raise TypeError('custom_properties should be list of'
                                    ' callables, got %s instead' % type(func))
                name = getattr(func, '__name__', '')
                if name == '':
                    name = 'func%s' % i
                self.CALLABLES.append(func)
                self.FEATURE_NAMES.append(name)

        if smarts_properties is None:
            # SMARTS definition for other properties
            self.SMARTS = [
                '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
                '[a]',
                '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
                '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
                '[r]'
            ]
            smarts_labels = ['hydrophobic', 'aromatic', 'acceptor', 'donor',
                             'ring']
        elif not isinstance(smarts_properties, (list, tuple, np.ndarray)):
            raise TypeError('smarts_properties must be a list')
        else:
            self.SMARTS = smarts_properties

        if smarts_labels is not None:
            if len(smarts_labels) != len(self.SMARTS):
                raise ValueError('Incorrect number of SMARTS labels: %s'
                                 ' instead of %s'
                                 % (len(smarts_labels), len(self.SMARTS)))
        else:
            smarts_labels = ['smarts%s' % i for i in range(len(self.SMARTS))]

        # Compile patterns
        self.compile_smarts()
        self.FEATURE_NAMES += smarts_labels
        x=1

    def compile_smarts(self):
        self.__PATTERNS = []
        for smarts in self.SMARTS:
            self.__PATTERNS.append(pybel.Smarts(smarts))

    def encode_num(self, atomic_num):
        """Encode atom type with a binary vector. If atom type is not included in
        the `atom_classes`, its encoding is an all-zeros vector.

        Parameters
        ----------
        atomic_num: int
            Atomic number

        Returns
        -------
        encoding: np.ndarray
            Binary vector encoding atom type (one-hot or null).
        """

        if not isinstance(atomic_num, int):
            raise TypeError('Atomic number must be int, %s was given'
                            % type(atomic_num))

        encoding = np.zeros(self.NUM_ATOM_CLASSES)
        try:
            encoding[self.ATOM_CODES[atomic_num]] = 1.0
        except:
            pass
        return encoding

    def find_smarts(self, molecule):
        """Find atoms that match SMARTS patterns.

        Parameters
        ----------
        molecule: pybel.Molecule

        Returns
        -------
        features: np.ndarray
            NxM binary array, where N is the number of atoms in the `molecule`
            and M is the number of patterns. `features[i, j]` == 1.0 if i'th
            atom has j'th property
        """

        if not isinstance(molecule, pybel.Molecule):
            raise TypeError('molecule must be pybel.Molecule object, %s was given'
                            % type(molecule))
       # print 'Here'
        #print len(self.__PATTERNS)
        features = np.zeros((len(molecule.atoms), len(self.__PATTERNS)))
        #print 'Here'
        for (pattern_id, pattern) in enumerate(self.__PATTERNS):
            atoms_with_prop = np.array(list(*list(zip(*pattern.findall(molecule)))),
                                       dtype=int) - 1
            features[atoms_with_prop, pattern_id] = 1.0
        return features

    def get_features(self, molecule, molcode=None):
        """Get coordinates and features for all heavy atoms in the molecule.

        Parameters
        ----------
        molecule: pybel.Molecule
        molcode: float, optional
            Molecule type. You can use it to encode whether an atom belongs to
            the ligand (1.0) or to the protein (-1.0) etc.

        Returns
        -------
        coords: np.ndarray, shape = (N, 3)
            Coordinates of all heavy atoms in the `molecule`.
        features: np.ndarray, shape = (N, F)
            Features of all heavy atoms in the `molecule`: atom type
            (one-hot encoding), pybel.Atom attributes, type of a molecule
            (e.g protein/ligand distinction), and other properties defined with
            SMARTS patterns
        """

        if not isinstance(molecule, pybel.Molecule):
            raise TypeError('molecule must be pybel.Molecule object,'
                            ' %s was given' % type(molecule))
        if molcode is None:
            if self.save_molecule_codes is True:
                raise ValueError('save_molecule_codes is set to True,'
                                 ' you must specify code for the molecule')
        elif not isinstance(molcode, (float, int)):
            raise TypeError('motlype must be float, %s was given'
                            % type(molcode))

        coords = []
        features = []
        heavy_atoms = []
        #print 'nAtoms=',len(molecule.atoms)
        for i, atom in enumerate(molecule):
            # ignore hydrogens and dummy atoms (they have atomicnum set to 0)
            if atom.atomicnum > 1:
                heavy_atoms.append(i)
                coords.append(atom.coords)

                features.append(np.concatenate((
                    self.encode_num(atom.atomicnum),
                    [round(atom.__getattribute__(prop),7) * 0.1 if prop == 'exactmass' else atom.__getattribute__(prop) for prop in self.NAMED_PROPS],
                    [func(atom) for func in self.CALLABLES],
                )))


        #print 'nAtoms=',len(molecule.atoms)
        coords = np.array(coords, dtype=np.float32)
        features = np.array(features, dtype=np.float32)

        if self.save_molecule_codes:
            features = np.hstack((features, molcode * np.ones((len(features), 1)))) # here the features become from 13 --> 18

        features = np.hstack([features, self.find_smarts(molecule)[heavy_atoms]])

        if np.isnan(features).any():
            raise RuntimeError('Got NaN when calculating features')

        # make z-score normalization over all features
        # features_z_score_normalized = []
        # for feature_vector in features:
        #     mean = np.mean(feature_vector)
        #     std = np.std(feature_vector)
        #     # Perform Z-score normalization
        #     normalized_vector = (feature_vector - mean) / std
        #     rounded_normalized_vector = np.round(normalized_vector, 8)
        #     features_z_score_normalized.append(rounded_normalized_vector)

        # features = np.array(features_z_score_normalized)
        x=1
        return coords, features

    def to_pickle(self, fname='featurizer.pkl'):
        """Save featurizer in a given file using JSON format.

        Parameters
        ----------
        fname: str, optional
           Path to file in which featurizer will be saved
        """

        with open(fname, 'w') as f:
            json_str = jsonpickle.encode(self)
            f.write(json_str)

    @staticmethod
    def from_pickle(fname):
        """Load pickled featurizer from a given file using JSON format.

        Parameters
        ----------
        fname: str, optional
           Path to file with saved featurizer

        Returns
        -------
        featurizer: Featurizer object
           Loaded featurizer
        """
        with open(fname, 'r') as f:
            json_str = f.read()
            featurizer = jsonpickle.decode(json_str)
        featurizer.compile_smarts()
        return featurizer

    def get_features_with_force_fields(self, molecule, molcode=None, add_atom_radius=False):
        """Get coordinates and features for all heavy atoms in the molecule.

        Parameters
        ----------
        molecule: pybel.Molecule
        molcode: float, optional
            Molecule type. You can use it to encode whether an atom belongs to
            the ligand (1.0) or to the protein (-1.0) etc.

        Returns
        -------
        coords: np.ndarray, shape = (N, 3)
            Coordinates of all heavy atoms in the `molecule`.
        features: np.ndarray, shape = (N, F)
            Features of all heavy atoms in the `molecule`: atom type
            (one-hot encoding), pybel.Atom attributes, type of a molecule
            (e.g protein/ligand distinction), and other properties defined with
            SMARTS patterns
        """

        if not isinstance(molecule, pybel.Molecule):
            raise TypeError('molecule must be pybel.Molecule object,'
                            ' %s was given' % type(molecule))
        if molcode is None:
            if self.save_molecule_codes is True:
                raise ValueError('save_molecule_codes is set to True,'
                                 ' you must specify code for the molecule')
        elif not isinstance(molcode, (float, int)):
            raise TypeError('motlype must be float, %s was given'
                            % type(molcode))

        coords = []
        features = []
        heavy_atoms = []
        #print 'nAtoms=',len(molecule.atoms)
        for i, atom in enumerate(molecule):
            # ignore hydrogens and dummy atoms (they have atomicnum set to 0)
            if atom.atomicnum > 1:
                heavy_atoms.append(i)
                coords.append(atom.coords)

                features.append(np.concatenate((
                    self.encode_num(atom.atomicnum),
                    [round(atom.__getattribute__(prop),7) * 0.1 if prop == 'exactmass' else atom.__getattribute__(prop) for prop in self.NAMED_PROPS],
                    [func(atom) for func in self.CALLABLES],
                )))

        #print 'nAtoms=',len(molecule.atoms)
        coords = np.array(coords, dtype=np.float32)
        features = np.array(features, dtype=np.float32)

        if self.save_molecule_codes:
            features = np.hstack((features, molcode * np.ones((len(features), 1)))) # here the features become from 13 --> 18

        features = np.hstack([features, self.find_smarts(molecule)[heavy_atoms]])

        if np.isnan(features).any():
            raise RuntimeError('Got NaN when calculating features')


        #### ADD FORCE FIELDS ##########
        # Define the relative path to `pdb2pqr` based on the repository root
        def get_pdb2pqr_path():
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))

            # Try locating `pdb2pqr` by going up directory levels
            possible_root_paths = [
                os.path.join(script_dir, '..'),  # For blind_predict.py
                os.path.join(script_dir, '../..'),  # For ParaSurf/preprocess scripts
            ]

            for root in possible_root_paths:
                pdb2pqr_path = os.path.abspath(os.path.join(root, 'pdb2pqr-linux-bin64-2.1.1', 'pdb2pqr'))
                if os.path.isfile(pdb2pqr_path):
                    return pdb2pqr_path

            # If not found, raise an error
            raise FileNotFoundError("pdb2pqr executable not found in expected directories.")

        # Usage in the code
        pdb2pqr_software_path = get_pdb2pqr_path()

        # Ensure the path is valid
        if not os.path.isfile(pdb2pqr_software_path):
            raise FileNotFoundError(f"pdb2pqr not found at {pdb2pqr_software_path}")

        cur_pdb = molecule.title
        ff_names = self.FORCE_FIELDS

        all_ff = []
        all_atom_radius = []
        for force_field in ff_names:
            out_pqr_file = '/tmp/' + molecule.title.split('/')[-1].split('_')[0] + f'_{force_field}.pqr'
            command = [
                pdb2pqr_software_path,
                '--ff=' + force_field,
                cur_pdb,
                out_pqr_file
            ]

            print(f'calcualting Force Field: {force_field}...\n')
            # Run the command
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("pdb2pqr ran successfully:", result.stdout)
            except subprocess.CalledProcessError as e:
                print("An error occurred while running pdb2pqr:", e.stderr)

            cur_ff, cur_atoms_radius = extract_ff_values(cur_pdb, out_pqr_file, add_ff_atom_radius=add_atom_radius)
            all_ff.append(cur_ff)

            if add_atom_radius:
                all_atom_radius.append(cur_atoms_radius)

            # remove the created pqr file from ./tmp
            os.remove(out_pqr_file)


        final_ff_list = np.column_stack(all_ff)

        features = np.hstack((features, final_ff_list))
        if add_atom_radius:
            final_atom_radius_list = np.column_stack((all_atom_radius))
            features = np.hstack((features, final_atom_radius_list))
        # make z-score normalization over all features
        # # Indices to normalize
        # indices_to_normalize = [18, 19]           # which features you want to normalize?
        # # Calculate mean and std for each selected feature across all samples
        # means = features[:, indices_to_normalize].mean(axis=0)
        # stds = features[:, indices_to_normalize].std(axis=0)
        # # Apply Z-score normalization only to the specific indices
        # features[:, indices_to_normalize] = (features[:, indices_to_normalize] - means) / stds

        # features = np.array(features_z_score_normalized)
        x=1
        return coords, features

    def to_pickle(self, fname='featurizer.pkl'):
        """Save featurizer in a given file using JSON format.

        Parameters
        ----------
        fname: str, optional
           Path to file in which featurizer will be saved
        """

        with open(fname, 'w') as f:
            json_str = jsonpickle.encode(self)
            f.write(json_str)


    @staticmethod
    def from_pickle(fname):
        """Load pickled featurizer from a given file using JSON format.

        Parameters
        ----------
        fname: str, optional
           Path to file with saved featurizer

        Returns
        -------
        featurizer: Featurizer object
           Loaded featurizer
        """
        with open(fname, 'r') as f:
            json_str = f.read()
            featurizer = jsonpickle.decode(json_str)
        featurizer.compile_smarts()
        return featurizer

def extract_ff_values(pdb_file, pqr_file, add_ff_atom_radius=False):
    def separate_coordinates(coord_str):
        # Use a regular expression to handle concatenated negative signs
        return re.sub(r'(?<=\d)(-)', r' \1', coord_str)

    pdb_coords = []

    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith('ATOM'):
                try:
                    # Extracting x, y, z from fixed-width fields
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    pdb_coords.append((x, y, z))
                except ValueError:
                    # Use separate_coordinates if there is a ValueError
                    corrected_str = separate_coordinates(line[30:54])
                    x, y, z = map(float, corrected_str.split())
                    pdb_coords.append((x, y, z))

    # Read PQR coordinates and charges into a dictionary for faster access
    pqr_dict = {}
    atom_radius_dict = {}
    with open(pqr_file, 'r') as file:
        for line in file:
            if line.startswith('ATOM'):
                parts = line.split()
                try:
                    # Attempt to parse the coordinates directly
                    x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
                    charge = float(parts[8])
                    atom_radius = float(parts[9])
                except ValueError:
                    # If there's a ValueError, there might be concatenated negative numbers
                    # Use the separate_coordinates function to correct the string
                    corrected_str = separate_coordinates(' '.join(parts[5:7])) # todo change thhis maybe
                    x, y, z = map(float, corrected_str.split())
                    charge = float(parts[7])    # todo change thhis maybe
                    atom_radius = float(parts[8])

                pqr_dict[(x, y, z)] = charge  # Use coordinates as keys
                if add_ff_atom_radius:
                    atom_radius_dict[(x, y, z)] = atom_radius

    # Initialize charges list with zeros
    ff = [0.0] * len(pdb_coords)
    final_atom_radius = [0.0] * len(pdb_coords)

    # Match PQR coordinates with PDB coordinates and assign charges
    for i, coord in enumerate(pdb_coords):
        if coord in pqr_dict:
            ff[i] = pqr_dict[coord]  # Directly assign charge from dictionary

            if add_ff_atom_radius:
                final_atom_radius[i] = atom_radius_dict[coord]
        elif i > 0:
            ff[i] = ff[i - 1]  # Assign the charge of the previous atom if no match is found

    if add_ff_atom_radius:
        return (ff, final_atom_radius)
    else:
        return (ff, None)



def rotation_matrix(axis, theta):
    """Counterclockwise rotation about a given axis by theta radians"""

    if not isinstance(axis, (np.ndarray, list, tuple)):
        raise TypeError('axis must be an array of floats of shape (3,)')
    try:
        axis = np.asarray(axis, dtype=np.float32)
    except ValueError:
        raise ValueError('axis must be an array of floats of shape (3,)')

    if axis.shape != (3,):
        raise ValueError('axis must be an array of floats of shape (3,)')

    if not isinstance(theta, (float, int)):
        raise TypeError('theta must be a float')

    axis = axis / sqrt(np.dot(axis, axis))
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


# Create matrices for all possible 90* rotations of a box
ROTATIONS = [rotation_matrix([1, 1, 1], 0)]

# about X, Y and Z - 9 rotations
for a1 in range(3):
    for t in range(1, 4):
        axis = np.zeros(3)
        axis[a1] = 1
        theta = t * pi / 2.0
        ROTATIONS.append(rotation_matrix(axis, theta))

# about each face diagonal - 6 rotations
for (a1, a2) in combinations(list(range(3)), 2):
    axis = np.zeros(3)
    axis[[a1, a2]] = 1.0
    theta = pi
    ROTATIONS.append(rotation_matrix(axis, theta))
    axis[a2] = -1.0
    ROTATIONS.append(rotation_matrix(axis, theta))

# about each space diagonal - 8 rotations
for t in [1, 2]:
    theta = t * 2 * pi / 3
    axis = np.ones(3)
    ROTATIONS.append(rotation_matrix(axis, theta))
    for a1 in range(3):
        axis = np.ones(3)
        axis[a1] = -1
        ROTATIONS.append(rotation_matrix(axis, theta))


def rotate(coords, rotation):
    """Rotate coordinates by a given rotation

    Parameters
    ----------
    coords: array-like, shape (N, 3)
        Arrays with coordinates and features for each atoms.
    rotation: int or array-like, shape (3, 3)
        Rotation to perform. You can either select predefined rotation by
        giving its index or specify rotation matrix.

    Returns
    -------
    coords: np.ndarray, shape = (N, 3)
        Rotated coordinates.
    """

    global ROTATIONS

    if not isinstance(coords, (np.ndarray, list, tuple)):
        raise TypeError('coords must be an array of floats of shape (N, 3)')
    try:
        coords = np.asarray(coords, dtype=np.float32)
    except ValueError:
        raise ValueError('coords must be an array of floats of shape (N, 3)')
    shape = coords.shape
    if len(shape) != 2 or shape[1] != 3:
        raise ValueError('coords must be an array of floats of shape (N, 3)')

    if isinstance(rotation, int):
        if rotation >= 0 and rotation < len(ROTATIONS):
            return np.dot(coords, ROTATIONS[rotation])
        else:
            raise ValueError('Invalid rotation number %s!' % rotation)
    elif isinstance(rotation, np.ndarray) and rotation.shape == (3, 3):
        return np.dot(coords, rotation)

    else:
        raise ValueError('Invalid rotation %s!' % rotation)


# TODO: add make_grid variant for GPU

def make_grid(coords, features, grid_resolution=1.0, max_dist=10.0):
    """Convert atom coordinates and features represented as 2D arrays into a
    fixed-sized 3D box.

    Parameters
    ----------
    coords, features: array-likes, shape (N, 3) and (N, F)
        Arrays with coordinates and features for each atoms.
    grid_resolution: float, optional
        Resolution of a grid (in Angstroms).
    max_dist: float, optional
        Maximum distance between atom and box center. Resulting box has size of
        2*`max_dist`+1 Angstroms and atoms that are too far away are not
        included.

    Returns
    -------
    coords: np.ndarray, shape = (M, M, M, F)
        4D array with atom properties distributed in 3D space. M is equal to
        2 * `max_dist` / `grid_resolution` + 1
    """

    try:
        coords = np.asarray(coords, dtype=np.float32)
    except ValueError:
        raise ValueError('coords must be an array of floats of shape (N, 3)')
    c_shape = coords.shape
    if len(c_shape) != 2 or c_shape[1] != 3:
        raise ValueError('coords must be an array of floats of shape (N, 3)')
   # print 'coods shape', c_shape
    N = len(coords)
    try:
        features = np.asarray(features, dtype=np.float32)
    except ValueError:
        raise ValueError('features must be an array of floats of shape (N, F)')
    f_shape = features.shape
    if len(f_shape) != 2 or f_shape[0] != N:
        raise ValueError('features must be an array of floats of shape (N, F)')
    #print 'features shape', f_shape
    if not isinstance(grid_resolution, (float, int)):
        raise TypeError('grid_resolution must be float')
    if grid_resolution <= 0:
        raise ValueError('grid_resolution must be positive')

    if not isinstance(max_dist, (float, int)):
        raise TypeError('max_dist must be float')
    if max_dist <= 0:
        raise ValueError('max_dist must be positive')

    num_features = f_shape[1]
    max_dist = float(max_dist)
    grid_resolution = float(grid_resolution)
    #print 'max_dist =', max_dist
    #print 'grid_resolution =', grid_resolution
    box_size = int(ceil(2 * max_dist / grid_resolution + 1))
   # print 'box_size =', box_size
    # move all atoms to the nearest grid point
    grid_coords = (coords + max_dist) / grid_resolution
    grid_coords = grid_coords.round().astype(int)

    # remove atoms outside the box
    in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)
    grid = np.zeros((1, box_size, box_size, box_size, num_features),dtype=np.float32)
    for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
        grid[0, x, y, z] += f

    return grid
