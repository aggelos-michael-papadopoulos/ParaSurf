import numpy as np, re
from .rotation import rotation_quaternion
from scipy.cluster.hierarchy import fclusterdata
from sklearn.cluster import MeanShift


def mol2_reader(mol_file):
    if mol_file[-4:] != 'mol2':
        print('cant read no mol2 file')
        return
    
    with open(mol_file,'r') as f:
        lines = f.readlines()
    
    for i,line in enumerate(lines):
        if '@<TRIPOS>ATOM' in line:
            first_atom_idx = i+1
        if '@<TRIPOS>BOND' in line:
            last_atom_idx = i-1
    
    resids = set()    # (resname,resid)
    for line in lines[first_atom_idx:last_atom_idx+1]:
        temp = line.split()[-2]
        resids.add((temp[:3],int(temp[3:])))
    
    return list(resids), lines[first_atom_idx:last_atom_idx+1]


def readSurfpoints(receptor_surf_file):
    try:
        coordinates = []
        normals = []
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
                    coordinates.append([x, y, z])

                if len(parts) >= 11:
                    nx = float(parts[8])
                    ny = float(parts[9])
                    nz = float(parts[10])
                    normals.append([nx, ny, nz])
                else:
                    normals.append([0.0, 0.0, 0.0])

            return np.array(coordinates), np.array(normals)

    except Exception as e:
        raise ValueError(f"Error processing file {receptor_surf_file}. Original error: {str(e)}")


def readSurfpoints_only_receptor_atoms(receptor_surf_file):
    try:
        coordinates = []
        normals = []
        with open(receptor_surf_file, 'r') as file:
            for line in file:
                parts = line.split()

                if len(parts) < 8:
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
                        coordinates.append([x, y, z])

                    if len(parts) >= 11:
                        nx = float(parts[8])
                        ny = float(parts[9])
                        nz = float(parts[10])
                        normals.append([nx, ny, nz])
                    else:
                        normals.append([0.0, 0.0, 0.0])

                else:
                    continue
            return np.array(coordinates), np.array(normals)

    except Exception as e:
        raise ValueError(f"Error processing file {receptor_surf_file}. Original error: {str(e)}")



def readSurfpoints_with_residues(receptor_surf_file):
    residues = {}  # Initialize an empty dictionary to hold residue data
    try:
        with open(receptor_surf_file, 'r') as file:
            line_idx = 0
            for line in file:
                parts = line.split()
                # Check if the line is formatted correctly
                if len(parts) < 6:
                    continue  # Skip lines that are too short

                res_id = parts[1]  # Extract the residue identifier
                if res_id not in residues:
                    residues[res_id] = {'atoms': [], 'normals': [], 'idx': []}  # Initialize data structure for new residues

                # Extract coordinates
                atom_data = [float(parts[3]), float(parts[4]), float(parts[5])]
                residues[res_id]['atoms'].append(atom_data)

                # Extract normals if they exist
                if len(parts) >= 11:
                    normal_data = [float(parts[8]), float(parts[9]), float(parts[10])]
                else:
                    normal_data = [0.0, 0.0, 0.0]

                residues[res_id]['normals'].append(normal_data)
                # add the idx
                residues[res_id]['idx'].append(line_idx)

                line_idx += 1
        return residues

    except Exception as e:
        raise ValueError(f"Error processing file {receptor_surf_file}. Original error: {str(e)}")



def dist_point_from_lig(p, lig_coords):
    """
    Calculates the minimum distance from a point to a set of ligand coordinates.
    """
    return np.min(np.linalg.norm(lig_coords - p, axis=1))



def clustering(data,T,cls_method,bw):
    #data = np.concatenate((points,np.expand_dims(probs,axis=1)),axis=1)    
    T_new = T
    while sum(data[:,-1]>=T_new) < 10 and T_new>0.3001:    # at least 10 points with prob>T  and T>=0.3
        T_new -= 0.1 
    
    data = data[data[:,-1]>T_new]
    if T_new != T:
        print('T changed to', T_new)
    if len(data)<5:
        return np.array([]), np.array([])

    # normalize over [T,1]
    mn = min(data[:,0])   # choose the limits of the 0th dim (w/o reason)
    Mn = max(data[:,0])
    new_data = np.zeros_like(data)
    new_data[:,:-1] = data[:,:-1]
    new_data[:,-1] = (data[:,-1]-T_new)/(1-T_new) * (Mn-mn) + mn
            
    if cls_method == 'ms':
        ms = MeanShift(bandwidth=bw,bin_seeding=True,cluster_all=False,n_jobs=4)
        #clustering = ms.fit(new_data)
        clustering = ms.fit(data[:,:-1])
        labels = clustering.labels_
    elif cls_method == 'fcluster':
        labels = fclusterdata(new_data,t=3,criterion='distance')
        labels -= 1

    return data, labels


class Grid:
    def __init__(self,gridSize,voxelSize):
        grid_limit = (gridSize/2-0.5)*voxelSize
        gridCenters = np.mgrid[-grid_limit:grid_limit+voxelSize:voxelSize,-grid_limit:grid_limit+voxelSize:voxelSize,-grid_limit:grid_limit+voxelSize:voxelSize]
        self.gridCenters = np.resize(gridCenters,(3,gridSize**3))
        self.gridSize = gridSize
        self.radius = grid_limit*np.sqrt(3)

    def make_grid(self,point,normal,rotate_grid):
        if rotate_grid:
            Q = rotation_quaternion(normal)
            centers = np.matmul(Q,self.gridCenters)
            centers = np.transpose(centers)
        else:
            centers = np.transpose(self.gridCenters)
        centers = np.add(centers,point)
        return centers
    
    def get_protrusion(self,centers,coords,radius):
        close = np.zeros((len(centers),len(coords)),dtype=bool)    
        for c,coord in enumerate(coords):
            close[:,c] = np.sqrt(np.sum((centers-coord)**2,axis=1))<radius
               
        return np.expand_dims(np.count_nonzero(close,axis=1),axis=-1) / 200.0    # normalization


class BindingSite:
    def __init__(self,coords,prob):
        self.coords = coords
        self.center = np.average(self.coords,axis=0)
        self.prob = prob

if __name__ == "__main__":
    c, n = readSurfpoints('/home/angepapa/PycharmProjects/DeepSurf2.0/eraseme/3eck_receptor_1.surfpoints')
    coord, noorms = readSurfpoints('/home/angepapa/PycharmProjects/DeepSurf2.0/eraseme/3eck_receptor_1.surfpoints')