# %% Imports
import time
from itertools import combinations

import pandas as pd
import numpy as np

from scipy.spatial.distance import pdist, squareform


#import seaborn as sns
#import matplotlib.pyplot as plt
# %% Number of hops and number of neighbors to include in features
path_hops = 4
neis_hops = 2

# Lattice parameters
lp = [[ 7.4368090630,0.0000000000,0.0000000000],
      [0.0000000000,7.7957539558,0.0000000000],
      [-7.4039145041,0.0000000000,8.0082301811]]

# %% Load data extracted from calculations 
data = pd.read_parquet('./data-MoO3.parquet')

# %% Data organization
data['Structure'] = data.index.str.split('--').str[1]#.astype(int)
data['Atom'] = data.index.str.split('--').str[0]#.str.join('_')
data.index.name = 'Index'
structures = data['Structure'].unique()


# %% 

def pbc(v, lp, atoms, ns_x, ns_y, ns_z): # Periodic bondary conditions function
    '''
    Generate a PBC image of the lattice.
    Inputs:
    v - original coordinates
    lp - lattice parameters
    atoms - list of atoms in the structure
    ns_x, ns_y, ns_z - number of images in x, y and z

    Outputs:
    l - PBC coordinates
    atoms_list - PBC atoms
    '''

    # Define the ranges of periodic images for each dimension.
    ns_x = range(-ns_x, ns_x+1)
    ns_y = range(-ns_y, ns_y+1)
    ns_z = range(-ns_z, ns_z+1)

    # Initialize empty lists to store replicated atomic positions and names.
    l = []
    atoms_list = []

    # Loop over the periodic images in each dimension.
    for i_x in ns_x:
        for i_y in ns_y:
            for i_z in ns_z:
                # Create a name for the current periodic image using the format (i_x, i_y, i_z).
                name = '({},{},{})'.format(i_x, i_y, i_z)

                # Make a copy of the original atomic positions array.
                v_tmp = v.copy()

                # Apply the periodic boundary conditions by translating the atomic positions.
                v_translated = np.dot(lp, [i_x, i_y, i_z])
                v_tmp += v_translated

                # Generate names for the atoms in the current periodic image by appending the name to each atom.
                atoms_tmp = [str(x) + name for x in atoms]

                # Append the translated atomic positions and atom names to the corresponding lists.
                l.append(v_tmp)
                atoms_list.append(atoms_tmp)

    # Concatenate all the replicated atomic positions into a single array.
    l = np.concatenate(l, axis=0)

    # Flatten the list of atom names to convert it into a 1D array.
    atoms_list = [item for sublist in atoms_list for item in sublist]

    # Convert the flattened atom names list into a numpy array.
    atoms_list = np.array(atoms_list)

    # Return the concatenated atomic positions array and the array of atom names.
    return l, atoms_list

# %% 
def distances(coordinates): # Calculate interatomic distances
    '''
    Calculate interatomic distances
    
    Input:
    coordinates - cartesian coordinates of the lattice 
    
    Output:
    mol_pdist - interatomic distances 
    mol_pdist_matrix - matrix of interatomic distances
    '''
    # calculate interatomic distances using pdist
    mol_pdist = pdist(coordinates) 
    # generate a matrix of the coordinates
    mol_pdist_matrix = squareform(mol_pdist)
    return(mol_pdist, mol_pdist_matrix)

# %%
def network(dists, atoms): # Generate a network of bonds
    '''
    Generate a network (graph) for the structure based on bonds. Bonds are determined based on interatomic distnace.
    
    Input:
    dists - distances (including PBC)
    atoms - list of atoms' names (including PBC)
    
    Output:
    neis_graph - a dictionary which is a graph representation of the structre
    '''
    # Initialize dictionary
    neis_graph = {}
    for i, atom in enumerate(atoms): # iterate over all atoms
        #if '(0,0,0)' in atom:
        # Identify the specie
        specie = atom.split('_')[0]
        # Find neighbors (which are not of the same specie)
        neis_bool = (dists[i] < 3.1) & (dists[i] > 0) & [specie not in x for x in atoms]
        # Store
        neis_graph[atom] = atoms[neis_bool]
    return(neis_graph)

# %%


# %%
def self_returning_paths(graph, start, steps):
    def dfs(curr, path):
        path.append(curr)
        if curr == start and len(path) == steps + 1:
            self_returning.append(path[:])
        if len(path) > steps + 1:
            path.pop()
            return
        for neighbor in graph[curr]:
            dfs(neighbor, path)
        path.pop()

    self_returning = []
    path = []
    dfs(start, path)
    return self_returning

# %%
def find_neighbors(graph, start, steps):
    def dfs(curr, path):
        path.append(curr)
        if len(path) == steps +1 and path[-1] != start:
            self_returning.append(path[-1])
        if len(path) > steps + 1:
            path.pop()
            return
        for neighbor in graph[curr]:
            dfs(neighbor, path)
        path.pop()

    self_returning = []
    path = []
    dfs(start, path)
    return self_returning

# %%
def cycles(graph, start, steps):
    def dfs(curr, path):
        path.append(curr)
        if (curr == start) and (len(path) == steps + 1) and len(set(path[1:])) == len(path[1:]):
            self_returning.append(path[:])
        if len(path) > steps + 1:
            path.pop()
            return
        for neighbor in graph[curr]:
            dfs(neighbor, path)
        path.pop()

    self_returning = []
    path = []
    dfs(start, path)
    return self_returning

# %%


# %%
def bonds_dict(atoms, mol_pdist):
    bonds = dict(zip(combinations(atoms, 2), mol_pdist))
    bonds_inv = {k[::-1]: v for k, v in bonds.items()}
    bonds = {**bonds, **bonds_inv}
    return(bonds)

# %%
def dists_mask(paths, bonds):
    distances = []
    masks = []
    for path in paths:
        distances.append([bonds[(path[i], path[i+1])] for i in range(len(path)-1)])
        masks.append([4 if 'C' in i else 1 for i in path])
    return distances, masks

# %%
def molecule_paths(neis_graph, bonds, max_hops):
    paths_list, distances_list, masks_list = {}, {}, {}
    hops = np.arange(2, max_hops+1, 2)
    for hop in hops:
        paths, distances, masks = {}, {}, {}
        for atom in neis_graph:
            if '(0,0,0)' in atom:
                key = atom
                atom_paths = self_returning_paths(neis_graph, atom, hop)
                paths_distances, paths_masks = dists_mask(atom_paths, bonds)
                paths[key], distances[key], masks[key] = atom_paths, pd.DataFrame(paths_distances), pd.DataFrame(paths_masks)
        paths_list[hop] = paths
        distances_list[hop] = distances 
        masks_list[hop] = masks
    return(paths_list, distances_list, masks_list)

# %%
def molecule_cycles(neis_graph, bonds, max_hops):
    paths_list, distances_list, masks_list = {}, {}, {}
    hops = np.arange(2, max_hops+1, 2)
    for hop in hops:
        paths, distances, masks = {}, {}, {}
        for atom in neis_graph:
            if '(0,0,0)' in atom:
                key = atom
                atom_paths = cycles(neis_graph, atom, hop)
                paths_distances, paths_masks = dists_mask(atom_paths, bonds)
                paths[key], distances[key], masks[key] = atom_paths, pd.DataFrame(paths_distances), pd.DataFrame(paths_masks)
        paths_list[hop] = paths
        distances_list[hop] = distances 
        masks_list[hop] = masks
    return(paths_list, distances_list, masks_list)

# %%
def path_exponents(paths_distances, hops):
    all_sums = []
    for length in np.arange(2, hops+1, 2):
        path_sum = pd.concat(paths_distances[length]).sum(axis = 1).reset_index().sort_values(by = ['level_0', 0])
        path_sum = path_sum.set_index(['level_0','level_1'])[0].unstack(level = 'level_1')
        path_sum.index.name = None
        path_sum.columns.name = None
        path_sum = path_sum.add_prefix(str(length)+'_(').add_suffix(')')
        all_sums.append(path_sum)
    paths_exp = np.exp(-1*pd.concat(all_sums, axis = 1))
    return(paths_exp)

# %%
def coords_mask(atom_neis, neis_coords):
    neis_mask = np.array([8 if i == 'O' else 22 if i == 'Ti' else 56 if i == 'Ba' else 42 for i in atom_neis]).reshape(-1,1)
    neis_array = np.concatenate([neis_mask, neis_coords], axis = 1).reshape(-1)
    return(neis_array)

# %%
def neis_exponents(neis, hops):
    all_neis = []
    for length in np.arange(1, hops+1, 1):
        df = pd.DataFrame.from_dict(neis[length], orient='index')
        df = df.add_prefix(str(length)+'_[').add_suffix(']')
        all_neis.append(df)
    all_neis = pd.concat(all_neis, axis = 1)
    return(all_neis)

# %%
def molecule_nei_coords(neis_graph, xyz, max_hops):
    neis_list = {}
    hops = np.arange(1, max_hops+1, 1)
    for hop in hops:
        neis = {}
        for atom in neis_graph:
            if '0,0,0' in atom:
                r = xyz[atom]
                atom_neis = find_neighbors(neis_graph, atom, hop)

                neis_coords = [xyz[x] for x in atom_neis]
                neis[atom] = coords_mask(atom_neis, neis_coords-r)
            
        neis_list[hop] = neis
    return(neis_list)

# %%
def initialize(atoms, coords, path_hops, neis_hops):
    mol_pdist, mol_pdist_matrix = distances(coords)
    neis_graph = network(mol_pdist_matrix, atoms)
    bonds = bonds_dict(atoms, mol_pdist)
    paths, paths_distances, paths_masks = molecule_paths(neis_graph, bonds, path_hops)
    #paths, paths_distances, paths_masks = molecule_cycles(neis_graph, bonds, path_hops)
    
    paths_exp = path_exponents(paths_distances, path_hops)
    xyz = {atoms[i]:coords[i,:] for i in range(len(coords))}
    neis = molecule_nei_coords(neis_graph, xyz, neis_hops)
    neis_exp = neis_exponents(neis, neis_hops)
    #return(paths_exp, neis_exp)
    return(paths, neis_graph, paths_exp, neis_exp)

# %%
# Depracted, kept for future sanity checks
def pd_exponents(dists, paths_bonds, path_hops):
    ls = []
    #ts = []
    for hops in np.arange(2, path_hops+1, 2):
        #t0_hop = time.time()
        ls_hop = {}
        for atom in paths_bonds[hops]:
            ls_hop[atom] = [[dists[b] for b in x] for x in paths_bonds[hops][atom]]
        ls_hop = pd.Series(ls_hop, name = str(hops)).apply(lambda x: np.sort(np.sum(x, axis = 1)))
        ls_hop.index.names = ['Atom']
        ls_hop = ls_hop.to_frame().reset_index()
        ls_hop = ls_hop.explode(str(hops))
        ls_hop['Count'] = ls_hop.groupby(['Atom']).cumcount() +1
        ls_hop = ls_hop.set_index(['Atom','Count']).unstack()
        ls_hop.columns = [f'{i}_({j})' for i, j in ls_hop.columns]
        ls.append(ls_hop)
        #ts.append(time.time()-t0_hop)
    ls = pd.concat(ls, axis = 1)
    ls = np.exp(-ls.astype('float64'))
    return(ls)

# %%
def ob_paths(dists, n_atoms, js, path_hops, path_bonds_numbered):
    j0 = 0
    lins = np.empty(shape = (n_atoms, np.sum(js)))
    for k, hops in enumerate(np.arange(2, path_hops+1, 2)):
        hops_data = path_bonds_numbered[hops]
        j_max =  js[k]
        for atom in range(n_atoms): 
            bnd = dists[[b for x in hops_data[atom] for b in x]].reshape(-1, hops)
            bnd_sum = np.sort(np.sum(bnd, axis = 1))
            j_bnd = bnd_sum.shape[0]
            if j_bnd < j_max:
                pad = np.empty(j_max-j_bnd)
                pad[:] = np.nan
                bnd_sum = np.hstack([bnd_sum, pad])
            lins[atom, j0:j_max+j0] = bnd_sum
        j0 = j0+j_max
    exps = np.exp(-lins)
    return(exps)

# %%
def ob_neis(neis_list, n_atoms, atoms, coords, coords_pbc, atoms_pbc, hops, js):
    j0 = 0
    neis_coords = np.empty(shape = (n_atoms, np.sum(js)))
    #hops = np.arange(1, hops+1, 1)
    for h, j_max in enumerate(js):
        for a, (atom, l) in enumerate(zip(atoms, neis_list[h])):
            vals = coords_pbc[l]-coords[a]
            j_tmp = vals.shape[0]
            neis_mask = np.array([8 if i == 'O' else 22 if i == 'Ti' else 56 if i == 'Ba' else 42 for i in atoms_pbc[l]]).reshape(j_tmp, 1)          
            vals = np.hstack([neis_mask, vals]).ravel().ravel()#np.concatenate([neis_mask, neis_coords], axis = 1).ravel()
            j_tmp = vals.shape[0]
            if j_tmp < j_max:
                pad = np.empty(j_max-j_tmp)
                pad[:] = np.nan
                vals = np.hstack([vals, pad])
            neis_coords[a, j0:j_max+j0] = vals
        j0 = j0+j_max
    return(neis_coords)

# %%
ts = []
dfs = []

for k, i in enumerate(structures):

    if k == 0:
        t0 = time.time()

        crystal = data.loc[data['Structure'] == str(i)]
        atoms = crystal.Atom.to_numpy()
        coords = crystal[['x','y','z']].astype(float).to_numpy() @ lp
        coords_pbc, atoms_pbc = pbc(coords, lp, atoms, 1, 1, 1)


        # Initialize
        paths, neis_graph, paths_exp, neis_exp = initialize(atoms_pbc, coords_pbc, path_hops, neis_hops)
        paths_exp.index = paths_exp.index.str.replace('\(0,0,0\)', '')
        bonds = []
        for length in paths:
            for head in paths[length]:
                for path in paths[length][head]:
                    for i in range(len(path)-1):
                        bonds.append(path[i]+'<->'+path[i+1])
        
        # Store the bonds of the structure
        bonds = set(bonds)
        bonds = np.array([x.split('<->') for x in bonds])
        bondsList = [tuple(x) for x in bonds.tolist()]
        bonds_numbering = np.arange(len(bondsList))
        bonds_numbering = dict(zip(bondsList, bonds_numbering))

        # Store the paths of the structure
        paths_bonds = {}
        for length in paths:
            paths_bonds[length] = {}
            for atom in paths[length]:
                paths_bonds[length][atom] = []
                for path in paths[length][atom]:
                    path_bonds = []
                    for i in range(length):
                        path_bonds.append((path[i], path[i+1]))
                    paths_bonds[length][atom].append(path_bonds)

        # Numbering for paths
        paths_bondsList = [list(paths_bonds[x].values()) for x in paths_bonds]
        path_bonds_numbered = {}
        for hop_length in paths_bonds:
            hop_dict = []
            for atom in paths_bonds[hop_length]:
                atom_list = []
                for path in paths_bonds[hop_length][atom]:
                    atom_list.append([bonds_numbering[x] for x in path])
                hop_dict.append(atom_list)
            path_bonds_numbered[hop_length] = hop_dict
        
        # Numbering for neis
        atoms_pbc_numbering = np.arange(len(atoms_pbc))
        atoms_pbc_numbering = dict(zip(atoms_pbc, atoms_pbc_numbering))

        neis_list = []
        hops = np.arange(1, neis_hops+1, 1)
        for hop in hops:
            neis = []
            for atom in atoms+'(0,0,0)':
                atom_neis = find_neighbors(neis_graph, atom, hop)
                atom_neis = [atoms_pbc_numbering[x] for x in atom_neis]
                neis.append(atom_neis)
            neis_list.append(neis)


        # Allocation
        js_path = [np.count_nonzero(paths_exp.columns.str.startswith(str(hops)+'_')) for hops in np.arange(2, path_hops+1, 2)]
        js_neis = [np.count_nonzero(neis_exp.columns.str.startswith(str(hops)+'_')) for hops in np.arange(1, neis_hops+1)]


        # Depracted, get a second version of paths_exp for sanity check
        # pbc_frame = pd.DataFrame(coords_pbc, index = atoms_pbc)
        # dists = np.linalg.norm(pbc_frame.loc[bonds[:,0]] - pbc_frame.loc[bonds[:,1]].to_numpy(), axis = 1)
        # df_exp = pd_path_exponents(dict(zip(bondsList, dists)), paths_bonds, path_hops)
        # df_exp.index = df_exp.index.str.replace('\(0,0,0\)', '')


# %%
ts = []
exps_clmns = paths_exp.columns
neis_clmns = neis_exp.columns
#n_structures = 1000
vs = []
nat = 32
for k, i in enumerate(structures):
    t0 = time.time()
    
    crystal = data.iloc[nat*(k):nat*(k+1)] #data.loc[data['Structure'] == str(i)]
    atoms = crystal.Atom.to_numpy()
    coords = crystal[['x','y','z']].astype(float).to_numpy() @ lp
    coords_pbc, atoms_pbc = pbc(coords, lp, atoms, 1, 1, 1)
    
    pbc_frame = pd.DataFrame(coords_pbc, index = atoms_pbc)
    dists = np.linalg.norm(pbc_frame.loc[[x.replace('--0','--'+str(k)) for x in bonds[:,0]]] - pbc_frame.loc[[x.replace('--0','--'+str(k)) for x in bonds[:,1]]].to_numpy(), axis = 1)


    exps = ob_paths(dists, nat, js_path, path_hops, path_bonds_numbered)
    neis = ob_neis(neis_list, nat, atoms, coords, coords_pbc, atoms_pbc, neis_hops, js_neis)


    v = np.concatenate([exps, neis], axis = 1)
    vs.append(v)
    #df_exps = pd.DataFrame(exps, index = atoms, columns = exps_clmns)
    #df_neis = pd.DataFrame(neis, index = atoms, columns = neis_clmns)

    #df = pd.concat([df_exps, df_exps], axis = 1)
    
    ts.append(time.time()-t0)
    if k == 500:
        break
#print(ts.mean())


# %%
np.mean(ts)

# %%
idx = ['--'.join([atom, structure]) for structure in structures for atom in atoms]
clmns =  list(exps_clmns) + list(neis_clmns)
df = pd.DataFrame(np.concatenate(vs, axis = 0), index = idx, columns = clmns)

# %%
df

# %%
df.to_parquet('X-BaTiO3-paths.parquet')

# %%



