import numpy as np
import gmsh
import meshio
import pickle
from os import path, makedirs

# unit_testing -> xXx NOT DONE xXx!
def save_model(model, save_dir, save_name):
    if not path.exists(save_dir):
        makedirs(save_dir)
    with open(f'{save_dir}/{save_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print('Model saved successfully.')

# unit_testing -> xXx NOT DONE xXx!
def load_model(load_dir):
    if not path.exists(load_dir):
        raise ValueError(f'The given directory "{load_dir}" does not exist!')
    with open(f'{load_dir}', 'rb') as f:
        return pickle.load(f)

# unit_testing -> xXx NOT DONE xXx!
def gen_uniform_1d_mesh_from_interval_and_resolution(domain, mesh_resolution):
    return np.linspace(domain[0], domain[1], (mesh_resolution+1))

# unit_testing -> xXx NOT DONE xXx!
def get_x_boundary_indices_on_1D_FEM_mesh(x_domain, meshInterval=128, minSpatialValue=0, maxSpatialValue=1):
    """..."""
    #----------- error check(s) -----------
    #--------------------------------------
    return np.array(np.round((x_domain-minSpatialValue)*meshInterval/(maxSpatialValue-minSpatialValue), decimals=0).astype(int))

# unit_testing -> DONE.
def flipStr(str: str) -> str:
    input_string = str

    #----------- error check(s) -----------
    if (type(input_string) is not type('')):
        raise ValueError(f'Cannot parse a {type(input_string)}. String input required.')
    #--------------------------------------

    tmpStr = ''
    for idx in range(len(input_string)-1,-1,-1):
        tmpStr += input_string[idx]
    return tmpStr


# unit_testing -> DONE.
def directBinStrSum(str: str) -> int:
    """...direct sum of a binary string..."""
    input_string = str

    #----------- error check(s) -----------
    if (type(input_string) is not type('')):
        raise ValueError(f'Cannot parse a {type(input_string)}. String input required.')
    #--------------------------------------

    sum = 0
    for s in input_string:
        if s not in ['0', '1']:
            raise ValueError(f'Input needs to be in a binary-string format. Received {input_string}, where {s} is non-binary.')
        sum += int(s)
    return sum

# unit_testing -> DONE.
def getIndexSuperset(vectSize: int, 
                    include_zero: bool = False, 
                    only_singletons: bool = False,
                    higher_order: bool = True) -> list:
        """Generate the index superset for Pick-and-Freeze Sobol sensitivity indices."""

        #----------- error check(s) -----------
        if type(vectSize) is not int:
            raise ValueError(f'vectSize must be an integer type. {type(vectSize)} was passed in.')
        if vectSize <= 0:
            raise ValueError(f'vectSize must be >=1. {vectSize} was passed in.')
        if type(include_zero) is not bool:
            raise ValueError(f'include_zero must be a bool type. {type(include_zero)} was passed in.')
        if type(only_singletons) is not bool:
            raise ValueError(f'only_singletons must be a bool type. {type(only_singletons)} was passed in.')
        #--------------------------------------

        supset = []
        cardinOfSupset = 2 ** vectSize
        for i in range(cardinOfSupset):
            if i == 0 and not include_zero:
                continue
            oneHotIdxSet = np.binary_repr(i, vectSize)
            if only_singletons:
                if directBinStrSum(oneHotIdxSet) > 1:
                    continue
            if not higher_order:
                if (directBinStrSum(oneHotIdxSet) > 1) and (directBinStrSum(oneHotIdxSet) != vectSize):
                    continue
            supset.append(oneHotIdxSet)
        return supset

# unit_testing -> DONE.
def getIndexSubsets(indexStr: str) -> list:
    """Generate all the unordered 1-hot-encoded list of subsets of the 1-hot-encoded indexStr. E.g. if indexStr = '101' then getIndexSubets returns the list ['000', '100', '001', '101']"""
    
    #----------- error check(s) -----------
    if (type(indexStr) is not type('')):
        raise ValueError(f'Cannot parse a {type(indexStr)}. String input required.')
    for s in indexStr:
        if s not in ['0', '1']:
            raise ValueError(f'Input needs to be in a binary-string format. Received {indexStr}, where {s} is non-binary.')
    #--------------------------------------

    ones_idx = [i for i, singleBit in enumerate(indexStr) if singleBit == '1']
    k = len(ones_idx)
    subsets = []
    # loop over all 2^k "masks", using the integer mask as a collection of k binary flags, one for each position where the idxA string had a '1'.
    for mask in range(2**k):
        # print(bin(mask))
        # build a fresh list of '0's
        t = ['0'] * len(indexStr)
        # loop over indices of the ones_idx list
        for j in range(k):
            # for each bit in mask, if it's 1 we save the '1' in t:
            #   Let a,b,x,y be non-negative integers:
            #   a >> b: right-shift the bits in the binary-representation of a by b digits (eg 1010 >> 1 == 101). 
            #   x & y operator: bitwise compare binary-representation of x and y (eg 1011 & 1101 == 1001; 101 & 1 == 1; 001 & 1 == 0; 110 & 1 == 0).
            # This means we're checking the least-significant bit of the binary-representation of the j-shifted mask integer.
            shiftMaskBinaryBy = j
            if (mask >> shiftMaskBinaryBy) & 1:
                t[ones_idx[j]] = '1'
        subsets.append(''.join(t))
    return subsets

# unit_testing -> DONE.
def getIndexComplement(idxBinString: str) -> str:
        """Get the complement of an index binary string."""

        #----------- error check(s) -----------
        if (type(idxBinString) is not type('')):
            raise ValueError(f'Cannot parse a {type(idxBinString)}. String input required.')
        for s in idxBinString:
            if s not in ['0', '1']:
                raise ValueError(f'Input needs to be in a binary-string format. Received {idxBinString}, where {s} is non-binary.')
        #--------------------------------------

        return ''.join('1' if bit == '0' else '0' for bit in idxBinString)

# unit_testing -> DONE.
def getSingletonIndexAsInt(idxBinString: str) -> int:
    """Convert the one-hot-encoding format of the idxBinString into an integer index value, read from left to right."""

    #----------- error check(s) -----------
    if (type(idxBinString) is not type('')):
            raise ValueError(f'Cannot parse a {type(idxBinString)}. String input required.')
    for s in idxBinString:
        if s not in ['0', '1']:
            raise ValueError(f'Input needs to be in a binary-string format. Received {idxBinString}, where {s} is non-binary.')
    if directBinStrSum(idxBinString) != 1:
        raise ValueError(f'One and only one index in the one-hot-encoded input, corresponding to the index-integer of interest, needs to be 1 (e.g. 100 -> 0, 0001 -> 3). {idxBinString} was passed in.')
    #--------------------------------------

    # assert directBinStrSum(idxBinString) == 1 ### DEPRECATED
    for i in range(len(idxBinString)):
        if idxBinString[i] == "1":
            return i
    return -1

# unit_testing -> xXx NOT DONE xXx!
def generate_xdmf_from_dot_msh(mesh_save_dir, mesh_save_name):
    """..."""
    #----------- error check(s) -----------
    #--------------------------------------

    mesh = meshio.read(f"{mesh_save_dir}/{mesh_save_name}.msh")
    cells = {"triangle": mesh.cells_dict["triangle"]}

    #create new mesh with only the triangle elements
    triangle_mesh = meshio.Mesh(points=mesh.points[:,:2],
                            cells=cells,
                            cell_data={"triangle": [mesh.cell_data_dict["gmsh:geometrical"]["triangle"]]})
    
    #write xdmf using only the triangle mesh elements
    triangle_mesh.write(f"{mesh_save_dir}/{mesh_save_name}.xdmf")

# unit_testing -> NOT done!
def generate_2D_mesh(domain_height, domain_width, mesh_steps, mesh_save_dir, mesh_save_name='rectangle', gen_xdmf=True):
    """..."""
    #----------- error check(s) -----------
    #--------------------------------------

    gmsh.initialize()
    gmsh.clear()

    gmsh.model.add("domain_mesh")

    points = [
        gmsh.model.geo.addPoint(0, 0, 0, mesh_steps),
        gmsh.model.geo.addPoint(domain_width, 0, 0, mesh_steps),
        gmsh.model.geo.addPoint(domain_width, domain_height, 0, mesh_steps),
        gmsh.model.geo.addPoint(0, domain_height, 0, mesh_steps)
    ]

    lines = [
        gmsh.model.geo.addLine(points[0], points[1]),
        gmsh.model.geo.addLine(points[1], points[2]),
        gmsh.model.geo.addLine(points[2], points[3]),
        gmsh.model.geo.addLine(points[3], points[0])
    ]

    loop = gmsh.model.geo.addCurveLoop(lines)
    surface = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    #save .msh
    gmsh.write(f"{mesh_save_dir}/{mesh_save_name}.msh")
    gmsh.finalize()

    if gen_xdmf:
        generate_xdmf_from_dot_msh(mesh_save_dir=mesh_save_dir, mesh_save_name=mesh_save_name)
