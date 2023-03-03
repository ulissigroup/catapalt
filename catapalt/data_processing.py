"""Various tools to support data analysis."""

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN
import warnings
import numpy as np
from collections import defaultdict


def tag_surface_atoms(surface_struct, bulk_struct = None, bulk_cn_dict = None):
    """
    Sets the tags of an `ase.Atoms` object. Any atom that we consider a "bulk"
    atom will have a tag of 0, and any atom that we consider a "surface" atom
    will have a tag of 1.
    Args:
        bulk_struct (pymatgen.structure.Structure):  An object of the bulk structure with
            wyckoff site info.
        surface_struct (pymatgen.structure.Structure):  An object of the surface structure
            with wyckoff site info.
    """
    if bulk_struct is None and bulk_cn_dict is None:
        warnings.warn("No bulk coordination information was provided, assuming the bulk atoms have CN = 12, this will cause errors if untrue.")
    bulk_cn_dict = get_bulk_cn(bulk_struct, bulk_cn_dict, surface_struct)
    surface_atoms = AseAtomsAdaptor.get_atoms(surface_struct)
    voronoi_tags = find_surface_atoms_with_voronoi(bulk_cn_dict, surface_struct)
    height_tags = find_surface_atoms_by_height(surface_atoms)
    # If either of the methods consider an atom a "surface atom", then tag it as such.
    tags = [max(v_tag, h_tag) for v_tag, h_tag in zip(voronoi_tags, height_tags)]
    surface_atoms.set_tags(tags)
    return surface_atoms

def get_bulk_cn(bulk_struct, bulk_cn_dict, surface_struct):
    if bulk_struct is None and bulk_cn_dict is None:
        cn_dict = {}
        for el in np.unique(AseAtomsAdaptor.get_atoms(surface_struct).get_chemical_symbols()):
            bulk_cn_dict[el] = 12
    elif bulk_struct is not None:
        bulk_cn_dict = calculate_coordination_of_bulk_struct(bulk_struct)
    return bulk_cn_dict

def find_surface_atoms_with_voronoi(bulk_cn_dict, surface_struct):
    """
    Labels atoms as surface or bulk atoms according to their coordination
    relative to their bulk structure. If an atom's coordination is less than it
    normally is in a bulk, then we consider it a surface atom. We calculate the
    coordination using pymatgen's Voronoi algorithms.
    Args:
        bulk_struct (pymatgen.structure.Structure):  An object of the bulk structure.
        surface_struct (pymatgen.structure.Structure):  An object of the surface structure.
    Returns:
        (list): A list of 0's and 1's whose indices align with the atoms in
            surface_struct. 0's indicate a subsurface atom and 1 indicates a surface atom.
    """
    # Initializations
    center_of_mass = get_center_of_mass(surface_struct)
    voronoi_nn = VoronoiNN(tol=0.1)  # 0.1 chosen for better detection
    
    tags = []
    for idx, site in enumerate(surface_struct):

        # Tag as surface atom only if it's above the center of mass
        if site.frac_coords[2] > center_of_mass[2]:
            try:

                # Tag as surface if atom is under-coordinated
                cn = voronoi_nn.get_cn(surface_struct, idx, use_weights=True)
                cn = round(cn, 5)
                if cn < min(bulk_cn_dict[site.species_string]):
                    tags.append(1)
                else:
                    tags.append(0)

            # Tag as surface if we get a pathological error
            except RuntimeError:
                tags.append(1)

        # Tag as bulk otherwise
        else:
            tags.append(0)
    return tags

def find_surface_atoms_by_height(surface_atoms):
    '''
    As discussed in the docstring for `_find_surface_atoms_with_voronoi`,
    sometimes we might accidentally tag a surface atom as a bulk atom if there
    are multiple coordination environments for that atom type within the bulk.
    One heuristic that we use to address this is to simply figure out if an
    atom is close to the surface. This function will figure that out.
    Specifically:  We consider an atom a surface atom if it is within 2
    Angstroms of the heighest atom in the z-direction (or more accurately, the
    direction of the 3rd unit cell vector).
    Arg:
        surface_atoms   The surface where you are trying to find surface sites in
                        `ase.Atoms` format
    Returns:
        tags            A list that contains the indices of
                        the surface atoms
    '''
    unit_cell_height = np.linalg.norm(surface_atoms.cell[2])
    scaled_positions = surface_atoms.get_scaled_positions()
    scaled_max_height = max(scaled_position[2] for scaled_position in scaled_positions)
    scaled_threshold = scaled_max_height - 2. / unit_cell_height

    tags = [0 if scaled_position[2] < scaled_threshold else 1
            for scaled_position in scaled_positions]
    return tags


def calculate_coordination_of_bulk_struct(bulk_struct):
    """
    Finds all unique sites in a bulk structure and then determines their
    coordination number. Then parses these coordination numbers into a
    dictionary whose keys are the elements of the atoms and whose values are
    their possible coordination numbers.
    For example: `bulk_cns = {'Pt': {3., 12.}, 'Pd': {12.}}`
    Args:
        bulk_struct (pymatgen.structure.Structure):  An object of the bulk structure.
    Returns:
        (dict): A dict whose keys are the wyckoff values in the bulk_struct
            and whose values are the coordination numbers of that site.
    """
    voronoi_nn = VoronoiNN(tol=0.1)  # 0.1 chosen for better detection

    # Object type conversion so we can use Voronoi
    sga = SpacegroupAnalyzer(bulk_struct)

    # We'll only loop over the symmetrically distinct sites for speed's sake
    sym_struct = sga.get_symmetrized_structure()

    # We'll only loop over the symmetrically distinct sites for speed's sake
    bulk_cn_dict = defaultdict(set)
    for idx in sym_struct.equivalent_indices:
        site = sym_struct[idx[0]]
        cn = voronoi_nn.get_cn(sym_struct, idx[0], use_weights=True)
        cn = round(cn, 5)
        bulk_cn_dict[site.species_string].add(cn)
    return bulk_cn_dict


def get_center_of_mass(pmg_struct):
    """
    Calculates the center of mass of a pmg structure.
    Args:
        pmg_struct (pymatgen.core.structure.Structure): pymatgen structure to be
            considered.
    Returns:
        numpy.ndarray: the center of mass
    """
    weights = [s.species.weight for s in pmg_struct]
    center_of_mass = np.average(pmg_struct.frac_coords, weights=weights, axis=0)
    return center_of_mass
