"""Various tools to support data analysis. Many functions adapted from ocdata."""

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN
import warnings
import numpy as np
from collections import defaultdict
import math

MIN_XY = 8


def tile_atoms(atoms):
    """
    This function will repeat an atoms structure in the x and y direction until
    the x and y dimensions are at least as wide as the MIN_XY constant.
    Args:
        atoms   `ase.Atoms` object of the structure that you want to tile
    Returns:
        atoms_tiled     An `ase.Atoms` object that's just a tiled version of
                        the `atoms` argument.
    """
    x_length = np.linalg.norm(atoms.cell[0])
    y_length = np.linalg.norm(atoms.cell[1])
    nx = int(math.ceil(MIN_XY / x_length))
    ny = int(math.ceil(MIN_XY / y_length))
    n_xyz = (nx, ny, 1)
    atoms_tiled = atoms.repeat(n_xyz)
    return atoms_tiled


def tag_surface_atoms(surface_atoms, bulk_atoms):
    """
    Sets the tags of an `ase.Atoms` object. Any atom that we consider a "bulk"
    atom will have a tag of 0, and any atom that we consider a "surface" atom
    will have a tag of 1. We use a combination of Voronoi neighbor algorithms
    (adapted from from `pymatgen.core.surface.Slab.get_surface_sites`; see
    https://pymatgen.org/pymatgen.core.surface.html) and a distance cutoff.
    Arg:
        bulk_atoms      `ase.Atoms` format of the respective bulk structure
        surface_atoms   The surface where you are trying to find surface sites in
                        `ase.Atoms` format
    """
    voronoi_tags = find_surface_atoms_with_voronoi(bulk_atoms, surface_atoms)
    height_tags = find_surface_atoms_by_height(surface_atoms)
    # If either of the methods consider an atom a "surface atom", then tag it as such.
    tags = [max(v_tag, h_tag) for v_tag, h_tag in zip(voronoi_tags, height_tags)]
    return tags


def find_surface_atoms_with_voronoi(bulk_atoms, surface_atoms):
    """
    Labels atoms as surface or bulk atoms according to their coordination
    relative to their bulk structure. If an atom's coordination is less than it
    normally is in a bulk, then we consider it a surface atom. We calculate the
    coordination using pymatgen's Voronoi algorithms.
    Note that if a single element has different sites within a bulk and these
    sites have different coordinations, then we consider slab atoms
    "under-coordinated" only if they are less coordinated than the most under
    undercoordinated bulk atom. For example:  Say we have a bulk with two Cu
    sites. One site has a coordination of 12 and another a coordination of 9.
    If a slab atom has a coordination of 10, we will consider it a bulk atom.
    Args:
        bulk_atoms      `ase.Atoms` of the bulk structure the surface was cut
                        from.
        surface_atoms   `ase.Atoms` of the surface
    Returns:
        tags    A list of 0's and 1's whose indices align with the atoms in
                `surface_atoms`. 0's indicate a bulk atom and 1 indicates a
                surface atom.
    """
    # Initializations
    surface_struct = AseAtomsAdaptor.get_structure(surface_atoms)
    center_of_mass = calculate_center_of_mass(surface_struct)
    bulk_cn_dict = calculate_coordination_of_bulk_atoms(bulk_atoms)
    voronoi_nn = VoronoiNN(tol=0.1)  # 0.1 chosen for better detection
    tags = []
    for idx, site in enumerate(surface_struct):

        # Tag as surface atom only if it's above the center of mass
        if site.frac_coords[2] > center_of_mass[2]:
            try:

                # Tag as surface if atom is under-coordinated
                cn = voronoi_nn.get_cn(surface_struct, idx, use_weights=True)
                cn = round(cn, 3)
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


def calculate_center_of_mass(struct):
    """
    Determine the surface atoms indices from here
    """
    weights = [site.species.weight for site in struct]
    center_of_mass = np.average(struct.frac_coords, weights=weights, axis=0)
    return center_of_mass


def calculate_coordination_of_bulk_atoms(bulk_atoms):
    """
    Finds all unique atoms in a bulk structure and then determines their
    coordination number. Then parses these coordination numbers into a
    dictionary whose keys are the elements of the atoms and whose values are
    their possible coordination numbers.
    For example: `bulk_cns = {'Pt': {3., 12.}, 'Pd': {12.}}`
    Arg:
        bulk_atoms  An `ase.Atoms` object of the bulk structure.
    Returns:
        bulk_cn_dict    A defaultdict whose keys are the elements within
                        `bulk_atoms` and whose values are a set of integers of the
                        coordination numbers of that element.
    """
    voronoi_nn = VoronoiNN(tol=0.1)  # 0.1 chosen for better detection

    # Object type conversion so we can use Voronoi
    bulk_struct = AseAtomsAdaptor.get_structure(bulk_atoms)
    sga = SpacegroupAnalyzer(bulk_struct)
    sym_struct = sga.get_symmetrized_structure()

    # We'll only loop over the symmetrically distinct sites for speed's sake
    bulk_cn_dict = defaultdict(set)
    for idx in sym_struct.equivalent_indices:
        site = sym_struct[idx[0]]
        cn = voronoi_nn.get_cn(sym_struct, idx[0], use_weights=True)
        cn = round(cn, 5)
        bulk_cn_dict[site.species_string].add(cn)
    return bulk_cn_dict


def find_surface_atoms_by_height(surface_atoms):
    """
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
    """
    unit_cell_height = np.linalg.norm(surface_atoms.cell[2])
    scaled_positions = surface_atoms.get_scaled_positions()
    scaled_max_height = max(scaled_position[2] for scaled_position in scaled_positions)
    scaled_threshold = scaled_max_height - 2.0 / unit_cell_height

    tags = [
        0 if scaled_position[2] < scaled_threshold else 1
        for scaled_position in scaled_positions
    ]
    return tags
