"""Various tools to support data analysis."""

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN


def tag_surface_atoms(bulk_struct, surface_struct):
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
    voronoi_tags = find_surface_atoms_with_voronoi(bulk_struct, surface_struct)
    surface_atoms = AseAtomsAdaptor.get_atoms(surface_struct)
    surface_atoms.set_tags(voronoi_tags)
    return surface_atoms


def find_surface_atoms_with_voronoi(bulk_struct, surface_struct):
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
    bulk_cn_dict = calculate_coordination_of_bulk_struct(bulk_struct)
    voronoi_nn = VoronoiNN(tol=0.1)  # 0.1 chosen for better detection

    tags = []
    for idx, site in enumerate(surface_struct):

        # Tag as surface atom only if it's above the center of mass
        if site.frac_coords[2] > center_of_mass[2]:
            try:

                # Tag as surface if atom is under-coordinated
                cn = voronoi_nn.get_cn(surface_struct, idx, use_weights=True)
                cn = round(cn, 5)
                if cn < bulk_cn_dict[site.full_wyckoff]:
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
    bulk_cn_dict = {}
    for idx, site in enumerate(bulk_struct):
        if site.full_wyckoff not in bulk_cn_dict:
            cn = voronoi_nn.get_cn(bulk_struct, idx, use_weights=True)
            cn = round(cn, 5)
            bulk_cn_dict[site.full_wyckoff] = cn
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
