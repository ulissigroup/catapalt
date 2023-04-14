"""A class for analyzing the nature of the surface.
Original nuclearity code taken from Unnatti Sharma:
https://github.com/ulissigroup/NuclearityCalculation
"""
from itertools import product, combinations
import numpy as np

# import graph_tool as gt
from ase import neighborlist
from ase.neighborlist import natural_cutoffs

# from graph_tool import topology
from scipy.spatial import Delaunay
import copy
from pymatgen.io.ase import AseAtomsAdaptor
import networkx as nx


class Plane:
    def __init__(self, points):
        """
        Initialize plane class.

        Args:
            p1: first point on the plane
            p2: second point on the plane
            p3: third point on the plane
            identifier: id for the plane
        """
        p1, p2, p3 = points
        v1 = p2 - p1
        v2 = p3 - p1
        self.norm = np.cross(v1, v2)


class SurfaceAnalyzer:
    def __init__(self, slab):
        """
        Initialize class to handle surface based analysis.

        Args:
            slab (ase.Atoms): object of the slab.
        """
        self.slab = slab

    def get_surface_composition(self):
        elements = self.slab.get_chemical_symbols()
        surface_elements = [
            elements[idx] for idx, tag in enumerate(self.slab.get_tags()) if tag == 1
        ]

        composition = {}
        for element in np.unique(elements):
            composition[element] = round(
                (surface_elements.count(element) / len(surface_elements)), 3
            )
        return composition

    def get_nuclearity(self):
        """
        Function to get the nuclearity for each element in a surface.

        Returns:
            dict: output with per element nuclearities
        """
        elements = np.unique(self.slab.get_chemical_symbols())
        slab_atoms = self.slab
        replicated_slab_atoms = self.slab.repeat((2, 2, 1))

        # Grab connectivity matricies
        overall_connectivity_matrix = self._get_connectivity_matrix(slab_atoms)
        overall_connectivity_matrix_rep = self._get_connectivity_matrix(
            replicated_slab_atoms
        )

        # Grab surface atom idxs
        surface_indices = [
            idx for idx, tag in enumerate(slab_atoms.get_tags()) if tag == 1
        ]
        surface_indices_rep = [
            idx for idx, tag in enumerate(replicated_slab_atoms.get_tags()) if tag == 1
        ]

        # Iterate over atoms and assess nuclearity
        output_dict = {}
        for element in elements:
            surface_atoms_of_element = [
                atom.symbol == element and atom.index in surface_indices
                for atom in slab_atoms
            ]
            surface_atoms_of_element_rep = [
                atom.symbol == element and atom.index in surface_indices_rep
                for atom in replicated_slab_atoms
            ]

            if sum(surface_atoms_of_element) == 0:
                output_dict[element] = {"nuclearity": 0, "nuclearities": []}

            else:
                hist = self._get_nuclearity_neighbor_counts(
                    surface_atoms_of_element, overall_connectivity_matrix
                )
                hist_rep = self._get_nuclearity_neighbor_counts(
                    surface_atoms_of_element_rep, overall_connectivity_matrix_rep
                )
                output_dict[element] = self._evaluate_infiniteness(hist, hist_rep)

        return output_dict

    def _get_nuclearity_neighbor_counts(
        self, surface_atoms_of_element, connectivity_matrix
    ):
        """
        Function that counts the like surface neighbors for surface atoms.
        Args:
            surface_atoms_of_element (list[bool]): list of all surface atoms which
                are of a specific element
            connectivity_matrix (numpy.ndarray[int8]): which atoms in the slab are connected
        Returns:
            numpy.ndarray[int]: counts of neighbor groups
        """
        connectivity_matrix = connectivity_matrix[surface_atoms_of_element, :]
        connectivity_matrix = connectivity_matrix[:, surface_atoms_of_element]
        graph = gt.Graph(directed=False)
        graph.add_vertex(n=connectivity_matrix.shape[0])
        graph.add_edge_list(np.transpose(connectivity_matrix.nonzero()))
        labels, hist = topology.label_components(graph, directed=False)
        return hist

    def _evaluate_infiniteness(self, hist, hist_rep):
        """
        Function that compares the connected counts between the minimal slab and a
        repeated slab to classify the type of infiniteness.
        Args:
            hist (list[int]): list of nuclearities observed in minimal slab
            hist_rep (list[int]): list of nuclearities observed in replicated slab
        Returns:
            nuclearity dict (dict): the max nuclearity and all nuclearities for the element on that surface
        """
        if max(hist) == max(hist_rep):
            return {"nuclearity": max(hist), "nuclearities": hist}
        elif max(hist) == 0.5 * max(hist_rep):
            return {"nuclearity": "semi-finite", "nuclearities": hist}
        elif max(hist) == 0.25 * max(hist_rep):
            return {"nuclearity": "infinite", "nuclearities": hist}
        else:
            return {"nuclearity": "somewhat-infinite", "nuclearities": hist}

    def _get_connectivity_matrix(self, slab_atoms):
        """
        Get connectivity matrix by looking at nearest neighbors.
        Args:
            slab_atoms (ase.Atoms): a slab object
        Returns:
            numpy.ndarray[int8]: an array describing what atoms are connected
        """
        cutOff = natural_cutoffs(slab_atoms)
        neighborList = neighborlist.NeighborList(
            cutOff, self_interaction=False, bothways=True
        )
        neighborList.update(slab_atoms)
        overall_connectivity_matrix = neighborList.get_connectivity_matrix()
        return overall_connectivity_matrix

    def get_stdev_of_z(self):
        """
        Calculate the standard deviation in the z-coordinates normalized by the radius.
        This indicates the degree of step character as perfectly planar surfaces will have a value
        close to zero.

        Returns:
            (float): The standard deviation in the z-coordinate
        """
        surface_atoms = self.slab[
            [idx for idx, tag in enumerate(self.slab.get_tags()) if tag == 1]
        ]
        surface_atoms.set_positions(
            surface_atoms.get_positions() - surface_atoms.get_center_of_mass()
        )
        cutoff = natural_cutoffs(surface_atoms)
        surface_z_coords_normalized = [
            atom.z / cutoff[idx] for idx, atom in enumerate(surface_atoms)
        ]
        return np.std(surface_z_coords_normalized)

    def get_surface_cn_info(self):
        """
        Calculates the surface coordination numbers (cn) for each surface atom which is used to
        return (1) the mean surface cn (2) a dictionary of the unique coordination numbers and
        their frequency

        Returns:
            (dict): the coordination info. ex. `{"mean": 5.5, "proportions": {5: 0.5, 6: 0.5}}
        """
        surface_atoms = self.slab[
            [idx for idx, tag in enumerate(self.slab.get_tags()) if tag == 1]
        ]
        connectivity_matrix = self._get_connectivity_matrix(surface_atoms).toarray()
        cns = [sum(row) for row in connectivity_matrix]
        proportion_cns = {}
        for cn in np.unique(cns):
            proportion_cns[cn] = cns.count(cn) / len(cns)
        cn_info = {"mean": np.mean(cns), "proportions": proportion_cns}
        return cn_info

    def get_3d_character_graph(self, categorize_elements=False):
        surface_atoms = self.slab[
            [idx for idx, tag in enumerate(self.slab.get_tags()) if tag == 1]
        ]
        surface_atoms_repeated = self._custom_tile_atoms(surface_atoms)
        delaunay_2d_mesh = Delaunay(surface_atoms_repeated.get_positions()[:, 0:2])
        graph = nx.Graph()

        # Define all planes with normal vectors of interest
        plane_dict = {}
        for simplex in delaunay_2d_mesh.simplices:
            if any([node_num in range(len(surface_atoms)) for node_num in simplex]):
                plane = Plane(surface_atoms_repeated[simplex].get_positions())
                plane_dict[str(simplex[0]) + str(simplex[1]) + str(simplex[2])] = plane

        # Iterate over nodes and define them in the graph
        for idx, atom in enumerate(surface_atoms):
            relevant_simplices = [
                simplex for simplex in delaunay_2d_mesh.simplices if idx in simplex
            ]
            average_angle = self._get_node_label(idx, relevant_simplices, plane_dict)
            graph.add_node(idx, average_angle=average_angle)
                
        # Iterate over simplices to define edges TODO: ADD PBC STITCHING
        for simplex in delaunay_2d_mesh.simplices:
            for edge_pair in combinations(simplex, 2):
                if all([node_num in range(len(surface_atoms)) for node_num in edge_pair]):
                    graph.add_edge(*edge_pair)
        return graph

    def _get_node_label(self, node, simplices, plane_dict):
        # Note there is definitely a faster way to do this so it should be improved
        angles = []
        nodes_of_interest = list(
            np.unique([item for sublist in simplices for item in sublist])
        )
        nodes_of_interest.remove(node)
        for idx in nodes_of_interest:
            pair_simplices = [simplex for simplex in simplices if idx in simplex]
            if len(pair_simplices) > 1:
                angles.append(
                    self._get_angle_between_planes(
                        plane_dict[
                            str(pair_simplices[0][0])
                            + str(pair_simplices[0][1])
                            + str(pair_simplices[0][2])
                        ],
                        plane_dict[
                            str(pair_simplices[1][0])
                            + str(pair_simplices[1][1])
                            + str(pair_simplices[1][2])
                        ],
                    )
                )
        return np.mean(angles)

    def _get_angle_between_planes(self, plane1, plane2):
        cos = abs(
            round(
                np.dot(plane1.norm, plane2.norm)
                / (
                    np.sqrt(plane2.norm.dot(plane2.norm))
                    * np.sqrt(plane1.norm.dot(plane1.norm))
                ),
                3,
            )
        )
        return np.arccos(cos)
    
    def _custom_tile_atoms(self, atoms):
        vectors = [v for v in atoms.cell if ((round(v[0], 3) != 0) or (round(v[1],3 != 0)))]
        repeats = list(product([-1,0,1],repeat = 2))
        repeats.remove((0,0))
        new_atoms = copy.deepcopy(atoms)
        for repeat in repeats:
            atoms_shifted = copy.deepcopy(atoms)
            atoms_shifted.set_positions(atoms.get_positions() + vectors[0]*repeat[0] + vectors[1]*repeat[1])
            new_atoms += atoms_shifted
        return new_atoms
            