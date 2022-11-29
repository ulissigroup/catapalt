"""A class for analyzing the nature of the surface.
Original nuclearity code taken from Unnatti Sharma:
https://github.com/ulissigroup/NuclearityCalculation
"""

import numpy as np
import graph_tool as gt
from ase import neighborlist
from ase.neighborlist import natural_cutoffs
from graph_tool import topology


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
            composition[element] = surface_elements.count(element) / len(
                surface_elements
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
