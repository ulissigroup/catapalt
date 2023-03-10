"""A class for analyzing the nature of the surface.
Original nuclearity code taken from Unnatti Sharma:
https://github.com/ulissigroup/NuclearityCalculation
"""

import numpy as np
import graph_tool as gt
from ase import neighborlist
from ase.neighborlist import natural_cutoffs
from graph_tool import topology
from scipy.spatial import Delaunay
from quad_mesh_simplify import simplify_mesh
import copy
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from pymatgen.io.ase import AseAtomsAdaptor


class Plane:
    def __init__(self, p1, p2, p3, identifier):
        """
        Initialize plane class.

        Args:
            p1: first point on the plane
            p2: second point on the plane
            p3: third point on the plane
            identifier: id for the plane
        """
        v1 = p2 - p1
        v2 = p3 - p1
        self.norm = np.cross(v1, v2)
        self.a = self.norm[0]
        self.b = self.norm[1]
        self.c = self.norm[2]
        self.d = np.dot(self.norm, p3)
        self.points = [p1, p2, p3]
        self.identity = identifier

    def point_is_on(self, point, tol=0.05):
        """
        Evaluate if a point lies along the plane.

        Args:
            point (list): the point under assessment
            tol (float): % tolerence on assessment
        """
        substitution = point[0] * self.a + point[1] * self.b + point[2] * self.c
        return self.d * (1 - tol) < substitution < self.d * (1 + tol)


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
        connectivity_matrix = self._get_connectivity_matrix(self.slab).toarray()
        cns = [sum(connectivity_matrix[idx]) for idx, tag in  enumerate(self.slab.get_tags()) if tag == 1]
        proportion_cns = {}
        for cn in np.unique(cns):
            proportion_cns[cn] = cns.count(cn) / len(cns)
        cn_info = {"mean": np.mean(cns), "proportions": proportion_cns}

        return cn_info

    def get_surface_gcn_info(self):
        """
        Calculates the surface generalized coordination numbers (gcn) for each surface atom which is used to
        return (1) the mean surface gcn (2) the minimum surface gcn (3) the maximum surface gcn
        
        Returns:
            (dict): the generalized coordination info. ex. `{"mean": 5.5, "min": 5, "max": 6}
        """
        connectivity_matrix = self._get_connectivity_matrix(self.slab).toarray()
        cns = np.sum(connectivity_matrix, axis=1)
        surface_idx = np.array([idx for idx, tag in  enumerate(self.slab.get_tags()) if tag == 1])
        cn_matrix = np.array(cns)*connectivity_matrix
        gcn = np.sum(cn_matrix, axis=1)/12
        gcn = gcn[surface_idx]
        gcn_info = {"mean": np.mean(gcn), "min": np.min(gcn), "max": np.max(gcn)}

        return gcn_info


    def find_nodes_per_area(self, thresh):
        surface_atoms = self.slab[
            [idx for idx, tag in enumerate(self.slab.get_tags()) if tag == 1]
        ]
        surface_atoms = surface_atoms.repeat((3, 3, 1))

        cell_vectors = AseAtomsAdaptor.get_structure(self.slab).lattice.matrix[:, 0:2]
        cell_vectors = [
            vector for vector in cell_vectors if not np.allclose([0, 0], vector)
        ]

        delaunay_2d_mesh = Delaunay(surface_atoms.get_positions()[:, 0:2])
        new_positions, new_faces = simplify_mesh(
            np.array(surface_atoms.get_positions(), np.double),
            np.array(delaunay_2d_mesh.simplices, np.uint32),
            3,
            max_err=thresh,
        )

        planes_3x = [
            Plane(
                new_positions[new_face[0]],
                new_positions[new_face[1]],
                new_positions[new_face[2]],
                idx,
            )
            for idx, new_face in enumerate(new_faces)
        ]
        planes_3x = self._drop_edge_planes(planes_3x, cell_vectors)
        planes_3x = self._drop_duplicate_planes(planes_3x)

        all_points = [plane.points for plane in planes_3x]
        directed_area = np.cross(*cell_vectors)
        envelope_area = np.sqrt(directed_area.dot(directed_area))
        return (
            len(planes_3x) / envelope_area
        )  # per area doesnt make sense because some features are infinite. What can I do instead?

    def _drop_edge_planes(self, planes, cell_vectors):
        planes_to_keep = []
        inner_slab = Polygon(
            [
                cell_vectors[0] + cell_vectors[1],
                2 * cell_vectors[0] + cell_vectors[1],
                cell_vectors[0] + 2 * cell_vectors[1],
                2 * cell_vectors[0] + 2 * cell_vectors[1],
            ]
        )
        for plane in planes:
            plane_polygon = Polygon(np.array(plane.points)[:, 0:2])
            if plane_polygon.intersects(inner_slab):
                planes_to_keep.append(plane)
        return planes_to_keep

    def _drop_duplicate_planes(self, planes):
        planes_reduced = copy.deepcopy(planes)
        for plane in planes:
            idx = plane.identity
            planes_to_compare = [
                plane_now for plane_now in planes_reduced if (plane_now.identity != idx)
            ]
            norms = [plane_now.norm for plane_now in planes_to_compare]
            norm_mag = np.sqrt(plane.norm.dot(plane.norm))
            norms_cos = [
                round(
                    np.dot(plane.norm, norm_now)
                    / (np.sqrt(norm_now.dot(norm_now)) * norm_mag),
                    3,
                )
                for norm_now in norms
            ]
            norms_same = [
                bool_index
                for bool_index, cos in enumerate(norms_cos)
                if (np.abs(cos) >= 0.996)
            ]
            for true_index in norms_same:
                if planes_to_compare[true_index].point_is_on(plane.points[0]):
                    idx_to_delete = [
                        idx_on
                        for idx_on, val in enumerate(planes_reduced)
                        if (val.identity == plane.identity)
                    ][0]
                    planes_reduced.pop(idx_to_delete)
                    break
        return planes_reduced
