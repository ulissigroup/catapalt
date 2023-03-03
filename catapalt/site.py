"""A class for analyzing the nature of the site and how the adsorbate is bound."""

import numpy as np
from ase import neighborlist
from ase.neighborlist import natural_cutoffs
from scipy.spatial.distance import euclidean


class SiteAnalyzer:
    def __init__(self, adslab, cutoff_multiplier):
        """
        Initialize class to handle site based analysis.

        Args:
            adslab (ase.Atoms): object of the slab with the adsorbate placed.
        """
        self.atoms = adslab
        self.cutoff_multiplier = cutoff_multiplier
        self.binding_info = self._find_binding_graph()

    def _find_binding_graph(self):
        tags = self.atoms.get_tags()
        elements = self.atoms.get_chemical_symbols()

        adsorbate_atom_idxs = [idx for idx, tag in enumerate(tags) if tag == 2]
        slab_atom_idxs = [idx for idx, tag in enumerate(tags) if tag != 2]

        connectivity = self._get_connectivity(self.atoms, self.cutoff_multiplier)

        binding_info = []
        adslab_positions = self.atoms.get_positions()
        for idx in adsorbate_atom_idxs:
            if sum(connectivity[idx][slab_atom_idxs]) >= 1:
                bound_slab_idxs = [
                    idx_slab
                    for idx_slab in slab_atom_idxs
                    if connectivity[idx][idx_slab] == 1
                ]
                ads_idx_info = {
                    "adsorbate_idx": idx,
                    "adsorbate_element": elements[idx],
                    "slab_atom_elements": [
                        element
                        for idx_el, element in enumerate(elements)
                        if idx_el in bound_slab_idxs
                    ],
                    "slab_atom_idxs": bound_slab_idxs,
                    "bound_position": adslab_positions[idx]
                }
                binding_info.append(ads_idx_info)
        return binding_info

    def get_dentate(self):
        return len(self.binding_info)

    def _get_connectivity(self, atoms, cutoff_multiplier=1.0):
        """
        Note: need to condense this with the surface method
        Generate the connectivity of an atoms obj.
        
        Args:
            atoms (ase.Atoms): object which will have its connectivity considered
            cutoff_multiplier (float, optional): cushion for small atom movements when assessing
                atom connectivity
        Returns:
            (np.ndarray): The connectivity matrix of the atoms object.
        """
        cutoff = natural_cutoffs(atoms, mult=cutoff_multiplier)
        neighbor_list = neighborlist.NeighborList(
            cutoff, self_interaction=False, bothways=True
        )
        neighbor_list.update(atoms)
        matrix = neighborlist.get_connectivity_matrix(neighbor_list.nl).toarray()
        return matrix
    
    def get_bound_atom_positions(self):
        """
        Get the euclidean coordinates of all bound adsorbate atoms.
        
        Returns:
            (list): euclidean coordinats of bound atoms
        """
        positions = []
        for atom in self.binding_info:
            positions.append(atom["bound_position"])
        return positions
    
    def get_minimum_site_proximity(self, site_to_compare):
        """
        Note: might be good to check the surfaces are identical and raise an error otherwise.
        Get the minimum distance between bound atoms on the surface between two adsorbates.

        Args:
            site_to_compare (catapalt.SiteAnalyzer): site analysis instance of the other adslab.
            
        Returns: 
            (float): The minimum distance between bound adsorbate atoms on a surface.
        """  
        this_positions = self.get_bound_atoms_positions()
        other_positions = site_to_compare.get_bound_positions()
        distances = []
        for this_position in this_positions:
            distances.extend([euclidean(this_position, other_position) for other_position in other_positions])
        return min(distances)

