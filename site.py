"""A class for analyzing the nature of the site and how the adsorbate is bound."""

import numpy as np
from ase import neighborlist
from ase.neighborlist import natural_cutoffs


class SiteAnalyzer:
    def __init__(self, adslab, cutoff_multiplier):
        """
        Initialize class to handle site based analysis.

        Args:
            adslab (ase.Atoms): object of the slab with the adsorbate placed.
        """
        self.atoms = adslab
        self.cutoff_multiplier = cutoff_multiplier

    def find_binding_graph(self):
        tags = self.atoms.get_tags()
        elements = self.atoms.get_chemical_symbols()

        adsorbate_atom_idxs = [idx for idx, tag in enumerate(tags) if tag == 2]
        slab_atom_idxs = [idx for idx, tag in enumerate(tags) if tag != 2]

        connectivity = self._get_connectivity(self.atoms, self.cutoff_multiplier)

        binding_info = []
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
                }
                binding_info.append(ads_idx_info)
        return binding_info

    def get_dentate(self):
        binding_info = self.find_binding_graph()
        return len(binding_info)

    def _get_connectivity(self, atoms, cutoff_multiplier=1.0):
        """
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
