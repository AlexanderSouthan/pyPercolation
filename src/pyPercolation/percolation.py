# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import label
from tqdm import tqdm


class percolation:
    def __init__(self, lattice='square', lattice_points_1d=[50, 50]):
        self.lattice = lattice
        self.lattice_points_1d = np.asarray(lattice_points_1d)

        lattice_structures = ['square', 'cubic', 'triangular', 'fcc', 'bcc']
        if self.lattice == lattice_structures[0]:  # 'square'
            self.lattice_structure = np.array(
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        elif self.lattice == lattice_structures[1]:  # 'cubic'
            self.lattice_structure = np.array([
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
        elif self.lattice == lattice_structures[2]:  # 'triangular'
            self.lattice_structure = np.array(
                [[0, 1, 1], [1, 1, 1], [1, 1, 0]])
        elif self.lattice == lattice_structures[3]:  # 'fcc'
            self.lattice_structure = np.array([
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]]])
        elif self.lattice == lattice_structures[4]:  # 'bcc'
            self.lattice_structure = np.array([
                [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[1, 0, 1], [0, 0, 0], [1, 0, 1]]])
        else:
            raise ValueError(
                'No valid value for lattice. Allowed values are in {}.'.format(
                    lattice_structures))
        self.dimensions = self.lattice_structure.ndim
        if len(self.lattice_points_1d) != self.dimensions:
            raise ValueError('Lattice point dimension does not match lattice dimension.')

    def calc_clusters(self, n_p=100, n_iter=1, preserve_cluster_no=False):
        self.p = np.linspace(0, 1, n_p)
        # the absolute number of filled lattice points for each p value
        filled_points = np.around(
            np.prod(self.lattice_points_1d)*self.p).astype(int)
        self.percolation_p = np.zeros(n_iter)
        self.percolation_iter = np.zeros(n_iter)
        fill_order = np.indices(
            self.lattice_points_1d).reshape(
                self.dimensions,-1).T

        # the basic lattice
        self.lattices = np.full(
            (n_p, ) + tuple(self.lattice_points_1d),
            False, dtype='bool')
        # the lattice with surface bound clusters with 2, other clusters 1
        self.lattices_surface = np.zeros_like(self.lattices, dtype='uint8')
        # the lattice where clusters wil be identified and numbered
        self.clusters = np.zeros_like(self.lattices, dtype='uint32')
        # the cluster numbers of clusters at the sample surface and the sample bottom
        self.surface_clusters = [np.array([])]
        self.bottom_clusters = [np.array([])]
        # self.percolating_clusters = []

        for curr_iter in tqdm(np.arange(n_iter)):
        
            # reset all datasets for new iteration. This means that the cluster
            # information is only stored for the last iteration
            if curr_iter > 0:
                self.lattices[:] = False
                self.lattices_surface[:] = 0
                self.clusters[:] = 0
                self.surface_clusters = self.surface_clusters[0:1]
                self.bottom_clusters = self.bottom_clusters[0:1]
            # make sure that the fill order is random
            np.random.shuffle(fill_order)

            limit_found = False
            for curr_idx, curr_lattice in enumerate(self.lattices[1:, :, :]):
                curr_lattice[tuple(fill_order[filled_points[curr_idx]:filled_points[curr_idx+1]].T)] = 1

                self.clusters[curr_idx+1], _ = label(
                    curr_lattice, structure=self.lattice_structure)
                if preserve_cluster_no:
                    self.clusters[curr_idx+1] = self._preserve_cluster_numbers(
                        self.clusters[curr_idx+1], self.clusters[curr_idx])
                    
                curr_surface_clusters = np.unique(self.clusters[curr_idx+1, 0])
                curr_surface_clusters = curr_surface_clusters[curr_surface_clusters!=0]
                curr_bottom_clusters = np.unique(self.clusters[curr_idx+1, -1])
                curr_bottom_clusters = curr_bottom_clusters[curr_bottom_clusters!=0]
                self.surface_clusters.append(curr_surface_clusters)
                self.bottom_clusters.append(curr_bottom_clusters)

                if not limit_found and np.any(np.in1d(
                        self.surface_clusters[-1], self.bottom_clusters[-1])):
                    self.percolation_p[curr_iter] = self.p[curr_idx+1]
                    self.percolation_iter[curr_iter] = curr_idx + 1
                    limit_found = True
                # if limit_found:
                #     percolating_clusters.append(
                #         surface_clusters[-1][np.in1d(surface_clusters[-1], bottom_clusters[-1])])
                # else:
                #     percolating_clusters.append(False)

                # lattices_surface[curr_idx+1, np.isin(clusters[curr_idx+1], surface_clusters[-1])] = 1
                self.lattices_surface[curr_idx+1, np.isin(self.clusters[curr_idx+1], self.surface_clusters[curr_idx+1])] = 1
                self.lattices_surface[curr_idx+1] += curr_lattice

                if curr_idx < len(self.lattices)-2:
                    self.lattices[curr_idx+2] = curr_lattice

    def _preserve_cluster_numbers(self, new_clusters, old_clusters):
        old_numbers = np.unique(old_clusters)
        old_numbers = old_numbers[old_numbers>0]
    
        new_numbers = np.unique(new_clusters)
        new_numbers = new_numbers[new_numbers>0]
    
        max_number = np.max(np.concatenate([old_numbers, new_numbers]),
                            initial=0)
        new_range = np.arange(1, max_number+1)
        free_numbers = new_range[~np.isin(new_range, new_numbers) *
                                 ~np.isin(new_range, old_numbers)]
    
        not_checked = np.full_like(new_clusters, True, dtype=bool)
        preserved_clusters = np.zeros_like(new_clusters)
    
        for curr_new in new_numbers:
            curr_new_mask = new_clusters == curr_new
            curr_old = np.unique(old_clusters[curr_new_mask])
            curr_old = curr_old[curr_old>0]
            if np.sum(curr_old) == 0:
                brand_new = True
            else:
                brand_new = False
            if not brand_new:
                largest_cluster_idx = np.argmax(
                    [(old_clusters == curr).sum() for curr in curr_old])
                preserved_clusters[curr_new_mask*not_checked] = curr_old[largest_cluster_idx]
                free_numbers = np.sort(np.concatenate([free_numbers, np.delete(curr_old, largest_cluster_idx)]))
            else:
                if len(free_numbers) > 0:
                    brand_new_number = free_numbers[0]
                    free_numbers = free_numbers[1:]
                else:
                    brand_new_number = max_number + 1
                    max_number = max_number + 1
                preserved_clusters[curr_new_mask*not_checked] = brand_new_number
            not_checked[curr_new_mask] = False
    
        return preserved_clusters
