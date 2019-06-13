"""
Script to show off the noise rejection part of Network Noise Rejection. Data taken from the original Network Noise Rejection repo: https://github.com/mdhumphries/NetworkNoiseRejection
"""
import os, sys, argparse
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import __init__ as nnr # network noise rejection
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.io import loadmat

### NB assuming that this script will be run from the project root directory
proj_dir = os.environ['PWD']
mat_dir = os.path.join(proj_dir, 'mat')
npy_dir = os.path.join(proj_dir, 'npy')
data_file_name = 'lesmis.mat'
data_file = os.path.join(mat_dir, data_file_name)

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading data...')
data = loadmat(data_file)
weighted_adjacency_matrix = data['Problem']['A'][0][0].todense().A
weighted_adjacency_matrix = nnr.checkDirected(weighted_adjacency_matrix)
weighted_adjacency_matrix, biggest_comp_indices, comp_assign, comp_size = nnr.getBiggestComponent(weighted_adjacency_matrix)

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Getting E-values and <P> of sparse WCM...')
samples_eig_vals, optional_returns = nnr.getPoissonWeightedConfModel(weighted_adjacency_matrix, 100, is_sparse=True, return_eig_vecs=True)
samples_eig_vecs = optional_returns['eig_vecs']
expected_wcm = optional_returns['expected_wcm']

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Constructing network modularity matrix...')
network_modularity_matrix = weighted_adjacency_matrix - expected_wcm

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Getting low dimensional space...')
below_eig_space, below_lower_bound_inds, [mean_mins_eig, min_confidence_ints], exceeding_eig_space, exceeding_upper_bound_inds, [mean_maxs_eig, max_confidence_ints] = nnr.getLowDimSpace(network_modularity_matrix, samples_eig_vals, 0, int_type='CI')
exceeding_space_dims = exceeding_eig_space.shape[1]
below_space_dims = below_eig_space.shape[1]

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Splitting network into noise and signal...')
reject_dict = nnr.nodeRejection(network_modularity_matrix, samples_eig_vals, 0, samples_eig_vecs, weight_type='linear', norm='L2', int_type='CI', bounds='upper')
signal_weighted_adjacency_matrix = weighted_adjacency_matrix[reject_dict['signal_inds']][:, reject_dict['signal_inds']]

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Constructing final signal network without leaves...')
biggest_signal_comp, biggest_signal_inds, biggest_signal_assing, biggest_signal_size = nnr.getBiggestComponent(signal_weighted_adjacency_matrix)
signal_comp_inds = reject_dict['signal_inds'][biggest_signal_inds]
strength_distn = biggest_signal_comp.sum(axis=0)
leaf_inds = np.flatnonzero(strength_distn == 1)
keep_inds = np.flatnonzero(strength_distn > 1)
signal_final_inds = signal_comp_inds[keep_inds]
signal_leaf_inds = signal_comp_inds[leaf_inds]
final_weighted_adjacency_matrix = biggest_signal_comp[keep_inds][:, keep_inds]

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saving data...')
np.save(os.path.join(npy_dir, 'lesmis_reject_dict.npy'), reject_dict)
np.save(os.path.join(npy_dir, 'lesmis_weighted_adjacency_matrix.npy'), weighted_adjacency_matrix)
np.save(os.path.join(npy_dir, 'lesmis_exceeding_space_dims.npy'), exceeding_space_dims)
np.save(os.path.join(npy_dir, 'lesmis_signal_final_inds.npy'), signal_final_inds)
np.save(os.path.join(npy_dir, 'lesmis_expected_wcm.npy'), expected_wcm)
np.save(os.path.join(npy_dir, 'lesmis_final_weighted_adjacency_matrix.npy'), final_weighted_adjacency_matrix)
np.save(os.path.join(npy_dir, 'lesmis_nodelabels.npy'), data['Problem']['aux'][0][0]['nodename'][0][0])

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Plotting modularity matrix eigenspectrum against null eigenspectrum...')
plt.axvline(x=mean_mins_eig, ymin=0, ymax=1, color='black', label='lower bound')
plt.axvline(x=mean_maxs_eig, ymin=0, ymax=1, color='black', label='upper bound')
plt.axvline(x=0, ymin=0, ymax=1, color='black', linestyle='--')
plt.scatter(np.linalg.eigvalsh(network_modularity_matrix), 0.5*np.ones(network_modularity_matrix.shape[0]), color='blue', marker='.', label='modularity matrix eigenvalues')
plt.xlim([-70, 70])
plt.xlabel('Eigenvalues')
plt.yticks([])
plt.legend()
plt.tight_layout()
plt.show(block=False)
