"""
Script to show off the clustering part of Network Noise Rejection. Data taken from the original Network Noise Rejection repo: https://github.com/mdhumphries/NetworkNoiseRejection
"""
import os, sys, argparse
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import __init__ as nnr # network noise rejection
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

### NB assuming that this script will be run from the project root directory
proj_dir = os.environ['PWD']
mat_dir = os.path.join(proj_dir, 'mat')
npy_dir = os.path.join(proj_dir, 'npy')

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading data...')
reject_dict = np.load(os.path.join(npy_dir, 'lesmis_reject_dict.npy'))
weighted_adjacency_matrix = np.load(os.path.join(npy_dir, 'lesmis_weighted_adjacency_matrix.npy'))
exceeding_space_dims = np.load(os.path.join(npy_dir, 'lesmis_exceeding_space_dims.npy'))
signal_final_inds = np.load(os.path.join(npy_dir, 'lesmis_signal_final_inds.npy'))
expected_wcm = np.load(os.path.join(npy_dir, 'lesmis_expected_wcm.npy'))
final_weighted_adjacency_matrix = np.load(os.path.join(npy_dir, 'lesmis_final_weighted_adjacency_matrix.npy'))
node_labels = np.load(os.path.join(npy_dir, 'lesmis_nodelabels.npy'))

signal_expected_wcm = expected_wcm[signal_final_inds][:, signal_final_inds]
max_mod_cluster, max_modularity, consensus_clustering, consensus_modularity, consensus_iterations = nnr.consensusCommunityDetect(final_weighted_adjacency_matrix, signal_expected_wcm, exceeding_space_dims+1, exceeding_space_dims+1)
nnr.plotClusterMap(final_weighted_adjacency_matrix, consensus_clustering, is_sort=True, node_labels=node_labels[signal_final_inds])
plt.show()
