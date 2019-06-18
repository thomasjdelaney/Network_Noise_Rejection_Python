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

parser = argparse.ArgumentParser(description='For showing off the noise clustering part of Network Noise Rejection')
parser.add_argument('-f', '--data_files_prefix', help='The prefix of the .npy files containing the required data.', type=str, default='lesmis')
args = parser.parse_args()

### NB assuming that this script will be run from the project root directory
proj_dir = os.environ['PWD']
mat_dir = os.path.join(proj_dir, 'mat')
npy_dir = os.path.join(proj_dir, 'npy')

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading data...')
reject_dict = np.load(os.path.join(npy_dir, args.data_files_prefix + '_reject_dict.npy'), encoding = 'latin1', allow_pickle=True)
weighted_adjacency_matrix = np.load(os.path.join(npy_dir, args.data_files_prefix + '_weighted_adjacency_matrix.npy'), encoding = 'latin1', allow_pickle=True)
exceeding_space_dims = np.load(os.path.join(npy_dir, args.data_files_prefix + '_exceeding_space_dims.npy'), encoding = 'latin1', allow_pickle=True)
signal_final_inds = np.load(os.path.join(npy_dir, args.data_files_prefix + '_signal_final_inds.npy'), encoding = 'latin1', allow_pickle=True)
expected_wcm = np.load(os.path.join(npy_dir, args.data_files_prefix + '_expected_wcm.npy'), encoding = 'latin1', allow_pickle=True)
final_weighted_adjacency_matrix = np.load(os.path.join(npy_dir, args.data_files_prefix + '_final_weighted_adjacency_matrix.npy'), encoding = 'latin1', allow_pickle=True)
node_labels = np.load(os.path.join(npy_dir, args.data_files_prefix + '_nodelabels.npy'), encoding = 'latin1', allow_pickle=True)

signal_expected_wcm = expected_wcm[signal_final_inds][:, signal_final_inds]
max_mod_cluster, max_modularity, consensus_clustering, consensus_modularity, consensus_iterations = nnr.consensusCommunityDetect(final_weighted_adjacency_matrix, signal_expected_wcm, exceeding_space_dims+1, exceeding_space_dims+1)
nnr.plotClusterMap(final_weighted_adjacency_matrix, consensus_clustering, is_sort=True, node_labels=node_labels[signal_final_inds])
plt.show()
