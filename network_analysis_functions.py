import bct # brain connectivity toolbox
import numpy as np
import datetime as dt
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu

def getBiggestComponent(pairwise_measure_matrix):
    """
    Get the biggest component of the weighted adjacency matrix.
    Arguments:  pairwise_measure_matrix, numpy.ndarray
    Returns:    biggest_comp, the biggest component of the weighted adjacency matrix
                keep_indices, the indices of the nodes in the biggest component
                comp_assign, integers indicating to which component each node was designated
                comp_size, the size of each component
    Note: Requires bct, the python version of the brain connectivity toolbox
            https://github.com/aestrivex/bctpy
    """
    adjacency_matrix = (pairwise_measure_matrix > 0).astype(int)
    comp_assign, comp_size = bct.get_components(adjacency_matrix)
    keep_indices = np.nonzero(comp_assign == comp_size.argmax() + 1)[0]
    biggest_comp = pairwise_measure_matrix[keep_indices][:,keep_indices]
    np.fill_diagonal(biggest_comp, 0)
    return biggest_comp, keep_indices, comp_assign, comp_size

def kMeansSweep(e_vectors, min_groups, max_groups, reps, dims):
    """
    Perform kmeans clustering on the given eigen vectors sweeping from min_groups clusters to max_groups clusters, repeating reps number of times.
    Arguments:  e_vectors, the eigenvectors to cluster
                min_groups, the minimum number of clusters to be used
                max_groups, the maximum number of clusters to be used
                reps, the number of clusterings to take for each integer value between min_groups and max_groups
                dims, string, options = ['all', 'scale'] either use all the e_vectors or scale accourding to the number of clusters we're trying to find.
    """
    assert min_groups >= 2, dt.datetime.now().isoformat() + ' ERROR: ' + 'Minimum number of groups must be at least 2!'
    assert min_groups <= max_groups, dt.datetime.now().isoformat() + ' ERROR: ' + 'Minimum number of groups greater than maximum number of groups!'
    num_nodes, num_e_vectors = e_vectors.shape
    if (dims == 'scale') & (num_e_vectors < max_groups - 1):
        sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Not enough embedding dimensions to scale upper bound!')
    num_possible_groups = 1 + max_groups - min_groups
    clusterings = np.zeros([num_nodes, reps], dtype=int)
    for num_groups in range(min_groups, max_groups + 1):
        if dims == 'all':
            this_vector = e_vectors
        elif dims == 'scale':
            this_vector = e_vectors[:, num_groups-1]
        else:
            sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unknown dims!')
        for i in range(reps):
            kmeans_estimator = KMeans(init='k-means++', n_clusters=num_groups, n_init=10)
            kmeans_estimator.fit(this_vector)
            clusterings[:,i] = kmeans_estimator.labels_ # each column is a clustering
    return clusterings

def getClusteringModularity(clustering, modularity_matrix, m=1):
    """
    For calculating the modularity of a clustering.
    Arguments:  clustering, numpy.ndarry (num_nodes) (int), an array of labels indicting to which cluster each node belongs
                modularity_matrix, numpy.ndarray (num nodes, num noes), the modularity matrix of the network
                m, the number of unique edges or sum of unique weights, used in normalising the modularity
    Returns:    modularity, float
    Reference:  Newman, M. E. J. (2006) "Finding community structure in networks using the eigenvectors of matrices". Phys Rev E, 74, 036104.
    """
    num_nodes = clustering.size
    num_clusters = clustering.max() + 1
    membership_matrix = np.zeros([num_nodes, num_clusters], dtype=int)
    for k in range(num_nodes):
        membership_matrix[k, clustering[k]] = 1
    modularity = np.matrix.trace(np.dot(np.dot(membership_matrix.T, modularity_matrix), membership_matrix)) / (2*m)
    return modularity

def checkConvergenceConsensus(consensus_matrix):
    """
    Checks if the consensus matrix contains a single clustering of the nodes.
    Arguments:  consensus_matrix, numpy.ndarray (num nodes, num nodes), a symmetric matrix indicating the number of times
                any two nodes are in the same clustering, over a number of clusterings, 0 along the main diagonal
    Returns:    is_converged, boolean, flag for whether or not the consensus matrix has converged
                consensus_clustering[sort_inds], the consensus clustering
                threshold, the detected threshold that minimises the intra-class variance
    """
    num_nodes = consensus_matrix.shape[0]
    pair_rows, pair_cols = np.array(list(combinations(range(num_nodes),2))).T # upper triangle indices
    consensuses = consensus_matrix[pair_rows, pair_cols] # order differs from Matlab
    bin_width = 0.01 # Otsu's method
    if (consensuses == 1).all():
        threshold = -np.inf
    else:
        bin_counts, edges = np.histogram(consensuses, bins = np.arange(consensuses.min(), consensuses.max(), bin_width))
        threshold_bin = threshold_otsu(bin_counts)
        threshold = edges[threshold_bin]
    high_consensus = consensus_matrix.copy()
    high_consensus[consensus_matrix < threshold] = 0
    consensus_clustering = np.array([], dtype=int)
    clustered_nodes = np.array([], dtype=int)
    is_converged = False
    consensus_clustering_iterations = 0
    for i in range(num_nodes):
        if not(i in clustered_nodes): # if not already sorted
            this_cluster = np.hstack([i, np.flatnonzero(high_consensus[i,:] > 0)])
            # if any of the nodes in this cluster are already in a cluster, then this is not transitive
            if np.intersect1d(clustered_nodes, this_cluster).size > 0:
                is_converged = False
                return is_converged, consensus_clustering, threshold
            else:
                is_converged = True
            consensus_clustering = np.hstack([consensus_clustering, consensus_clustering_iterations * np.ones(this_cluster.size, dtype=int)])
            clustered_nodes = np.hstack([clustered_nodes, this_cluster])
            consensus_clustering_iterations += 1
    sort_inds = np.argsort(clustered_nodes)
    return is_converged, consensus_clustering[sort_inds], threshold
