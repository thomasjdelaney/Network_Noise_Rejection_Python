import bct # brain connectivity toolbox
import numpy as np
import datetime as dt
from sklearn.cluster import KMeans
from itertools import combinations

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
    Perform k-means clustering on the given eigen vectors sweeping from min_groups clusters to max_groups clusters, repeating reps number of times.
    Arguments:  e_vectors, the eigenvectors to cluster
                min_groups, the minimum number of clusters to be used
                max_groups, the maximum number of clusters to be used
                reps, the number of clusterings to take for each integer value between min_groups and max_groups
                dims, string, options = ['all', 'scale'] either use all the e_vectors or scale accourding to the number of clusters we're trying to find.
    Returns:    clusterings, numpy.ndarray (num nodes, (1+max_groups-min_groups)*reps), the clusterings resulting from running k-means clustering repeatedly
    """
    assert min_groups >= 2, dt.datetime.now().isoformat() + ' ERROR: ' + 'Minimum number of groups must be at least 2!'
    assert min_groups <= max_groups, dt.datetime.now().isoformat() + ' ERROR: ' + 'Minimum number of groups greater than maximum number of groups!'
    num_nodes, num_e_vectors = e_vectors.shape
    if (dims == 'scale') & (num_e_vectors < max_groups - 1):
        sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Not enough embedding dimensions to scale upper bound!')
    num_diff_num_possible_groups = 1 + max_groups - min_groups # number of different number of possible groups
    clusterings = np.zeros([num_nodes, num_diff_num_possible_groups * reps], dtype=int)
    for i, num_groups in enumerate(range(min_groups, max_groups + 1)):
        if dims == 'all':
            this_vector = e_vectors
        elif dims == 'scale':
            this_vector = e_vectors[:, num_groups-1]
        else:
            sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unknown dims!')
        for j in range(reps):
            kmeans_estimator = KMeans(init='k-means++', n_clusters=num_groups, n_init=10)
            kmeans_estimator.fit(this_vector)
            clusterings[:,(reps * i) + j] = kmeans_estimator.labels_ # each column is a clustering
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

def otsusMethod(bin_counts):
    """
    implementing Otsu's method for finding a threshold to divide a histogram into two classes with maximal inter-class variance.
    Arguments:  bin_counts, counts of a histogram
    Returns:    thresh, int, the threshold bin, bin_counts[:thresh] are in class 0, bin_counts[thresh+1:end] are in class 2.
                eta_max, turning point
    References: Otsu, N (1979) A Threshold Selection Method from Gray-Level Histograms, IEEE Trans Sys Man Mach, 9, 62-66
    """
    if bin_counts.sum() != 1:
        bin_counts = bin_counts / bin_counts.sum().astype(float)
    global_mean = (np.arange(1, bin_counts.size+1) * bin_counts).sum()
    global_var = (np.power((np.arange(1, bin_counts.size+1) - global_mean),2) * bin_counts).sum()
    maximum = 0
    thresh = 1
    cum_prob = 0 # cumulative probability
    cum_mean = 0 # cumulative mean
    eta_max = 0
    inter_var = 0
    for i in range(1, bin_counts.size):
        cum_prob = cum_prob + bin_counts[i-1]
        cum_mean = cum_mean + i*bin_counts[i-1]
        if cum_prob != 1:
            inter_var = np.power((global_mean * cum_prob - cum_mean),2) / (cum_prob * (1-cum_prob)) # inter-class variance
        if inter_var > maximum:
            thresh = i
            maximum = inter_var
            eta_max = inter_var / global_var
    return thresh, eta_max

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
        threshold_bin, eta_max = otsusMethod(bin_counts)
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

def nullModelConsensusSweep(cluster_sizes, num_allowed_clusterings, num_nodes):
    """
    Constructs the expected null model for a consensus matrix of random clusterings from a k-means sweep.
    Arguments:  cluster_sizes, numpy.ndarray (num clusters), number of clusters tested in each k-means run
                num_allowed_clusterings, numpy.ndarray (num clusters), the number of clusterings taken
                num_nodes, int, the number of nodes
    Returns:    exp_proportions, numpy.ndarray (num nodes, num nodes), matrix of expected proportions
                proportion_var, numpy.ndarray (num nodes, num nodes), matrix of variances in those expected proportions
    """
    assert cluster_sizes.size == num_allowed_clusterings.size, dt.datetime.now().isoformat() + ' ERROR: ' + 'number of cluster sizes != number of allowed clusterings'
    num_draws = np.zeros(cluster_sizes.shape)
    draw_var = np.zeros(cluster_sizes.shape)
    for i, (cluster_size, num_allowed) in enumerate(zip(cluster_sizes, num_allowed_clusterings)):
        uniform_same_cluster_prob = 1/float(cluster_size)
        num_draws[i] = num_allowed * uniform_same_cluster_prob
        draw_var[i] = uniform_same_cluster_prob * num_allowed * (1 - uniform_same_cluster_prob)
    exp_proportions = np.zeros([num_nodes, num_nodes]) + num_draws.sum()/num_allowed_clusterings.sum()
    proportion_var = np.zeros([num_nodes, num_nodes]) + draw_var.sum()/num_allowed_clusterings.sum()
    np.fill_diagonal(exp_proportions, 0.0)
    np.fill_diagonal(proportion_var, 0.0)
    return exp_proportions, proportion_var

def embedConsensusNull(consensus_matrix, consensus_type, cluster_sizes, num_allowed_clusterings):
    """
    For getting the low dimensional projection of a consensus matrix.
    Arguments:  consensus_matrix, numpy.ndarray (num nodes, num nodes), a symmetric matrix indicating the number of times
                any two nodes are in the same clustering, over a number of clusterings, 0 along the main diagonal
                consensus_type, string, options = ['sweep', 'expect'], how the consensus matrix was built. 'expect' is not yet implemented
                cluster_sizes, numpy.ndarray (num clusters), number of clusters tested in each k-means run
                num_allowed_clusterings, numpy.ndarray (num clusters), the number of clusterings taken
    Returns:    low_d_consensus, numpy.ndarray (num nodes, n), eigenvectors of the consensus network's modularity matrix with corresponding eigenvalues > 0
                cons_mod_matrix, numpy.ndarray (num nodes, num nodes), the consensus network's modularity matrix
                est_num_groups, int, the estimated number of groups/clusters to be found in the consensus network
                cons_eig_vals, numpy.ndarray (num nodes), the eigenvalues of the consensus network's modularity matrix
    """
    num_nodes = consensus_matrix.shape[0]
    if consensus_type == 'sweep':
        exp_proportions, proportion_var = nullModelConsensusSweep(cluster_sizes, num_allowed_clusterings, num_nodes)
    elif consensus_type == 'expect':
        return 0 # implement this
    else:
        sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unknown consensus type!')
    cons_mod_matrix = consensus_matrix - exp_proportions
    cons_eig_vals, cons_eig_vecs = np.linalg.eigh(cons_mod_matrix)
    low_d_consensus = cons_eig_vecs[:,cons_eig_vals > 0]
    est_num_groups = (cons_eig_vals > 0).sum() + 1
    return low_d_consensus, cons_mod_matrix, est_num_groups, cons_eig_vals

def consensusCommunityDetect(signal_measure_matrix, signal_expected_wcm, min_groups, max_groups, kmeans_reps=100, dims='all', is_explore=True):
    """
    For partitioning the signal network using eigenvectors of the signal modularity matrix.
    Arguments:  signal_measure_matrix, numpy.ndarray (num nodes, num nodes) (signal nodes), undirected signal network matrix
                signal_expected_wcm, numpy.ndarray (num nodes, num nodes) (signal nodes), expected null model
                min_groups, int, minimum number of groups/clusters to detect
                max_groups, int, maximum number of groups/clusters to detect
                kmeans_reps, int, number of reps to run during the k-means sweep
                dims, string, options = ['all', 'scale'] either use all the e_vectors or scale accourding to the number of clusters we're trying to find.
                is_explore, boolean, if min_groups == max_groups then mark this flag as True in order to allow the network to explore greater numbers of
                    groups/clusters
    Returns:    max_mod_cluster, the clustering corresponding to the maximum modularity value
                max_modularity, the maximum modularity
                consensus_clustering, the clustering found by consensus community detection
                consensus_modularity, the modularity of the consensus_clustering
                consensus_iterations, the number of iterations used to reach the consensus clustering
    References: (1) Newman, M. E. J. (2006) "Finding community structure in networks using the eigenvectors of matrices". Phys Rev E, 74, 036104.
                (2) Reichardt & Bornhaldt (2006) "Statistical mechanics of community detection". Phys Rev E. 74, 016110
                (3) Lancichinetti, A. & Fortunato, S. (2012) Consensus clustering in complex networks. Scientific Reports, 2, 336
                (4) Arthur, D. & Vassilvitskii, S. (2007) k-means++: the advantages of careful seeding. SODA '07: Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms, Society for Industrial and Applied Mathematics, 1027-1035
    """
    if (np.diag(signal_measure_matrix) != 0).any():
        print(dt.datetime.now().isoformat() + ' WARN: ' + 'Measure matrix has self loops...')
    if (signal_measure_matrix < 0).any():
        print(dt.datetime.now().isoformat() + ' WARN: ' + 'Measure matrix has negative values...')
    num_nodes = signal_measure_matrix.shape[0]
    total_unique_weight = signal_measure_matrix.sum()/2.0
    consensus_iterations = 1
    is_converged = False
    modularity_matrix = signal_measure_matrix - signal_expected_wcm
    # mod_eig_vals, mod_eig_vecs = getDescSortedEigSpec(modularity_matrix)
    mod_eig_vals, mod_eig_vecs = np.linalg.eigh(modularity_matrix)
    e_vectors = mod_eig_vecs[:,1-max_groups:]
    kmeans_clusterings = kMeansSweep(e_vectors, min_groups, max_groups, kmeans_reps, dims) # C, can't get the clusterings to match
    clustering_modularities = np.array([getClusteringModularity(clustering, modularity_matrix, total_unique_weight) for clustering in kmeans_clusterings.T]) # Q
    if (kmeans_clusterings == 0).all() | (clustering_modularities <= 0).all():
        return 0 # return empty results
    max_modularity = clustering_modularities.max()
    max_mod_cluster = kmeans_clusterings[:,clustering_modularities.argmax()]
    while not(is_converged):
        allowed_clusterings = kmeans_clusterings[:,clustering_modularities > 0]
        consensus_matrix = bct.agreement(allowed_clusterings) / float(kmeans_clusterings.shape[1])
        is_converged, consensus_clustering, threshold = checkConvergenceConsensus(consensus_matrix) # doesn't match. Some investigation required.
        if is_converged:
            consensus_modularity = getClusteringModularity(consensus_clustering, modularity_matrix, total_unique_weight)
        else:
            consensus_iterations += 1
            if consensus_iterations > 50:
                print(dt.datetime.now().isoformat() + ' WARN: ' + 'Not converged after 50 reps. Exiting...')
                consensus_clustering = np.array([]); consensus_modularity = 0.0;
                return max_mod_cluster, max_modularity, consensus_clustering, consensus_modularity, consensus_iterations
            else:
                if (min_groups == max_groups) & (not(is_explore)):
                    num_allowed_clusterings = (clustering_modularities>0).reshape([kmeans_reps, 1+ max_groups - min_groups]).sum(axis=0)
                    low_d_consensus, cons_mod_matrix, est_num_groups, cons_eig_vals = embedConsensusNull(consensus_matrix, 'sweep', np.arange(min_groups, max_groups+1), num_allowed_clusterings)
                    if est_num_groups >= max_groups:
                        kmeans_clusterings = kMeansSweep(low_d_consensus[:,1-max_groups:], min_groups, max_groups, kmeans_reps, dims)
                    elif (low_d_consensus == 0).all():
                        kmeans_clusterings = np.array([])
                    else:
                        kmeans_clusterings = kMeansSweep(low_d_consensus, min_groups, max_groups, kmeans_reps, dims)
                if (min_groups != max_groups) | is_explore:
                    num_allowed_clusterings = (clustering_modularities>0).reshape([kmeans_reps, 1+ max_groups - min_groups]).sum(axis=0)
                    low_d_consensus, cons_mod_matrix, est_num_groups, cons_eig_vals = embedConsensusNull(consensus_matrix, 'sweep', np.arange(min_groups, max_groups + 1), num_allowed_clusterings)
                    max_groups = est_num_groups
                    if max_groups < min_groups: min_groups = max_groups
                    kmeans_clusterings = kMeansSweep(low_d_consensus, min_groups, max_groups, kmeans_reps, dims)
                if (kmeans_clusterings == 0.0).all():
                    print(dt.datetime.now().isoformat() + ' WARN: ' + 'Consensus matrix projection is empty. Exiting...')
                    consensus_clustering = np.array([]); consensus_modularity = 0.0;
                    return max_mod_cluster, max_modularity, consensus_clustering, consensus_modularity, consensus_iterations
                else:
                    clustering_modularities = np.array([getClusteringModularity(clustering, modularity_matrix, total_unique_weight) for clustering in kmeans_clusterings.T])
    return max_mod_cluster, max_modularity, consensus_clustering, consensus_modularity, consensus_iterations
