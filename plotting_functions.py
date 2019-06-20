import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def sortMatrixBySimilarity(clustering, measure_matrix):
    """
    Compute intra- and inter- group similarity and sort group/cluster order by similarity.
    Arguments:  clustering, numpy.ndarray (num nodes), an array of labels indicating to which group each node belongs
                measure_matrix, numpy.ndarray (num nodes, num nodes), weighted adjacency matrix
    Returns:    clustering sorted by similarity
                similarity_groups, numpy.ndarray (num groups), the mean similarity of each group
                similarity_in, numpy.ndarray (num nodes), the mean intragroup similarity of each node
                similarity_out,numpy.ndarray (num nodes), the mean intergroup similarity of each node
    """
    num_nodes = clustering.size
    groups = np.unique(clustering)
    num_groups = groups.size
    similarity_in = np.zeros(num_nodes); similarity_out = np.zeros(num_nodes); similarity_groups = np.zeros(num_groups)
    for i in range(num_nodes):
        this_group = clustering[i]
        in_group_clusters = np.setdiff1d(np.flatnonzero(clustering == this_group), this_group)
        out_group_clusters = np.flatnonzero(clustering != this_group)
        if in_group_clusters.size > 0:
            similarity_in[i] = measure_matrix[i, in_group_clusters].sum() / float(in_group_clusters.size)
        similarity_out[i] = measure_matrix[i, out_group_clusters].sum() / float(out_group_clusters.size)
    for g in groups:
        group_clusters = np.flatnonzero(clustering == g)
        similarity_groups[g] = similarity_in[group_clusters].sum() / float(group_clusters.size)
    sorted_similarity_inds = np.argsort(similarity_groups)
    return groups[sorted_similarity_inds], similarity_groups, similarity_in, similarity_out

def plotClusterMap(measure_matrix, clustering, is_sort=True, node_labels=np.array([None])):
    """
    Create matplotlib.pyplot figure object to display a sorted heatmap of detected clusters. Clusters are surrounded by white lines
    Arguments:  measure_matrix, numpy.ndarray (num nodes, num nodes), weighted adjacency matrix
                clustering, numpy.ndarray (num nodes), an array of labels indicating to which group each node belongs
                is_sort, boolean, to sort or not to sort
                node_labels, numpy.ndarray (num nodes) (string), labels for each node
    Returns:    Nothing
    """
    if is_sort:
        sorted_nodes = np.array([], dtype=int)
        sorted_clustering = np.array([], dtype=int)
        sorted_groups, similarity_groups, similarity_in, similarity_out = sortMatrixBySimilarity(clustering, measure_matrix)
        for g in sorted_groups:
            group_clusters = np.flatnonzero(clustering == g)
            group_clusters_similarity = similarity_in[group_clusters]
            sorted_similarity_inds = np.argsort(group_clusters_similarity)
            sorted_nodes = np.concatenate([sorted_nodes, group_clusters[sorted_similarity_inds]])
            sorted_clustering = np.concatenate([sorted_clustering, np.repeat(g, group_clusters.size)])
    else:
        sorted_clustering_inds = np.argsort(clustering)
        sorted_clustering = clustering[sorted_clustering_inds]
        sorted_nodes = sorted_clustering_inds
    cluster_changes = np.hstack([-1, np.flatnonzero(np.diff(sorted_clustering[::-1]) != 0), clustering.size-1]) + 0.5
    plt.figure()
    ax = plt.gca()
    im = ax.matshow(measure_matrix[sorted_nodes[::-1]][:,sorted_nodes[::-1]], cmap=cm.Blues_r)
    for i in range(cluster_changes.size-2):
        ax.plot([cluster_changes[i+1], cluster_changes[i+1]], [cluster_changes[i], cluster_changes[i+2]], color='white', linewidth=2.0)
        ax.plot([cluster_changes[i], cluster_changes[i+2]], [cluster_changes[i+1], cluster_changes[i+1]], color='white', linewidth=2.0)
    if (node_labels != None).all():
        plt.yticks(range(node_labels.size), node_labels[sorted_nodes[::-1]])
        plt.xticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    return None

def plotModEigValsVsNullEig(network_modularity_matrix, mean_mins_eig, mean_maxs_eig):
    mod_eig_vals = np.linalg.eigvalsh(network_modularity_matrix)
    plt.axvline(x=mean_mins_eig, ymin=0, ymax=1, color='black', label='lower bound')
    plt.axvline(x=mean_maxs_eig, ymin=0, ymax=1, color='black', label='upper bound')
    plt.axvline(x=0, ymin=0, ymax=1, color='black', linestyle='--')
    plt.scatter(mod_eig_vals, 0.5*np.ones(network_modularity_matrix.shape[0]), color='blue', marker='.', label='modularity matrix eigenvalues')
    plt.xlim([-10*np.ceil(mod_eig_vals.max()/10), 10*np.ceil(mod_eig_vals.max()/10)])
    plt.xlabel('Eigenvalues', fontsize='large')
    plt.yticks([])
    plt.legend(fontsize='large')
    plt.tight_layout()

def plotModEigValsVsNullEigHist(network_modularity_matrix, samples_eig_vals):
    mod_eig_vals = np.linalg.eigvalsh(network_modularity_matrix)
    plt.hist(samples_eig_vals.flatten(), bins=50, color='black', label='Null Model Samples Eigenvalue Distribution', alpha=0.4)
    y_max = plt.ylim()[1]/50.0
    plt.vlines(x=mod_eig_vals, ymin=0, ymax=y_max, color='blue', label='Modularity Matrix Eigenspectrum', alpha=0.4)
    plt.xlim([-10*np.ceil(mod_eig_vals.max()/10), 10*np.ceil(mod_eig_vals.max()/10)])
    plt.xlabel('Eigenvalues', fontsize='large')
    plt.ylabel('Num Eigenvalues', fontsize='large')
    plt.legend(fontsize='large')
    plt.tight_layout()
