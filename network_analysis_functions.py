import bct # brain connectivity toolbox
import numpy as np
import datetime as dt

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

def checkDirected(network_matrix): # a symmetric matrix represents an undirected network
    """
    Check if the matrix representing the network represents an undirected network, and if not, convert it to an undirected network.
    Arguments:  network_matrix, numpy.ndarray
    Returns:    network_matrix
    """
    if (network_matrix == network_matrix.T).all():
        return network_matrix
    else:
        print(dt.datetime.now().isoformat() + ' WARN: ' + 'Network is undirected! Converting to directed...')
        return (network_matrix + network_matrix.T)/2.0

def getUnifyingScaleCoef(pairwise_measure_matrix):
    """
    Get the scaling coefficient that converts the weighted adjacency matrix so that the minimum non-zero weight is 1
    Arguments:  pairwise_measure_matrix
    Returns:    scaling coefficient, float64
    """
    return 1/pairwise_measure_matrix[pairwise_measure_matrix > 0].min()

def convertPairwiseMeasureMatrix(pairwise_measure_matrix, scale_coef = 100):
    """
    Convert the weighted adjacency matrix to integers by scaling up, and rounding off.
    Arguments:  pairwise_measure_matrix, numpy.ndarray
                scale_coef, int, the scaling coefficient can be provided beforehand.
    Returns:    pairwise_measure_int, numpy.ndarray, the converted weighted adjacency matrix.
                scale_coef, the scaling coefficient used.
    """
    pairwise_measure_int = pairwise_measure_matrix.astype(int)
    is_all_int = (pairwise_measure_matrix == pairwise_measure_int).all()
    if is_all_int:
        scale_coef = 1
    else:
        pairwise_measure_int = (scale_coef * pairwise_measure_matrix).round().astype(int)
    return pairwise_measure_int, scale_coef

def recoverPairwiseMeasureMatrix(pairwise_measure_int, scale_coef):
    """
    Convert the integer version of the weighted adjacency matrix back to its origin scale.
    Arguments:  pairwise_measure_int, numpy.ndarray (int)
                scale_coef, int, the scaling coefficient to use.
    Returns:    numpy.ndarray(float), the descaled weighted adjacency matrix
    """
    if scale_coef == 1:
        return pairwise_measure_int
    else:
        return pairwise_measure_int / scale_coef

def getExpectedNetworkFromData(pairwise_measure_int):
    """
    Get the expected network from the null model without modules.
    Arguments:  pairwise_measure_int, numpy.ndarray (int), integer version of the weighted adjacency matrix.
    Returns:    numpy.ndarray
    """
    return np.outer(np.sum(pairwise_measure_int, axis=0), np.sum(pairwise_measure_int, axis=1)) / np.sum(pairwise_measure_int).astype(float)

def getExpectedNetworkFromSamples(null_net_samples):
    num_samples = null_net_samples.shape[0]
    return null_net_samples.sum(axis=0)/num_samples
