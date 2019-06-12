import numpy as np
import datetime as dt
from scipy import stats

def getDescSortedEigSpec(m):
    """
    Get the eigenvalues and eigenvectors of m, sorted is descending order.
    Arguments:  m, a matrix
    Returns:    eig_vals, the eigenvalues sorted in descending order
                eig_vecs, the eigenvectors sorted in correspondance with eig_vals.
    """
    if (m == m.T).all(): # check symmetric
        eig_vals, eig_vecs = np.linalg.eigh(m)
    else:
        print(dt.datetime.now().isoformat() + ' WARN: ' + 'Input matrix is not symmetric...')
        eig_vals, eig_vecs = np.linalg.eig(m)
    desc_sort_inds = np.argsort(eig_vals)[::-1]
    return eig_vals[desc_sort_inds], eig_vecs[desc_sort_inds]

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
    """
    Get the expected null network from samples from the null network.
    Arguments:  null_net_samples, 3-d numpy.ndarray (num samples, num nodes, num nodes)
    Returns:    numpy.ndarray (num nodes, num nodes)
    """
    num_samples = null_net_samples.shape[0]
    return null_net_samples.sum(axis=0)/num_samples

def getPoissonRates(expected_weights, strength_distn, weight_to_place, has_loops=False):
    """
    Returns a matrix of Poisson rates used to sample a network with weighted connections.
    Arguments:  expected_weights, numpy.ndarray (num nodes, num nodes), the expected network
                strength_distn, numpy.ndarry (num_nodes), the strength of each node
                weight_to_place, int, the total weight that needs to be distributed in the network
                has_loops, boolean, flag for allowed self-connections (non-zero entries on the main diagonal)
    Returns:    poisson_rates, numpy.ndarray (num_nodes, num nodes)
    """
    poisson_rates = np.zeros(expected_weights.shape)
    expected_num_links = np.triu(expected_weights, k=int(not(has_loops)))
    pair_rows, pair_cols = np.nonzero(expected_num_links)
    prob_is_link = strength_distn[pair_rows] * strength_distn[pair_cols]
    prob_is_link = prob_is_link / prob_is_link.sum()
    poisson_rates[pair_rows, pair_cols] = weight_to_place * prob_is_link # the rate calculation and indexing all work perfectly, checked
    return poisson_rates

def sampleNullNetworkFullPoisson(poisson_rates, scale_coef):
    """
    Sample from the null network using the full poisson method.
    Arguments:  poisson_rates, numpy.ndarray (num nodes, num nodes), a matrix of poisson rates
                scale_coef, used to convert the sample from integers back to original network scale.
    Returns:    sample_net, numpy.ndarray (num nodes, num nodes), a weighted adjacency matrix sampled from the null model
    """
    sample_net = np.random.poisson(poisson_rates)
    sample_net = sample_net.T + sample_net # symmetrise
    sample_net = recoverPairwiseMeasureMatrix(sample_net, scale_coef)
    return sample_net

def sampleNullNetworkSparsePoisson(strength_distn, scale_coef, total_degrees, int_total_strength, total_weights, prob_link, has_loops):
    """
    Sample from the null network using the sparse poisson method.
    Arguments:  strength_distn, numpy.ndarry (num_nodes), the strength of each node
                scale_coef, int, the scaling coefficient to use.
                total_degrees, int, the sum of degree of the nodes in the network
                int_total_strength, int, the sum of the strength of the nodes in the scaled network
                total_weights, int, the sum of the weight of the nodes in the scaled network
                prob_link, numpy.ndarray (num nodes, num nodes) symmetric, the probability of a link between each node
                has_loops, boolean, flag for allowed self-connections (non-zero entries on the main diagonal)
    Returns:    numpy.ndarray (num nodes, num nodes), a weighted adjacency matrix samples from the null model using the sparse method
    """
    adjacency_sample = (np.random.rand(strength_distn.size, strength_distn.size) < np.triu(prob_link, k=1)).astype(int)
    pair_rows, pair_cols = np.nonzero(np.triu(adjacency_sample, k=0))
    num_links_to_place = int(round(int_total_strength/2.0)) - pair_rows.size # nLinks
    if (total_weights != total_degrees) & (num_links_to_place > 0): # we have a weighted network, with links to place
        poisson_rates = getPoissonRates(adjacency_sample, strength_distn, num_links_to_place, has_loops=has_loops)
        sampled_weights = np.random.poisson(poisson_rates)
        sample_net = sampled_weights + adjacency_sample
        sample_net = sample_net.T + sample_net
    else:
        sample_net = adjacency_sample.T + adjacency_sample
    return recoverPairwiseMeasureMatrix(sample_net, scale_coef)

def getSparsePoissonWeightedConfModel(pairwise_measure_matrix, pairwise_measure_int, num_samples, expected_net, strength_distn, total_weights, scale_coef, has_loops, return_type, return_eig_vecs):
    """
    For getting the eigenspectrum of the weighted configuration model using the sparse poisson method. The eigenvalues of each sample are automatically returned as the first returned value. The second returned value can contain the expected network without modules, or the expected network sampled from the sparse model, and the samples themselves, and even the eigenvectors of the samples.
    Arguments:  pairwise_measure_matrix, numpy.ndarray (num nodes, num nodes), the weighted adjacency matrix
                pairwise_measure_int, numpy.ndarray (num nodes, num nodes) (int), the scaled weighted adjacency matrix
                num_samples, int, the number of samples to take
                expected_net, numpy.ndarray (num nodes, num nodes), the expected network without modules
                strength_distn, numpy.ndarry (num_nodes), the strength of each node
                total_weights, int, the sum of the weight of the nodes in the scaled network
                scale_coef, int, the scaling coefficient to use
                has_loops, boolean, flag for allowed self-connections (non-zero entries on the main diagonal)
                return_type, string, choices = ['expected', 'both', 'all'], 'expected' makes the optional_returns dictionary contain the expected network calculated from the weighted configuration model,
                                                                            'both' makes the optional_returns dictionary contain the expected network calculated from the weighted configuration model, and all the samples from the null model,
                                                                            'all' makes the optional_returns dictionary contain the expected network without modules, and all the samples from the null model (bear in mind that the WCM expected network can be calculated from the samples using getExpectedNetworkFromSamples)
                return_eig_vecs, boolean, if true the optional_returns dictionary will contain the eigenvectors of each sample
    Returns:    samples_eig_vals, numpy.ndarray (num_samples, num_nodes), the eigenspectrum of each sample
                optional_returns,   'expected_wcm', numpy.ndarray (num_nodes, num_nodes), the expected network of the weighted configuration model
                                    'expected_net', numpy.ndarray (num_nodes, num_nodes), the expected network without modules
                                    'net_samples', numpy.ndarray (num_samples, num_nodes, num_nodes), all the sampled networks
                                    'eig_vecs', numpy.ndarray (num_samples, num_nodes, num_nodes), the eigenvectors of each sample network
    """
    num_nodes = pairwise_measure_matrix.shape[0]
    total_degrees = (pairwise_measure_matrix > 0).astype(int).sum() # K
    int_total_strength = pairwise_measure_int.sum()
    prob_link = getExpectedNetworkFromData(pairwise_measure_matrix > 0) # pnode, matches
    net_samples = np.zeros([num_samples, num_nodes, num_nodes])
    for i in range(num_samples):
        net_samples[i] = sampleNullNetworkSparsePoisson(strength_distn, scale_coef, total_degrees, int_total_strength, total_weights, prob_link, has_loops=has_loops)
    expected = getExpectedNetworkFromSamples(net_samples) if return_type in ['expected', 'both'] else expected_net
    samples_eig_vals = np.zeros([num_samples, num_nodes])
    samples_eig_vecs = np.zeros([num_samples, num_nodes, num_nodes])
    for i in range(num_samples):
        # samples_eig_vals[i], samples_eig_vecs[i] = getDescSortedEigSpec(net_samples[i] - expected)
        samples_eig_vals[i], samples_eig_vecs[i] = np.linalg.eigh(net_samples[i] - expected)
    if return_type == 'expected':
        optional_returns = {'expected_wcm':expected}
    elif return_type in 'all':
        optional_returns = {'expected_net':expected, 'net_samples':net_samples} # note the difference in key here
    elif return_type == 'both':
        optional_returns = {'expected_wcm':expected, 'net_samples':net_samples}
    else:
        sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unrecognised return type...')
    if return_eig_vecs:
        optional_returns['eig_vecs'] = samples_eig_vecs
    return samples_eig_vals, optional_returns

def getFullPoissonWeightedConfModel(num_samples, expected_net, strength_distn, total_weights, scale_coef, has_loops, return_type, return_eig_vecs):
    """
    For getting the eigenspectrum of the weighted configuration model using the full (non-sparse) poisson method. The eigenvalues of each sample are automatically returned as the first returned value. The second returned value can contain the expected network without modules, or the expected network sampled from the model, and the samples themselves, and even the eigenvectors of the samples.
    Arguments:  num_samples, int, the number of samples to take
                expected_net, numpy.ndarray (num nodes, num nodes), the expected network without modules
                strength_distn, numpy.ndarry (num_nodes), the strength of each node
                total_weights, int, the sum of the weight of the nodes in the scaled network
                scale_coef, int, the scaling coefficient to use
                has_loops, boolean, flag for allowed self-connections (non-zero entries on the main diagonal)
                return_type, string, choices = ['expected', 'both', 'all'], 'expected' makes the optional_returns dictionary contain the expected network calculated from the weighted configuration model,
                                                                            'both' makes the optional_returns dictionary contain the expected network calculated from the weighted configuration model, and all the samples from the null model,
                                                                            'all' makes the optional_returns dictionary contain the expected network without modules, and all the samples from the null model (bear in mind that the WCM expected network can be calculated from the samples using getExpectedNetworkFromSamples)
                return_eig_vecs, boolean, if true the optional_returns dictionary will contain the eigenvectors of each sample
    Returns:    samples_eig_vals, numpy.ndarray (num_samples, num_nodes), the eigenspectrum of each sample
                optional_returns,   'expected_wcm', numpy.ndarray (num_nodes, num_nodes), the expected network of the weighted configuration model
                                    'expected_net', numpy.ndarray (num_nodes, num_nodes), the expected network without modules
                                    'net_samples', numpy.ndarray (num_samples, num_nodes, num_nodes), all the sampled networks
                                    'eig_vecs', numpy.ndarray (num_samples, num_nodes, num_nodes), the eigenvectors of each sample network
    """
    poisson_rates = getPoissonRates(expected_net, strength_distn, total_weights, has_loops=has_loops)
    num_nodes = expected_net.shape[0]
    net_samples = np.zeros([num_samples, num_nodes, num_nodes])
    for i in range(num_samples):
        net_samples[i] = sampleNullNetworkFullPoisson(poisson_rates, scale_coef)
    expected = getExpectedNetworkFromSamples(net_samples) if return_type == ['expected', 'both'] else expected_net
    samples_eig_vals = np.zeros([num_samples, num_nodes])
    samples_eig_vecs = np.zeros([num_samples, num_nodes, num_nodes])
    for i in range(num_samples):
        # samples_eig_vals[i], samples_eig_vecs[i] = getDescSortedEigSpec(net_samples[i] - expected)
        samples_eig_vals[i], samples_eig_vecs[i] = np.linalg.eigh(net_samples[i] - expected)
    if return_type == 'expected':
        optional_returns = {'expected_wcm':expected}
    elif return_type in 'all':
        optional_returns = {'expected_net':expected, 'net_samples':net_samples}
    elif return_type == 'both':
        optional_returns = {'expected_wcm':expected, 'net_samples':net_samples}
    else:
        sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unrecognised return type...')
    if return_eig_vecs:
        optional_returns['eig_vecs'] = samples_eig_vecs
    return samples_eig_vals, optional_returns

def getConfidenceIntervalFromStdErr(st_dev, num_samples, interval):
    """
    For getting confidence intervals on the mean values of the null model.
    Arguments:  st_dev, float, standard deviation of the samples
                num_samples, int, the number of samples used
                interval, the confidence level to use, eg: 0.95, (to just use the mean, enter 0)
    Returns:    a confidence interval or list of confidence intervals matching the shape of st_dev
    """
    if interval == 0:
        if np.isscalar(st_dev):
            return 0.0
        else:
            return np.repeat(0.0, st_dev.shape)
    else:
        symm_interval = 1-(1-interval)/2.0
        t_val = stats.t.ppf(symm_interval, num_samples)
        return t_val * st_dev / np.sqrt(num_samples)

def getNonParaPredictionInterval(sample):
    """
    Use a non parametric method to estimate the prediction interval.
    Arguments:  sample, the sample from which to calculate the interval
    Returns:    array of prediction intervals with row or the format [confidence, lower bound, upper bound]
    """
    num_data_points = sample.size
    cutoff = int(num_data_points/2.0)
    ordered_sample = np.sort(sample)
    prediction_intervals = np.zeros([cutoff, 3])
    for i in range(cutoff):
        pi = (1 - 2*(i+1) / (num_data_points + 1.0))
        prediction_intervals[i,:] = [pi, ordered_sample[i], ordered_sample[num_data_points - 1 - i]]
    return prediction_intervals

def getLowDimSpace(modularity_matrix, eig_vals, confidence_level, int_type='CI'):
    """
    Find and return a low-dimensional space for the network.
    Arguments:  modularity_matrix, numpy.ndarray (num nodes, num noes), the modularity matrix of the network
                eig_vals, numpy.ndarry (num samples, num nodes), the null-model eigenspectrum distribution
                confidence_level, float, the specified confidence interval on the maximum eigenvalue, (enter 0 to just use the mean)
                int_type, string, the type of confidence interval to use, either 'CI' or 'PI'.
    Returns:    below_eig_space, numpy.ndarry (n_below, num nodes), the eigenvectors of the modularity matrix whose corresponding eigenvalues fall below the null eigenvalue distribution
                below_lower_bound_inds, numpy.ndarray (n_below) (int), the indices of the below_eig_space
                [mean_mins_eig, min_confidence_ints], the mean of the minimum eigenvalues of the null space, and the confidence interval on that mean
                exceeding_eig_space, numpy.ndarry (n_exceed, num nodes), the eigenvectors of the modularity matrix whose corresponding eigenvalues exceed the null eigenvalue distribution
                exceeding_upper_bound_inds, numpy.ndarray (n_exceed) (int), the indices of the exceeding_eig_space
                [mean_maxs_eig, max_confidence_ints], the mean of the maximum eigenvalues of the null space, and the confidence interval on that mean
    """
    assert modularity_matrix.shape[0] == eig_vals.shape[1], dt.datetime.now().isoformat() + ' ERROR: ' + 'Eigenvalue matrix is the wrong shape...'
    # mod_eig_vals, mod_eig_vecs = getDescSortedEigSpec(modularity_matrix)
    mod_eig_vals, mod_eig_vecs = np.linalg.eigh(modularity_matrix)
    mins_eig, maxs_eig = eig_vals.min(axis=1), eig_vals.max(axis=1)
    if int_type == 'CI':
        mean_mins_eig = mins_eig.mean()
        min_confidence_ints = getConfidenceIntervalFromStdErr(mins_eig.std(), eig_vals.shape[0], confidence_level)
        eig_lower_confidence_int = mean_mins_eig - min_confidence_ints
        mean_maxs_eig = maxs_eig.mean()
        max_confidence_ints = getConfidenceIntervalFromStdErr(maxs_eig.std(), eig_vals.shape[0], confidence_level)
        eig_upper_confidence_int = mean_maxs_eig + max_confidence_ints
    elif int_type == 'PI':
        if 1 == np.mod(maxs_eig.size, 2):
            prediction_intervals_min, prediction_intervals_max = getNonParaPredictionInterval(mins_eig), getNonParaPredictionInterval(maxs_eig)
        else:
            prediction_intervals_min, prediction_intervals_max = getNonParaPredictionInterval(mins_eig[0:-1]), getNonParaPredictionInterval(maxs_eig[0:-1])
        try:
            mean_mins_eig, min_confidence_ints = prediction_intervals_min[prediction_intervals_min[:,0] == confidence_level, 1:3][0]
            mean_maxs_eig, max_confidence_ints = prediction_intervals_max[prediction_intervals_max[:,0] == confidence_level, 1:3][0]
            eig_upper_confidence_int = prediction_intervals_max[prediction_intervals_max[:,0] == confidence_level, 2][0]
            eig_lower_confidence_int = prediction_intervals_min[prediction_intervals_min[:,0] == confidence_level, 1][0]
        except:
            sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Cannot find specified prediction interval!')
    else:
        sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unknown interval type!')
    exceeding_upper_bound_inds = np.flatnonzero(mod_eig_vals >= eig_upper_confidence_int)
    below_lower_bound_inds = np.flatnonzero(mod_eig_vals <= eig_lower_confidence_int)
    exceeding_eig_space = mod_eig_vecs[:,exceeding_upper_bound_inds]
    below_eig_space = mod_eig_vecs[:,below_lower_bound_inds]
    return below_eig_space, below_lower_bound_inds, [mean_mins_eig, min_confidence_ints], exceeding_eig_space, exceeding_upper_bound_inds, [mean_maxs_eig, max_confidence_ints]

def nodeRejection(modularity_matrix, eig_vals, confidence_level, eig_vecs, weight_type='linear', norm='L2', int_type='CI', bounds='upper'):
    """
    Separate the nodes into 'signal' and 'noise' based on the eigenvalues predicted by a null model.
    Arguments:  modularity_matrix, numpy.ndarray (num nodes, num noes), the modularity matrix of the network
                eig_vals, numpy.ndarry (num samples, num nodes), the null-model eigenspectrum distribution
                confidence_level, float, the specified confidence interval on the maximum eigenvalue, (enter 0 to just use the mean)
                eig_vecs, numpy.ndarry (num samples, num nodes, num nodes), the eigenvectors for each sample from the null model
                weight_type, string, options = ['linear', 'none', 'sqrt'] how the eigenvalues should be weighted when used for projection
                norm, string, options = ['L2', 'L1', 'Lmax'] the norm to use when calculating the lengths of low-dimensional models
                int_type, string, the type of confidence interval to use, either 'CI' or 'PI'
                bounds, string, options = ['upper', 'lower'] use the low-d space above the null model space (communities), or below the null model space (n-partite graphs)
    Returns:    reject_dict, dictionary,    'noise_inds', the indices of the nodes in the noise part of the network
                                            'neg_signal_inds', the indices of the nodes below the lower bound of the null network
                                            'CI_model', confidence interval model (could otherwise be 'PI_model')
                                            'm_model', the model lengths
                                            'neg_difference', the raw and normalised differences between the data and the model-derived lower bound for each node
                                            'difference', the raw and normalised differences between the data values and model-derived threshold values for each node
                                            'signal_inds', the indices of the nodes in the signal part of the network
    """
    def getBoundedSpace(bounds, modularity_matrix, eig_vals, confidence_level, int_type):
        """
        Get the eigenvectors and indices of those eigenvectors in the low-d space.
        Arguments:  bounds, string, options = ['upper', 'lower'] use the low-d space above the null model space (communities), or below the null model space (n-partite graphs)
                    modularity_matrix, numpy.ndarray (num nodes, num noes), the modularity matrix of the network
                    eig_vals, numpy.ndarry (num samples, num nodes), the null-model eigenspectrum distribution
                    confidence_level, float, the specified confidence interval on the maximum eigenvalue, (enter 0 to just use the mean)
                    int_type, string, the type of confidence interval to use, either 'CI' or 'PI'
        Returns:    lowd_eig_space, eigenvectors corresponding to eigenvalues exceed or below the null model eigenspectrum distribution
                    lowd_indices, the indices of those eigenvectors
        """
        upper_and_lower = getLowDimSpace(modularity_matrix, eig_vals, confidence_level, int_type=int_type)
        lowd_eig_space, lowd_indices = upper_and_lower[3:5] if bounds == 'upper' else upper_and_lower[0:2]
        if not(bounds in ['upper', 'lower']):
            sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unknown bound!')
        return lowd_eig_space, lowd_indices

    def getWeightedLowDSpaceProjections(weight_type, num_samples, lowd_eig_space, lowd_indices, eig_vals, eig_vecs, mod_eig_vals):
        """
        Get the weighted projection of the eigenvectors of the samples into the low-d space
        Arguments:  weight_type, string, options = ['linear', 'none', 'sqrt'] how the eigenvalues should be weighted when used for projection
                    num_samples, int, the number of samples
                    lowd_eig_space, eigenvectors corresponding to eigenvalues exceed or below the null model eigenspectrum distribution
                    lowd_indices, the indices of those eigenvectors
                    eig_vals, numpy.ndarry (num samples, num nodes), the null-model eigenspectrum distribution
                    eig_vecs, numpy.ndarry (num samples, num nodes, num nodes), the eigenvectors for each sample from the null model
                    mod_eig_vals, the eigenvalues of the modularity matrix of the network
        Returns:    weighted_lowd_space, weighted low dimensional eigenspace
                    weighted_model_projections, projections of the low-d eigenvectors of the samples
        """
        if weight_type == 'none':
            weighted_lowd_space = lowd_eig_space
            weighted_model_projections = eig_vecs[:,:,lowd_indices]
        elif weight_type == 'linear':
            weighted_lowd_space = mod_eig_vals[lowd_indices] * lowd_eig_space # Vweighted
            weighted_model_projections = np.array([eig_vals[i, lowd_indices] * eig_vecs[i,:,lowd_indices].T for i in range(num_samples)]) # VmodelW
        elif weight_type == 'sqrt':
            weighted_lowd_space = np.sqrt(mod_eig_vals[lowd_indices]) * lowd_eig_space
            weighted_model_projections = np.array([np.sqrt(eig_vals[i, lowd_indices]) * eig_vecs[i,:,lowd_indices].T for i in range(num_samples)])
        else:
            sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unknown weight type!')
        return weighted_lowd_space, weighted_model_projections

    def getLowDModelLengths(norm, weighted_lowd_space, weighted_model_projections):
        """
        Get the lengths of the low dimensional space, and the lengths of the model projections into that space
        Arguments:  norm, string, options = ['L2', 'L1', 'Lmax'] the norm to use when calculating the lengths of low-dimensional models
                    weighted_lowd_space, weighted low dimensional eigenspace
                    weighted_model_projections, projections of the low-d eigenvectors of the samples
        Returns:    lowd_lengths, the lengths of the low dimensional space
                    model_lengths, the lengths of the model projections into that space
        """
        if norm == 'L2':
            lowd_lengths = np.sqrt(np.power(weighted_lowd_space, 2).sum(axis=1))
            model_lengths = np.sqrt(np.power(weighted_model_projections,2).sum(axis=2)) # VmodelL
        elif norm == 'L1':
            lowd_lengths = np.abs(weighted_lowd_space).sum(axis=1)
            model_lengths = np.abs(weighted_model_projections).sum(axis=2)
        elif norm == 'Lmax':
            lowd_lengths = np.abs(weighted_lowd_space).max(axis=1)
            model_lengths = np.abs(weighted_model_projections).max(axis=2)
        else:
            sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unknown norm!')
        return lowd_lengths, model_lengths

    num_samples, num_nodes = eig_vals.shape
    # mod_eig_vals = getDescSortedEigSpec(modularity_matrix)[0]
    mod_eig_vals = np.linalg.eigh(modularity_matrix)[0]
    lowd_eig_space, lowd_indices = getBoundedSpace(bounds, modularity_matrix, eig_vals, confidence_level, int_type)
    weighted_lowd_space, weighted_model_projections = getWeightedLowDSpaceProjections(weight_type, num_samples, lowd_eig_space, lowd_indices, eig_vals, eig_vecs, mod_eig_vals)
    lowd_lengths, model_lengths = getLowDModelLengths(norm, weighted_lowd_space, weighted_model_projections)
    reject_dict = {}
    reject_dict['m_model'] = model_lengths.mean(axis=0)
    if int_type == 'CI':
        reject_dict['CI_model'] = getConfidenceIntervalFromStdErr(model_lengths.std(axis=0), num_samples, confidence_level)
        reject_dict['difference'] = {}; reject_dict['neg_difference'] = {}
        reject_dict['difference']['raw'] = lowd_lengths - (reject_dict['m_model'] + reject_dict['CI_model'])
        reject_dict['difference']['norm'] = reject_dict['difference']['raw'] / (reject_dict['m_model'] + reject_dict['CI_model'])
        reject_dict['neg_difference']['raw'] = lowd_lengths - (reject_dict['m_model'] - reject_dict['CI_model'])
        reject_dict['neg_difference']['norm'] = reject_dict['neg_difference']['raw'] / (reject_dict['m_model'] - reject_dict['CI_model'])
    elif int_type == 'PI':
        reject_dict['PI_model'] = np.zeros([num_nodes, 2])
        for i in range(num_nodes):
            prediction_intervals = getNonParaPredictionInterval(model_lengths[:,i]) if 1 == np.mod(model_lengths.shape[0], 2) else getNonParaPredictionInterval(model_lengths[0:-1,i])
            try:
                reject_dict['PI_model'][i,:] = prediction_intervals[prediction_intervals[:,0] == confidence_level, 1:3][0]
            except:
                sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Cannot find specified prediction interval!')
        reject_dict['difference'] = {}; reject_dict['neg_difference'] = {}
        reject_dict['difference']['raw'] = lowd_lengths - reject_dict['PI_model'][:,1]
        reject_dict['difference']['norm'] = reject_dict['difference']['raw'] / reject_dict['PI_model'][:,1]
        reject_dict['neg_difference']['raw'] = lowd_lengths - reject_dict['PI_model'][:,0]
        reject_dict['neg_difference']['raw'] = lowd_lengths / reject_dict['PI_model'][:,0]
    else:
        sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unknown interval type!')
    reject_dict['signal_inds'] = np.flatnonzero(reject_dict['difference']['raw'] > 0)
    reject_dict['noise_inds'] = np.flatnonzero(reject_dict['difference']['raw'] <= 0)
    reject_dict['neg_signal_inds'] = np.flatnonzero(reject_dict['neg_difference']['raw'] <= 0)
    return reject_dict
