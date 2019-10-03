import numpy as np
from scipy import stats


def clr(data):
    """Centred log ratio transformation"""

    log_data = np.log(data)
    clr_data = log_data.subtract(np.mean(log_data, axis=1), axis=0)

    return clr_data


def alr(data):
    """Additive log ratio transformation"""

    alr_data = np.log(data.divide(data.iloc[:, 0], axis=0))

    return alr_data


def geometrics(data):
    """Calculate geometric mean and geometric standard deviation

    Parameters
    ----------
    data : array
        Values to average

    Returns
    -------
    geo_mean : float
        Geometric mean
    geo_std : float
        Geometric standard deviation
    """

    geo_mean = stats.mstats.gmean(data)

    geo_std = np.exp(np.sqrt(np.sum((np.log(data/geo_mean))**2)/len(data)))

    return geo_mean, geo_std
