import numpy as np
import os
import subprocess

from . import log; log = log.getChild(__name__)

def get_quantile_width(arr, cl=0.68):
    """
    """
    q1 = (1. - cl) / 2.
    q2 = 1. - q1
    y = np.quantile(arr, [q1, q2])
    width = (y[1] - y[0]) / 2.
    return width


def response_curve(res, var, bins, cl=0.68):
    """
    """
    _bin_centers = []
    _bin_errors = []
    _means = []
    _mean_stat_err = []
    _resol = []
    for _bin in bins:
        a = res[(var > _bin[0]) & (var < _bin[1])]
        # mean = 
        # mean, std = norm.fit(a)
        # y = np.quantile(a, [q1, q2])
        # res_68 = (y[1] - y[0]) / 2.
        # print (_bin, len(a), a.mean(), mean, a.std(), std, (y[1] - y[0]) / 2., np.quantile(a-1, 0.95))
        # make some cuts here to take care of outlierss
        _means += [np.mean(a)]
        _mean_stat_err += [np.std(a, ddof=1) / np.sqrt(np.size(a))]
        _resol += [get_quantile_width(a, cl=cl)]
        _bin_centers += [_bin[0] + (_bin[1] - _bin[0]) / 2]
        _bin_errors += [(_bin[1] - _bin[0]) / 2]
    return _bin_centers, np.array(_bin_errors), np.array(_means), np.array(_mean_stat_err), np.array(_resol)

def response_curve_2vars(res, var1, var2, bins1, bins2, cl=0.68):
    """
    """
    _bin_centers_var1 = []
    _bin_centers_var2 = []
    _resol = np.empty((len(bins1), len(bins2)))
    for _bin1 in bins1:
        _bin_centers_var1 += [_bin1[0] + (_bin1[1] - _bin1[0]) / 2]
        for _bin2 in bins2:
            a = res[(var2 > _bin2[0]) & (var2 < _bin2[1]) & (var1 > _bin1[0]) & (var1 < _bin1[1])]
            if _bin1 == bins1[0]:
                _bin_centers_var2 += [_bin2[0] + (_bin2[1] - _bin2[0]) / 2]
            _resol[bins1.index(_bin1)][bins2.index(_bin2)] = get_quantile_width(a, cl=cl)
    return np.array(_bin_centers_var1), np.array(_bin_centers_var2), np.array(_resol)

def copy_plots_to_cernbox(fmt='pdf', location='taunet_plots'):
    """
    """
    _cernbox = os.path.join(
        '/eos/user/',
        os.getenv('USER')[0],
        os.getenv('USER'),
        location)
    if not os.path.exists(_cernbox):
        cmd = 'mkdir -p {}'.format(_cernbox)
        log.info(cmd)
        subprocess.run(cmd, shell=True)

    #! kinda a sketch way but should work...
    if location != 'taunet_plots':
        os.listdir(os.path.join(location, 'plots'))
        for _fig in os.listdir(os.path.join(location, 'plots')):
            if _fig.endswith(fmt):
                cmd = 'cp {} {}'.format(
                    os.path.join(location, 'plots', _fig),
                    _cernbox)
                log.info(cmd)
                subprocess.run(cmd, shell=True)
    else:
        for _fig in os.listdir('./plots/'):
            if _fig.endswith(fmt):
                cmd = 'cp {} {}'.format(
                    os.path.join('./plots', _fig),
                    _cernbox)
                log.info(cmd)
                subprocess.run(cmd, shell=True)