import numpy as np
from scipy.interpolate import interp1d

def relative_error(expdata, simdata):
    explen = len(expdata[0])
    simfunc = interp1d(simdata[0], simdata[1])
    simmin = min(simdata[0])
    simmax = max(simdata[0])
    err_sum = 0
    err_max = 0
    err_min = 100
    counter = 0
    for i in range(explen):
        if expdata[0][i] <= simmax and expdata[0][i] >= simmin:
            error = abs(expdata[1][i] - simfunc(expdata[0][i]))/expdata[1][i]
            if error > err_max:
                err_max = error
            if error < err_min:
                err_min = error
            err_sum += error
            counter += 1
    err_sum /= counter
    return (err_sum, err_min, err_max)

def error_std(expdata, simdata):
    explen = len(expdata[0])
    simfunc = interp1d(simdata[0], simdata[1])
    simmin = min(simdata[0])
    simmax = max(simdata[0])
    errs = []
    for i in range(explen):
        if expdata[0][i] <= simmax and expdata[0][i] >= simmin:
            error = abs(expdata[1][i] - simfunc(expdata[0][i]))/expdata[1][i]
            errs.append(error)
    std = np.std(np.array(errs), ddof=1)
    return std