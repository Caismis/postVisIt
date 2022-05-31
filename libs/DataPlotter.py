import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

def expscatter(expx, expv, label='Exp', marker='o'):
    if label:
        plt.scatter(expx, expv, c='Black', label=label, marker=marker)
    else:
        plt.scatter(expx, expv, c='Black', marker=marker)

def simplot(simx, simv, label=False, xtitle=False ,ytitle=False, color='Red', xlim='Auto', ylim='Auto', ylog=False, linestyle='-', filter_strenth=1):
    if label:
        plt.plot(simx, gaussian_filter1d(simv, sigma=filter_strenth), c=color, label=label, linestyle=linestyle)
    else:
        plt.plot(simx, gaussian_filter1d(simv, sigma=filter_strenth), c=color, linestyle=linestyle)
    if xlim == 'Base':
        plt.xlim(min(simx), max(simx))
    elif xlim != 'Auto':
        plt.xlim(xlim)
    if ylim != 'Auto':
        plt.ylim(ylim)
    if ylog:
        plt.yscale('log')
    if ytitle:
        plt.ylabel(ytitle)
    if xtitle:
        plt.xlabel(xtitle)