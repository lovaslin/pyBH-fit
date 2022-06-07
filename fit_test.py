###########################################################################################
## Test the behavior of the fit and the bias for different width of the scan window.     ##
## Fit approximation is done with 50_000 PEs and the bias comupted using 5_000_000 PEs.  ##
## The scans are parllelized across 6x10 cores (for width and batches).                  ##
##                                                                                       ##
## Command line arguments                                                                ##
##     Nb : Number of bins in the histograms (width varies between 1 and Nb-1)           ##
##     loc_path : Path to the local result directory                                     ##
##     tem_path : Path to the temprorary results directory (used for PE batches results) ##
##     width : Width of the scan window to be used (integer)                             ##
##     bkg : The background shape to use, can be 'lin' or 'exp'                          ##
##     Nbkg : The number of background event to be generated in the data (integer)       ##
###########################################################################################


import numpy as np
import matplotlib.pyplot as plt
import pyBumpHunter as BH

from scipy.special import erfc, erfcinv
from scipy import optimize as So
from scipy.integrate import quad
from scipy.stats import norm

from concurrent.futures import ProcessPoolExecutor as PPE

import sys
import os
import getopt
import h5py


## Check command line arguments

print('INITIALIZATION')

# Get argument list
print('Starting')
arg_list = ['Nb=', 'loc_path=', 'tem_path=', 'width=', 'bkg=', 'Nbkg=']
argv = sys.argv[1:]
opts, args = getopt.getopt(argv,"",arg_list)

# Initialize argument values (default)
loc_path = 'results/'
tem_path = '/AtlasDisk/user/vaslin/'
Nb = 20
width = 1
bkg_type = 'exp'
Nbkg = 100_000


# Check argument and update values
for opt, val in opts:
    if opt == '--Nb':
        Nb = int(val)
    elif opt == '--loc_path':
        loc_path = val
    elif opt == '--tem_path':
        tem_path = val
    elif opt == '--width':
        width = int(val)
    elif opt == '--bkg':
        if val in ['exp', 'uni']:
            bkg_type = val
    elif opt ==  '--Nbkg':
        Nbkg = int(val)


## Create dataset

print(f'GENERATING HISTOGRAMS (Nbin={Nb})')

# Background
np.random.seed(42)
if bkg_type == 'exp':
    bkg = np.random.exponential(3.5, 10*Nbkg)
else:
    bkg = np.random.uniform(0, 35, 10*Nbkg)

# Data
data = np.empty(Nbkg)
if bkg_type == 'exp':
    data = np.random.exponential(3.5, Nbkg)
else:
    data = np.random.uniform(0, 35, Nbkg)

# Make histograms with Nb bins
rng = [0, 35]
bkg_hist, bins = np.histogram(bkg, bins=Nb, range=rng)
bkg_hist = bkg_hist / 10
data_hist, _ = np.histogram(data, bins=bins, range=rng)


## Functions

# Function to fit the test statistic distribution
def BHstat(x, pM, m, A):
    '''
    x : min p-value
    pM : median of p
    m : Number of test (window)
    A : Global scale
    '''
    xe = np.exp(-x)
    res = erfcinv(2*pM) * (2*erfcinv(2*xe) - erfcinv(2*pM))
    res = m * np.exp(res)
    res2 = (1 - 0.5*erfc(erfcinv(2*xe) - erfcinv(2*pM)))**(m-1)
    return A * (res * res2) * xe

# Function to compute a chi2 test with errors
def chi2_test(obs, err, fit, Nparam):
    '''
    Compute a ch2 test to evaluate goodness of fit.
    It takes the error on the observation into account (unlike scipy)
    
    Arguments
        obs : The observed values (numpy array)
        err : The error on the observation (numpy array)
        fit : The coresponding predicted values given by the fit function ((numpy array))
    '''
    res = (obs - fit) / err
    res = np.sum(res**2)
    return res / (obs.size - Nparam)

# Function to scan a batch of PEs and save results in a temp file
def scan_batch(Hdata, Hbkg, bins, N, w, sdir, ith):
    '''
    Function that runs a BH scan given data and reference histograms with N pseud-experiment.
    The resulting min p-value and BHstat arrays are writen in a h5 file.
    
    Arguments
        Hdata : The data histogram bin yields (numpy array)
        Hbkg : The reference background histogram bin yields (numpy array)
        bins : The histograms bin edges (numpy array)
        N : The number of pseudo-experiment to generate (int)
        w : The width of the window
        sdir : Path to the temporary directory
        ith : Unique ID of the process (used to identify temp files)
    '''
    
    # Create a BH1D instance FIXME
    bhi = BH.BumpHunter1D(
        rang = [bins[0], bins[-1]],
        bins=bins,
        width_min=w,
        width_max=w,
        width_step=1,
        scan_step=1,
        npe=N,
        nworker=1,
        seed=100 + (2 * ith)
    )
    
    # Run the scan
    bhi.bump_scan(Hdata, Hbkg, is_hist=True)
    
    # Get results in an array
    # If ith>0, do not keep the value of the data
    if ith==0:
        res = np.empty((N+1, 2))
        res[:,0] = bhi.min_Pval_ar
        res[:,1] = bhi.t_ar
    else:
        res = np.empty((N, 2))
        res[:,0] = bhi.min_Pval_ar[1:]
        res[:,1] = bhi.t_ar[1:]
    
    # Write results in a h5 file
    fname = sdir+f'temp{ith}.h5'
    if os.path.exists(fname):
        os.remove(fname)
    with h5py.File(fname, mode='a') as f:
        f.create_dataset('data', data=res)
    
    # Free some memmory
    del res
    del bhi
    
    return

# Function to test the fit with windows of width w
def fit_w(data, bkg, bins, w, lpath, tpath):
    '''
    Run the BH scans with a window fixed window width.
    The scanss are run in parallels for 50k and 5M PEs.
    A fit is performed with the 50k PEs run,
    
    Arguments
        data : The data bin yields (numpy array)
        bkg : The reference background bin yields (numpy array)
        bins : The histograms bin edges (numpy array)
        w : The window width to use (int)
        lpath : Path to the local result directory (str)
        tpath : Path to the temp result directory for PE batches (str)
    '''
    
    print(f'\tSTARTING w={w}')
    
    # Define a local path for width w
    lpath = lpath+f'width{w}/'
    if not os.path.exists(lpath):
        os.mkdir(lpath, 0o755)
    
    # Initialize BH instance with 50_000 PEs FIXME
    bh = BH.BumpHunter1D(
        bins=bins,
        rang=[bins[0], bins[-1]],
        width_min=w,
        width_max=w,
        width_step=1,
        scan_step=1,
        npe=50_000,
        nworker=1,
        #use_sideband=True,
        seed=666
    )
    
    # Run the scan
    bh.bump_scan(data, bkg, is_hist=True)
    
    # Plot scan results results
    bh.plot_tomography(bkg, is_hist=True, filename=lpath+'tomography.pdf')
    bh.plot_bump(data, bkg, is_hist=True, filename=lpath+'bump.pdf')
    bh.plot_stat(show_Pval=True, filename=lpath+'BHstat.pdf')
    
    # Get the BHstat histogram
    Hbh, bbh = np.histogram(bh.t_ar[1:][bh.t_ar[1:]>1e-3], bins=50)
    x = (bbh[:-1] + bbh[1:]) / 2
    
    # Compute stat error on y (Poisson)
    erry = np.sqrt(Hbh)
    erry[Hbh==0] = 1.5
    
    # Do the fit of BHstat distribution with scipy
    param, cov = So.curve_fit(
        BHstat,
        x,
        Hbh,
        p0=[0.5, data.size, 42],
        sigma=erry,
        absolute_sigma=True
    )
    
    # Compute chi2 goodness of fit test
    chi2_s = chi2_test(Hbh, erry, BHstat(x, *param), len(param))
    
    # Do the fit plot in lin scale
    xcont = np.linspace(bbh[0], bbh[-1], 499)
    F = plt.figure(figsize=(12,8))
    plt.hist(bbh[:-1], bins=bbh, weights=Hbh, histtype='step', lw=2, label='pseudo-data')
    plt.plot(xcont, BHstat(xcont, *param), lw=2, label=f'fit (chi2/ndf={chi2_s:.3g})')
    plt.legend(fontsize=24)
    plt.xlabel('Test statistic', size=24)
    plt.ylabel('PE count / bin', size=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.savefig(lpath+'fit_lin.pdf', bbox_inches='tight')
    plt.close(F)
    
    # Do the fit plot in log scale
    F = plt.figure(figsize=(12,8))
    plt.hist(bbh[:-1], bins=bbh, weights=Hbh, histtype='step', lw=2, label='pseudo-data')
    plt.plot(xcont, BHstat(xcont, *param), lw=2, label=f'fit (chi2/ndf={chi2_s:.3g})')
    plt.legend(fontsize=24)
    plt.xlabel('Test statistic', size=24)
    plt.ylabel('PE count / bin', size=24)
    plt.yscale('log')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.savefig(lpath+'fit_log.pdf', bbox_inches='tight')
    plt.close(F)
    
    # Save a few things
    t_ar = bh.t_ar[1:]
    
    # Delete some stuff to free a bit of memorry
    del bh
    
    # Define the number of batches to scan for the big test
    Ntot = 5_000_000
    batch_size = 100_000
    Nbatch = Ntot // batch_size
    
    # Check if the tempory file ofr the 5M PEs exists
    tpath = tpath+f'width{w}/'
    if not os.path.exists(tpath):
        # If not create it
        os.mkdir(tpath, 0o755)
        
        # Start a process pool with 10 cores
        with PPE(max_workers=10) as exe:
            # Run the batches in parallel on 10 cores
            for th in range(Nbatch):
                exe.submit(scan_batch, data, bkg, bins, batch_size, w, tpath, th)
    
    # Read and merge results from the temp files
    big_ar = np.empty(((Nbatch * batch_size) + 1, 2))
    at = 0
    for b in range(Nbatch):
        # Read file b into array
        fname = tpath+f'temp{b}.h5'
        with h5py.File(fname, mode='r') as f:
            # Check if we are reading file 0
            if b == 0:
                f.get('data').read_direct(big_ar[at:at+batch_size+1, :])
            else:
                f.get('data').read_direct(big_ar[at:at+batch_size, :])
        
        # Increment cursor position
        if b == 0:
            at += batch_size + 1
        else:
            at += batch_size
    
    # Make the big BHstat histogram and compute global p-value
    Hbh_big, bbh_big = np.histogram(big_ar[1:,1][big_ar[1:,1]>1e-3], bins=50)
    tdata = big_ar[0,1]
    gpval_big = big_ar[1:,1][big_ar[1:,1] > tdata].size
    gpval_big = gpval_big / big_ar[1:,1].size
    
    # Compute the scale factor to acount for the difference in stat (scipy fit)
    scale = (Hbh_big * (bbh_big[1:] - bbh_big[:-1])).sum()
    scale = scale / quad(BHstat, bbh_big[0], bbh_big[-1], args=tuple(param))[0]
    param[2] = param[2] * scale
    
    # Plot the rescaled fit function in lin scale
    F = plt.figure(figsize=(12,8))
    plt.hist(bbh_big[:-1], bins=bbh_big, weights=Hbh_big, histtype='step', lw=2, label='pseudo-data (5M)')
    plt.plot(xcont, BHstat(xcont, *param), lw=2, label='fit (scaled)')
    plt.legend(fontsize=24)
    plt.xlabel('Test statistic', size=24)
    plt.ylabel('Number of pseudo-experiments', size=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.savefig(lpath+'big_lin.pdf', bbox_inches='tight')
    plt.close(F)
    
    # Plot the rescaled fit function in log scale
    F = plt.figure(figsize=(12,8))
    plt.hist(bbh_big[:-1], bins=bbh_big, weights=Hbh_big, histtype='step', lw=2, label='pseudo-data (5M)')
    plt.plot(xcont, BHstat(xcont, *param), lw=2, label='fit (scaled)')
    plt.yscale('log')
    plt.legend(fontsize=24)
    plt.xlabel('Test statistic', size=24)
    plt.ylabel('Number of pseudo-experiments', size=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.savefig(lpath+'big_log.pdf', bbox_inches='tight')
    plt.close(F)
    
    # Plot both 50k and 5M hist on the same figure
    F = plt.figure(figsize=(12*2,8))
    plt.subplot(1,2,1)
    plt.hist(bbh_big[:-1], bins=bbh_big, weights=Hbh_big, density=True, histtype='step', lw=2, label='pseudo-data (5M)')
    plt.hist(t_ar, bins=bbh_big, density=True, histtype='step', lw=2, label='pseudo-data (50k)')
    plt.legend(fontsize=24)
    plt.xlabel('Test statistic', size=24)
    plt.ylabel('Number of pseudo-experiments (normalized)', size=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    
    plt.subplot(1,2,2)
    plt.hist(bbh_big[:-1], bins=bbh_big, weights=Hbh_big, density=True, histtype='step', lw=2, label='pseudo-data (5M)')
    plt.hist(t_ar, bins=bbh_big, density=True, histtype='step', lw=2, label='pseudo-data (50k)')
    plt.yscale('log')
    plt.legend(fontsize=24)
    plt.xlabel('Test statistic', size=24)
    plt.ylabel('Number of pseudo-experiments (normalized)', size=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.savefig(lpath+'BHstat_all.pdf', bbox_inches='tight')
    plt.close(F)
    
    # Compute global p-vale with fit function normalized to 1 (A = 1)
    gpval_fit, _ = quad(BHstat, 1e-10, tdata, args=(param[0], param[1], 1))
    gpval_fit = 1 - gpval_fit
    
    # Compute the global p-value bias for various tdata
    tdata = np.arange(4, 16, 1)
    bias = np.empty((tdata.size, 4))
    Ntot = big_ar[1:,1].size
    for i, t in enumerate(tdata):
        # Global p-value from BH (direct method)
        bias[i,0] = big_ar[1:,1][big_ar[1:,1] > t].size
        bias[i,0] = bias[i,0] / Ntot
        
        # Global p-value from scipy (fit method)
        bias[i,1] = 1 - quad(BHstat, 1e-10, t, args=(param[0], param[1], 1))[0]
    
    # Global significance
    bias[:,2] = norm.ppf(1-bias[:,0])
    bias[:,3] = norm.ppf(1-bias[:,1])
    
    # Do the plot bias VS tdata for width w
    F = plt.figure(figsize=(12,8))
    plt.plot(tdata, bias[:,0] / bias[:,1], 'o-', lw=2)
    plt.xlabel('tdata', size=24)
    plt.ylabel('Global p-value bias', size=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.savefig(lpath+'pval_bias.pdf', bbox_inches='tight')
    plt.close(F)
    
    F = plt.figure(figsize=(12,8))
    plt.plot(tdata, bias[:,2] / bias[:,3], 'o-', lw=2)
    plt.xlabel('tdata', size=24)
    plt.ylabel('Global significance bias', size=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.savefig(lpath+'sigma_bias.pdf', bbox_inches='tight')
    plt.close(F)
    
    # Plot of all glob p-val and sig
    F = plt.figure(figsize=(12,8))
    plt.plot(tdata, bias[:,0], 'x:', lw=2, label='direct')
    plt.plot(tdata, bias[:,1], 'o-', lw=2, label='fit')
    plt.yscale('log')
    plt.legend(fontsize=24)
    plt.xlabel('tdata', size=24)
    plt.ylabel('Global p-value', size=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.savefig(lpath+'pval.pdf', bbox_inches='tight')
    plt.close(F)
    
    F = plt.figure(figsize=(12,8))
    plt.plot(tdata, bias[:,2], 'x:', lw=2, label='direct')
    plt.plot(tdata, bias[:,3], 'o-', lw=2, label='fit')
    plt.legend(fontsize=24)
    plt.xlabel('tdata', size=24)
    plt.ylabel('Global significance', size=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.savefig(lpath+'sigma.pdf', bbox_inches='tight')
    plt.close(F)
    
    # Make parameters array
    errs = np.sqrt(np.diag(cov))
    res = {
        'pM': [param[0], errs[0]],
        'A': [param[1], errs[1]],
    }
    
    # Write fit parameters in a text file
    if os.path.exists(lpath+'param.txt'):
        os.remove(lpath+'param.txt')
    with open(lpath+'param.txt', 'w') as f:
        for p, v in res.items():
            print(f'{p} = {v[0]:.3g} +- {v[1]:.4g}', file=f)
    
    # Make global results array
    res_glob = np.empty((tdata.size, 5))
    res_glob[:,0] = tdata
    res_glob[:,1:] = bias
    
    # Write all global results in a h5 file
    if os.path.exists(lpath+'global.h5'):
        os.remove(lpath+'global.h5')
    with h5py.File(lpath+'global.h5', 'a') as f:
        f.create_dataset('data', data=res_glob)
    
    return


## Run the all the scans

print('RUNNING SCANS')

# Create a local result directory for the current number of bins
if not os.path.exists(loc_path+f'{bkg_type}{Nb}_{Nbkg}/'):
    os.mkdir(loc_path+f'{bkg_type}{Nb}_{Nbkg}/', 0o755)

# Create a temp result directory for the required background shape and number of bins (if needed)
if not os.path.exists(tem_path+f'{bkg_type}{Nb}_{Nbkg}/'):
    os.mkdir(tem_path+f'{bkg_type}{Nb}_{Nbkg}/', 0o755)

fit_w(data_hist, bkg_hist, bins, width, loc_path+f'{bkg_type}{Nb}_{Nbkg}/', tem_path+f'{bkg_type}{Nb}_{Nbkg}/')

print('DONE !!!')



