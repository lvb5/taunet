import subprocess
from genericpath import exists
import os
from urllib import response
import numpy as np
from matplotlib import pyplot as plt

from taunet.database import PATH, DATASET, testing_data
from taunet.fields import FEATURES, TRUTH_FIELDS, OTHER_TES
if __name__ == '__main__':

    from taunet.parser import plot_parser
    # train_parser = argparse.ArgumentParser(parents=[common_parser])
    # train_parser.add_argument('--use-cache', action='store_true')
    args = plot_parser.parse_args()

    if not args.use_cache:
        import tensorflow as tf

    if args.debug:
        n_files = 7
    else:
        n_files = -1

    #path = args.path # path to where training is stored
    path = 'bestModels/bestNN'
    # make plot folder if it doesn't already exist
    if not os.path.exists(os.path.join(path, 'plots')):
        cmd = 'mkdir -p {}'.format(os.path.join(path, 'plots'))
        subprocess.run(cmd, shell=True)
    # loads result of training to make plots 
    if path != '' and not args.use_cache:
        regressor = tf.keras.models.load_model(os.path.join(path, args.model))
    elif not args.use_cache:
        regressor = tf.keras.models.load_model(os.path.join('cache', args.model))
    else:
        regressor = ''
        #print("Using cached data")

    d = testing_data(
        PATH, DATASET, FEATURES, TRUTH_FIELDS + OTHER_TES, regressor, nfiles=n_files, 
        optional_path=path, select_1p=args.oneProng, select_3p=args.threeProngs, 
        no_normalize=args.no_normalize, no_norm_target=args.no_norm_target, 
        saveToCache=args.add_to_cache, useCache=args.use_cache)

#%% ----------------------------------------------------
    #local save to cernbox
    def local_copy_to_cernbox(fmt='pdf', location='explorationPlots'):
        _cernbox = os.path.join('/Users/miles_cb/cernbox', location)
        if not os.path.exists(_cernbox):
            cmd = 'mkdir -p {}'.format(_cernbox)
            subprocess.run(cmd, shell=True)
        for _fig in os.listdir(location):
            if _fig.endswith(fmt):
                cmd = 'cp {} {}'.format(os.path.join(location, _fig), _cernbox)
                subprocess.run(cmd, shell=True) 

#%% define some variables to be used below
    response_reg = d['regressed_target'] * d['TauJetsAuxDyn.ptCombined'] / d['TauJetsAuxDyn.truthPtVisDressed']
    response_ref = d['TauJetsAuxDyn.ptFinalCalib'] / d['TauJetsAuxDyn.truthPtVisDressed']
    response_comb = d['TauJetsAuxDyn.ptCombined'] / d['TauJetsAuxDyn.truthPtVisDressed']
    truth_pt = d['TauJetsAuxDyn.truthPtVisDressed'] / 1000. 
    truth_eta = d['TauJetsAuxDyn.truthEtaVisDressed']
    mu = d['TauJetsAuxDyn.mu']

#%% ----------------------------------------------------
# make resolution plot with eta, mu, and p_T
    def run_resolution_plots():

        from taunet.utils import response_curve
        plotSaveLoc = path

        pT_bins = [
            # (0, 10),
            (10, 20),
            (20, 30),
            (30, 40),
            (40, 50),
            (50, 60),
            (60, 70),
            (70, 80),
            (80, 90),
            (90, 100),
            (100, 150),
            #(150, 200),
        ]

        eta_bins = [
            (-2.5, -2.0),
            (-2.0, -1.5),
            (-1.5, -1.0),
            (-1.0, -0.5),
            (-0.5, 0.0),
            (0.0, 0.5),
            (0.5, 1.0),
            (1.0, 1.5),
            (1.5, 2.0),
            (2.0, 2.5)
        ]

        mu_bins = [
            (0, 10),
            (10, 20),
            (20, 30),
            (30, 40),
            (40, 50), 
            (50, 60),
            (60, 65),
            (65, 70),
            (70, 75),
            (75, 80)
        ]

        CI = args.CI #confidence internval

        bins_reg, bin_errors_reg, means_reg, errs_reg, resol_reg = response_curve(response_reg, truth_pt, pT_bins, ci=CI)
        bins_rgEta, bin_err_rgEta, means_rgEta, errs_rgEta, resol_rgEta = response_curve(response_reg, truth_eta, eta_bins, ci=CI)
        bins_rgMu, bin_err_rgMu, means_rgMu, errs_rgMu, resol_rgMu = response_curve(response_reg, mu, mu_bins, ci=CI)
        bins_ref, bin_errors_ref, means_ref, errs_ref, resol_ref = response_curve(response_ref, truth_pt, pT_bins, ci=CI)
        bins_rfEta, bin_err_rfEta, mean_tfEta, errs_rfETa, resol_rfEta = response_curve(response_ref, truth_eta, eta_bins, ci=CI)
        bins_rfMu, bin_err_rfMu, mean_tfMu, errs_rfMu, resol_rfMu = response_curve(response_ref, mu, mu_bins, ci=CI)
        bins_comb, bin_errors_comb, means_comb, errs_comb, resol_comb = response_curve(response_comb, truth_pt, pT_bins, ci=CI)
        bins_cEta, bins_errors_cEta, means_cEta, errs_cEta, resol_cEta = response_curve(response_comb, truth_eta, eta_bins, ci=CI)
        bins_cMu, bins_errors_cMu, means_cMu, errs_cMu, resol_cMu = response_curve(response_comb, mu, mu_bins, ci=CI)

        def plot1dResolution(x, yref, ycomb, yreg, xtitle, savetitle=''):
            fig = plt.figure(figsize=(5,5), dpi = 100)
            plt.plot(x, 100 * yref, color='red', label='Final')
            plt.plot(x, 100 * ycomb, color='black', label='Combined')
            plt.plot(x, 100 * yreg, color='purple', label='This work')
            plt.ylabel('$p_{T} (\\tau_{had-vis})$ resolution, '+str(round(CI*100))+'% C.I. [%]', loc = 'top')
            plt.xlabel(xtitle, loc = 'right')
            plt.legend()
            if savetitle != '':
                plt.savefig(savetitle)

        plot1dResolution(bins_ref, resol_ref, resol_comb, resol_reg, 
            'True $p_{T}(\\tau_{had-vis})$ [GeV]', savetitle='explorationPlots/pT_resol_{}CI.pdf'.format(int(CI*100)))
        plot1dResolution(bins_rfEta, resol_rfEta, resol_cEta, resol_rgEta, 
            'True $\\eta (\\tau_{had-vis})$', savetitle='explorationPlots/eta_resol_{}CI.pdf'.format(int(CI*100)))
        plot1dResolution(bins_rfMu, resol_rfMu, resol_cMu, resol_rgMu, 
            'Sample $\\mu (\\tau_{had-vis}$', savetitle='explorationPlots/mu_resol_{}CI.pdf'.format(int(round(CI*100))))

        def plotHeatMapResolution(x, xbins, y, ybins, restitle, xtitle, ytitle, savetitle=''):
            data = np.empty((len(x), len(y)))
            for i in range(len(x)):
                for j in range(len(y)):
                    #! Should this be multiplication instead??? ASK
                    data[i][j] = (x[i] + y[j]) * 100
            fig, ax = plt.subplots()
            im = ax.imshow(data)
            ax.set_xticks(np.arange(len(xbins)), map(str, xbins))
            ax.set_yticks(np.arange(len(ybins)), map(str, ybins))
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel(restitle)
            plt.xlabel(xtitle, loc = 'right')
            plt.ylabel(ytitle, loc = 'top')
            if savetitle != '':
                plt.savefig(savetitle)

        plotHeatMapResolution(resol_reg, bins_reg, resol_rgEta, bins_rgEta, 
                                '$p_T (\\tau_{had-vis})$ resolution, '+str(round(CI * 100))+'% CI [%]', 
                                'True $p_T (\\tau_{had-vis})$ [GeV]', 
                                'True $\\eta (\\tau_{had-vis})$', 
                                savetitle='explorationPlots/pT_eta_reg_{}CI.pdf'.format(int(CI*100)))

        plotHeatMapResolution(resol_reg, bins_reg, resol_rgMu, bins_rgMu, 
                                '$p_T (\\tau_{had-vis})$ resolution, '+str(round(CI * 100))+'% CI [%]',
                                'True $p_T (\\tau_{had-vis})$ [GeV]',
                                'Sample $\\mu (\\tau_{had-vis})$')

        if not args.no_plot:
            plt.show()

        if args.copy_to_cernbox:
            local_copy_to_cernbox(location='explorationPlots')
    #execute above code
    #run_resolution_plots()

#%% -----------------------------------------------------------------------
    # explore which events are preferentially favored by our model over final
    plt.figure(figsize=(5,5), dpi = 100)
    plt.yscale('log')
    lims = (0,3)
    counts_reg, bins_reg, bars_reg = plt.hist(response_reg, bins=200, range=lims,
                                histtype='step', color='purple', label='This work')
    counts_ref, bins_ref, bars_ref = plt.hist(response_ref, bins=200, range=lims, 
                                histtype='step', color='red', label='Final')
    counts_comb, bins_comb, bars_comb = plt.hist(response_comb, bins=200, range=lims,
                                histtype='step', color='black', label='Combined')
    plt.legend()
    plt.show()