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
        n_files = args.nfiles

    #path = args.path # path to where training is stored
    path = 'bestModels/newbestNN'
    # make plot folder if it doesn't already exist
    if not os.path.exists(os.path.join(path, 'plots')):
        cmd = 'mkdir -p {}'.format(os.path.join(path, 'plots'))
        subprocess.run(cmd, shell=True)
    # loads result of training to make plots 
    if path != '' and not args.use_cache and not args.mdn:
        regressor = tf.keras.models.load_model(os.path.join(path, args.model))
    elif not args.use_cache and not args.mdn:
        regressor = tf.keras.models.load_model(os.path.join('cache', args.model))
    elif not args.use_cache and args.mdn:
        from taunet.computation import tf_mdn_loss
        import tensorflow_probability as tfp
        regressor = tf.keras.models.load_model(os.path.join(path, args.model), custom_objects={'MixtureNormal': tfp.layers.MixtureNormal, 'tf_mdn_loss': tf_mdn_loss})
    else:
        regressor = ''
        #print("Using cached data")

    KINEMATICS = ['TauJetsAuxDyn.truthEtaVisDressed', 'TauJetsAuxDyn.etaCombined', 'TauJetsAuxDyn.etaFinalCalib', 
                    'TauJetsAuxDyn.truthPhiVisDressed', 'TauJetsAuxDyn.nTracks']

    d = testing_data(
        PATH, DATASET, FEATURES, list(set(TRUTH_FIELDS + OTHER_TES + KINEMATICS)), regressor, nfiles=n_files, 
        optional_path=path, select_1p=args.oneProng, select_3p=args.threeProngs, 
        no_normalize=args.no_normalize, no_norm_target=args.no_norm_target, 
        saveToCache=args.add_to_cache, useCache=args.use_cache, debug=args.debug)

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

    def makeBins(bmin, bmax, nbins):
        returnBins = []
        stepsize = (bmax - bmin) / nbins
        for i in range(nbins):
            returnBins.append((bmin + i*stepsize, bmin + (i+1)*stepsize))
        return returnBins


#%% ----------------------------------------------------
# make resolution plot with eta, mu, and p_T
    def run_resolution_plots():

        from taunet.utils import response_curve, response_curve_2vars
        plotSaveLoc = path

        nbins = 25
        pT_bins = makeBins(10, 200, nbins)
        eta_bins = makeBins(-2.5, 2.5, nbins)
        mu_bins = makeBins(0, 80, nbins)

        CL = args.CL #confidence internval

        # 1d response curve vals
        bins_reg, bin_errors_reg, means_reg, errs_reg, resol_reg = response_curve(response_reg, truth_pt, pT_bins, cl=CL)
        bins_rgEta, bin_err_rgEta, means_rgEta, errs_rgEta, resol_rgEta = response_curve(response_reg, truth_eta, eta_bins, cl=CL)
        bins_rgMu, bin_err_rgMu, means_rgMu, errs_rgMu, resol_rgMu = response_curve(response_reg, mu, mu_bins, cl=CL)
        bins_ref, bin_errors_ref, means_ref, errs_ref, resol_ref = response_curve(response_ref, truth_pt, pT_bins, cl=CL)
        bins_rfEta, bin_err_rfEta, mean_tfEta, errs_rfETa, resol_rfEta = response_curve(response_ref, truth_eta, eta_bins, cl=CL)
        bins_rfMu, bin_err_rfMu, mean_tfMu, errs_rfMu, resol_rfMu = response_curve(response_ref, mu, mu_bins, cl=CL)
        bins_comb, bin_errors_comb, means_comb, errs_comb, resol_comb = response_curve(response_comb, truth_pt, pT_bins, cl=CL)
        bins_cEta, bins_errors_cEta, means_cEta, errs_cEta, resol_cEta = response_curve(response_comb, truth_eta, eta_bins, cl=CL)
        bins_cMu, bins_errors_cMu, means_cMu, errs_cMu, resol_cMu = response_curve(response_comb, mu, mu_bins, cl=CL)

        # 2d response curve vals
        bins_pT, bins_eta, resol_pTEta = response_curve_2vars(response_reg, truth_pt, truth_eta, pT_bins, eta_bins, cl=CL)
        bins_pT, bins_mu, resol_pTMu = response_curve_2vars(response_reg, truth_pt, mu, pT_bins, mu_bins, cl=CL)

        def plot1dResolution(x, yref, ycomb, yreg, xtitle, savetitle=''):
            fig = plt.figure(figsize=(5,5), dpi = 100)
            plt.plot(x, 100 * yref, color='red', label='Final')
            plt.plot(x, 100 * ycomb, color='black', label='Combined')
            plt.plot(x, 100 * yreg, color='purple', label='This work')
            plt.ylabel('$p_{T} (\\tau_{had-vis})$ resolution, '+str(round(CL*100))+'% CL [%]', loc = 'top')
            plt.xlabel(xtitle, loc = 'right')
            plt.legend()
            if savetitle != '':
                plt.savefig(os.path.join(path, savetitle))

        plot1dResolution(bins_ref, resol_ref, resol_comb, resol_reg, 
            'True $p_{T}(\\tau_{had-vis})$ [GeV]', savetitle='explorationPlots/pT_resol_{}CL.pdf'.format(int(CL*100)))
        plot1dResolution(bins_rfEta, resol_rfEta, resol_cEta, resol_rgEta, 
            'True $\\eta (\\tau_{had-vis})$', savetitle='explorationPlots/eta_resol_{}CL.pdf'.format(int(CL*100)))
        plot1dResolution(bins_rfMu, resol_rfMu, resol_cMu, resol_rgMu, 
            'Sample $\\mu (\\tau_{had-vis})$', savetitle='explorationPlots/mu_resol_{}CL.pdf'.format(int(round(CL*100))))

        def plotHeatMapResolution(data, xbins, ybins, restitle, xtitle, ytitle, savetitle=''):
            fig, ax = plt.subplots()
            im = ax.imshow(data.T)
            ax.set_xticks(np.arange(len(xbins)), map(str, map(round,xbins)))
            ax.set_yticks(np.arange(len(ybins)), [str(round(ybins[i], 1)) for i in range(len(ybins))])
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            ax.invert_yaxis()
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel(restitle)
            plt.xlabel(xtitle, loc = 'right')
            plt.ylabel(ytitle, loc = 'top')
            if savetitle != '':
                plt.savefig(os.path.join(path, savetitle))

        plotHeatMapResolution(resol_pTEta, bins_pT, bins_eta, 
                                '$p_T (\\tau_{had-vis})$ resolution, '+str(round(CL * 100))+'% CL [%]', 
                                'True $p_T (\\tau_{had-vis})$ [GeV]', 
                                'True $\\eta (\\tau_{had-vis})$', 
                                savetitle='explorationPlots/pT_eta_reg_{}CL.pdf'.format(int(CL*100)))

        plotHeatMapResolution(resol_pTMu, bins_pT, bins_mu,
                                '$p_T (\\tau_{had-vis})$ resolution, '+str(round(CL * 100))+'% CL [%]',
                                'True $p_T (\\tau_{had-vis})$ [GeV]',
                                'Sample $\\mu (\\tau_{had-vis})$', 
                                savetitle='explorationPlots/pT_mu_reg_{}CL.pdf'.format(int(CL*100)))

        if not args.no_plot:
            plt.show()

        if args.copy_to_cernbox:
            local_copy_to_cernbox(location=os.path.join(path, 'explorationPlots'))
    #execute above code
    run_resolution_plots()

#%% -----------------------------------------------------------------------
    # explore which events are preferentially favored by our model over final
    def run_best_worst_events():
        def plotPerformance(reg, ref, comb, lims=(0,2), savetitle='', bins=300):
            plt.figure(figsize=(5,5), dpi = 100)
            plt.yscale('log')
            counts_reg, bins_reg, bars_reg = plt.hist(reg, bins=bins, range=lims, histtype='step', color='purple', label='This work')
            counts_ref, bins_ref, bars_ref = plt.hist(ref, bins=bins, range=lims, histtype='step', color='red', label='Final')
            counts_comb, bins_comb, bars_comb = plt.hist(comb, bins=bins, range=lims, histtype='step', color='black', label='Combined')
            plt.legend()
            if savetitle != '':
                plt.savefig(savetitle)
            return np.array(counts_reg), np.array(counts_ref), np.array(bins_reg)

        reg, ref, bins = plotPerformance(response_reg, response_ref, response_comb)

        # get the bins where perforamce is better
        better_bins = []
        for i in range(len(reg)):
            if reg[i] < ref[i]:
                better_bins.append((bins[i], bins[i+1]))

        # construct array of bools indexing which events are in the above bins
        better_events_bools = np.full(len(response_reg), False)
        for _bin in better_bins:
            temp = (response_reg > _bin[0]) & (response_reg < _bin[1])
            better_events_bools = better_events_bools | temp

        # select events that perform better and worse
        better_events = d[better_events_bools]
        worse_events = d[~better_events_bools]

        # histogram these two regimes 
        response_reg_better = better_events['regressed_target'] * better_events['TauJetsAuxDyn.ptCombined'] / better_events['TauJetsAuxDyn.truthPtVisDressed']
        response_ref_better = better_events['TauJetsAuxDyn.ptFinalCalib'] / better_events['TauJetsAuxDyn.truthPtVisDressed']
        response_comb_better = better_events['TauJetsAuxDyn.ptCombined'] / better_events['TauJetsAuxDyn.truthPtVisDressed']

        #plotPerformance(response_reg_better, response_ref_better, response_comb_better, bins=200)

        response_reg_worse = worse_events['regressed_target'] * worse_events['TauJetsAuxDyn.ptCombined'] / worse_events['TauJetsAuxDyn.truthPtVisDressed']
        response_ref_worse = worse_events['TauJetsAuxDyn.ptFinalCalib'] / worse_events['TauJetsAuxDyn.truthPtVisDressed']
        response_comb_worser = worse_events['TauJetsAuxDyn.ptCombined'] / worse_events['TauJetsAuxDyn.truthPtVisDressed']

        #plotPerformance(response_reg_worse, response_ref_worse, response_comb_worser, bins=200)

        # now, let's inspect some further properties of the variables 
        ## P_T
        plt.figure(figsize=(5,5), dpi = 100)
        plt.ticklabel_format(axis='y',style='sci',scilimits=(-3,3), useMathText=True)
        plt.hist(better_events['regressed_target'] * better_events['TauJetsAuxDyn.ptCombined'] / 1000., 
                histtype='step', bins=200, range=(0,200), color='blue', label='Better Events')
        plt.hist(worse_events['regressed_target'] * worse_events['TauJetsAuxDyn.ptCombined'] / 1000., 
                histtype='step', bins=200, range=(0,200), color='red', label='Worse Events')
        plt.hist(d['regressed_target'] * d['TauJetsAuxDyn.ptCombined'] / 1000., 
                histtype='step', bins=200, range=(0,200), color='black', label='All Events')
        plt.xlabel('$p_T (\\tau_{had-vis})$ [GeV]', loc='right')
        plt.ylabel('Number of $\\tau_{had-vis}$', loc = 'top')
        plt.legend()

        if not args.no_plot:
            plt.show()

    #run_best_worst_events()