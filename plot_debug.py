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
        #normIndices=list(map(int, args.normIDs)),
        saveToCache=args.add_to_cache, useCache=args.use_cache)

#%% ----------------------------------------------------
# make resolution plot with eta and p_T

    from taunet.utils import response_curve
    plotSaveLoc = path

    pT_response_reg = d['regressed_target'] * d['TauJetsAuxDyn.ptCombined'] / d['TauJetsAuxDyn.truthPtVisDressed']
    eta_reponse_reg = d['regressed_target'] * d['TauJetsAuxDyn.ptCombined'] / d['TauJetsAuxDyn.truthEtaVisDressed']
    pT_response_ref = d['TauJetsAuxDyn.ptFinalCalib'] / d['TauJetsAuxDyn.truthPtVisDressed']
    eta_reponse_ref = d['TauJetsAuxDyn.etaFinalCalib'] / d['TauJetsAuxDyn.truthEtaVisDressed']
    pT_response_comb = d['TauJetsAuxDyn.ptCombined'] / d['TauJetsAuxDyn.truthPtVisDressed']
    eta_response_comb = d['TauJetsAuxDyn.etaCombined'] / d['TauJetsAuxDyn.truthEtaVisDressed']
    truth_pt = d['TauJetsAuxDyn.truthPtVisDressed'] / 1000. # dived my 1000 to get some scale as above
    truth_eta = d['TauJetsAuxDyn.truthEtaVisDressed']

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
        (150, 200),
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

    bins_reg, bin_errors_reg, means_reg, errs_reg, resol_reg = response_curve(pT_response_reg, truth_pt, pT_bins)
    bins_rgEta, bin_err_rgEta, means_rgEta, errs_rgEta, resol_rgEta = response_curve(eta_reponse_reg, truth_eta, eta_bins)
    bins_ref, bin_errors_ref, means_ref, errs_ref, resol_ref = response_curve(pT_response_ref, truth_pt, pT_bins)
    bins_rfEta, bin_err_rfEta, mean_tfEta, errs_rfETa, resol_rfEta = response_curve(eta_reponse_ref, truth_eta, eta_bins)
    bins_comb, bin_errors_comb, means_comb, errs_comb, resol_comb = response_curve(pT_response_comb, truth_pt, pT_bins)
    bins_cEta, bins_errors_cEta, means_cEta, errs_cEta, resol_cEta = response_curve(pT_response_comb, truth_eta, eta_bins)

    fig = plt.figure(figsize=(5,5), dpi = 100)
    plt.plot(bins_ref, 100 * resol_ref, color='red', label='Final')
    plt.plot(bins_ref, 100 * resol_comb, color='black', label='Combined')
    plt.plot(bins_ref, 100 * resol_reg, color='purple', label='This work')
    plt.ylabel('$p_{T}(\\tau_{had-vis})$ resolution [%]', loc = 'top')
    plt.xlabel('True $p_{T}(\\tau_{had-vis})$ [GeV]', loc = 'right')
    plt.legend()
    plt.show()
    # plt.savefig(os.path.join(plotSaveLoc, 'pT_response.pdf'))
    # plt.close(fig)
    plt.close()

    fig = plt.figure(figsize=(5,5), dpi = 100)
    plt.plot(bins_rfEta, 100 * resol_rfEta, color='red', label='Final')
    plt.plot(bins_rfEta, 100 * resol_cEta, color='black', label='Combined')
    plt.plot(bins_rfEta, 100 * resol_rgEta, color='purple', label='This work')
    plt.ylabel('$\\eta (\\tau_{had-vis})$ resolution [%]', loc = 'top')
    plt.xlabel('True $\\eta (\\tau_{had-vis})$', loc = 'right')
    plt.legend()
    plt.show()