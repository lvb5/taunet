"""
Authors: Quentin Buat and Miles Cochran-Branson
Date: 6/24/22

Create plots of performance of machine learning analysis in comparison
to standard methods currently in place. 

Optional command-line arguments:

    --debug : run with only three files
    --copy-to-cernbox : copy plots to cernbox
    --path : specify path where plots are saved. If used, training *.h5 file
             may also be located in this folder without needing to specify a location 
             as in --model below. 
    --model : specify path to training model
"""
#from asyncio import subprocess
import subprocess
from genericpath import exists
import os
import tensorflow as tf

from taunet.database import PATH, DATASET, testing_data
from taunet.fields import FEATURES, TRUTH_FIELDS, OTHER_TES
if __name__ == '__main__':
    
    from taunet.parser import plot_parser
    # train_parser = argparse.ArgumentParser(parents=[common_parser])
    # train_parser.add_argument('--use-cache', action='store_true')
    args = plot_parser.parse_args()

    if args.debug:
        n_files = 2
    else:
        n_files = -1

    path = args.path # path to where training is stored
    # make plot folder if it doesn't already exist
    if not os.path.exists(os.path.join(path, 'plots')):
        cmd = 'mkdir -p {}'.format(os.path.join(path, 'plots'))
        subprocess.run(cmd, shell=True)
    # loads result of training to make plots 
    #! I did some weird things here... be careful when running with different 
    #! models, etc
    if path != '' and args.model == 'simple_dnn.h5':
        regressor = tf.keras.models.load_model(os.path.join(
        path, args.model))
    else:
        regressor = tf.keras.models.load_model(args.model)

    d = testing_data(
        PATH, DATASET, FEATURES, TRUTH_FIELDS + OTHER_TES, regressor, nfiles=n_files)

    from taunet.plotting import nn_history
    nn_history(os.path.join(path, "history.p"), path)

    from taunet.plotting import pt_lineshape
    pt_lineshape(d, path)

    from taunet.plotting import response_lineshape
    response_lineshape(d, path)

    from taunet.plotting import target_lineshape
    target_lineshape(d, plotSaveLoc=path)
    target_lineshape(d, bins=100, range=(0.5, 1.5), basename='tes_target_lineshape_zoomedin', logy=False)

    from taunet.plotting import response_and_resol_vs_pt
    response_and_resol_vs_pt(d, path)

    if args.copy_to_cernbox:
        from taunet.utils import copy_plots_to_cernbox
        if path != '':
            copy_plots_to_cernbox(location=path)
        else:
            copy_plots_to_cernbox()