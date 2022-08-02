import subprocess
from genericpath import exists
import os
import uproot

from taunet.database import PATH, DATASET, testing_data
from taunet.fields import FEATURES, TRUTH_FIELDS, OTHER_TES
if __name__ == '__main__':
    
    from taunet.parser import plot_parser
    # train_parser = argparse.ArgumentParser(parents=[common_parser])
    # train_parser.add_argument('--use-cache', action='store_true')
    args = plot_parser.parse_args()

    if args.debug:
        n_files = 3
    else:
        n_files = args.nfiles

    path = args.path # path to where training is stored
    # make plot folder if it doesn't already exist
    if not os.path.exists(os.path.join(path, 'plots')):
        cmd = 'mkdir -p {}'.format(os.path.join(path, 'plots'))
        subprocess.run(cmd, shell=True)
    # load potentially required packages
    if not args.use_cache:
        import tensorflow as tf
        import tensorflow_probability as tfp
    # loads result of training to make plots 
    #! I did some weird things here... be careful when running with different 
    #! models, etc
    from taunet.computation import tf_mdn_loss
    if path != '' and not args.use_cache:
        regressor = tf.keras.models.load_model(os.path.join(path, args.model), custom_objects={'MixtureNormal': tfp.layers.MixtureNormal, 'tf_mdn_loss': tf_mdn_loss})
    elif not args.use_cache:
        regressor = tf.keras.models.load_model(os.path.join('cache', args.model), custom_objects={'MixtureNormal': tfp.layers.MixtureNormal, 'tf_mdn_loss': tf_mdn_loss})
    else:
        regressor = ''

    d = testing_data(
        PATH, DATASET, FEATURES, TRUTH_FIELDS + OTHER_TES, regressor, nfiles=n_files, 
        optional_path=path, select_1p=args.oneProng, select_3p=args.threeProngs, normIndices=list(map(int, args.normIDs)),
        no_normalize=args.no_normalize, no_norm_target=args.no_norm_target, debug=args.debug)

    # write each variable in d to .root file
    var = list(TRUTH_FIELDS + OTHER_TES + ['regressed_target'])
    for i in range(len(var)):
        file = uproot.recreate("rootfiles/{}.root".format(var[i]))
        file["tree"] = {"branch": d[var[i]]}
        print(var[i])
