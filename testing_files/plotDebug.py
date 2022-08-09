import subprocess
from genericpath import exists
import os
import numpy as np
from matplotlib import pyplot as plt

from taunet.database import PATH, DATASET, testing_data
from taunet.fields import FEATURES, TRUTH_FIELDS, OTHER_TES

from taunet.parser import plot_parser
# train_parser = argparse.ArgumentParser(parents=[common_parser])
# train_parser.add_argument('--use-cache', action='store_true')
args = plot_parser.parse_args()

if not args.use_cache:
    import tensorflow as tf

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
if path != '' and not args.use_cache:
    regressor = tf.keras.models.load_model(os.path.join(path, args.model))
elif not args.use_cache:
    regressor = tf.keras.models.load_model(os.path.join('cache', args.model))
else:
    regressor = ''
    #print("Using cached data")

d = testing_data(
    PATH, DATASET, FEATURES, TRUTH_FIELDS + OTHER_TES + FEATURES, regressor, nfiles=n_files, 
    optional_path=path, select_1p=args.oneProng, select_3p=args.threeProngs, 
    no_normalize=args.no_normalize, no_norm_target=args.no_norm_target, normIndices=list(map(int, args.normIDs)), 
    saveToCache=args.add_to_cache, useCache=args.use_cache)

bins = [
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

dnew = [np.empty(len(d)) for _ in range(len(bins))]
for i in range(len(bins)):
    dnew[i] = d[d['TauJetsAuxDyn.truthPtVisDressed'] / 1000. > bins[i][0]]
    dnew[i] = d[d['TauJetsAuxDyn.truthPtVisDressed'] / 1000. < bins[i][1]]

# from taunet.plotting import response_lineshape
# response_lineshape(d, 'debug_plots', 'plots/all.pdf')
# for i in range(len(bins)):
#     response_lineshape(dnew[i], 'debug_plots', 'plots/cut{}.pdf'.format(i))

# response_reg = d['regressed_target'] * d['TauJetsAuxDyn.ptCombined'] / d['TauJetsAuxDyn.truthPtVisDressed']
# truth_pt = d['TauJetsAuxDyn.truthPtVisDressed'] / 1000.

# from taunet.utils import response_curve
# bins_reg, bin_errors_reg, means_reg, errs_reg, resol_reg = response_curve(response_reg, truth_pt, bins)

response_reg6 = dnew[6]['regressed_target'] * dnew[6]['TauJetsAuxDyn.ptCombined'] / dnew[6]['TauJetsAuxDyn.truthPtVisDressed']

# print(len(response_reg6))
# print(np.mean(response_reg6))
# lim = 10000
# print(sum(response_reg6 > lim))
# print(np.mean(response_reg6[response_reg6 < lim]))
# print(response_reg6[response_reg6 > lim])

print(max(dnew[6]['regressed_target'] * dnew[6]['TauJetsAuxDyn.ptCombined']))
print(sum(dnew[6]['regressed_target'] * dnew[6]['TauJetsAuxDyn.ptCombined'] > 1e6))

print(max(dnew[6]['regressed_target']))
print(sum(dnew[6]['regressed_target'] > 1e5))

# plt.figure(figsize = (5,5))
# plt.yscale('log')
# plt.hist(d['regressed_target'], bins = 200)
# plt.xlabel('regressed target')
# plt.ylabel('counts')
# plt.savefig('debug_plots/plots/regressed_target.pdf')

temp = d['regressed_target'] > 1e5
plot_vars = TRUTH_FIELDS + OTHER_TES + FEATURES + ['regressed_target']
for i in range(len(temp)):
    if temp[i]:
        for j in range(len(d[i])):
            #print(plot_vars[j], " = ", d[i][j])
            print(plot_vars[j], " = ", d[0][j])

# from taunet.utils import copy_plots_to_cernbox
# copy_plots_to_cernbox(location='debug_plots')