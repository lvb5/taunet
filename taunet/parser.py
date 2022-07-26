import argparse
from tokenize import Double

common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument('--debug', default=False, action='store_true')
common_parser.add_argument('--nfiles', default=-1, type=int)
common_parser.add_argument('--no-norm-target', default=False, action='store_true')
common_parser.add_argument('--no-normalize', default=False, action='store_true')
common_parser.add_argument('--normIDs', default = ['0', '1', '2', '3', '4', '5', '6', '7', '8'], type=lambda x: x.split(','))
common_parser.add_argument('--use-cache', default=False, action='store_true')
common_parser.add_argument('--add-to-cache', default=False, action='store_true')
prongs = common_parser.add_mutually_exclusive_group()
prongs.add_argument('--oneProng', default=False, action='store_true')
prongs.add_argument('--threeProngs', default=False,  action='store_true')

# parse commands for training (fit.py)
train_parser = argparse.ArgumentParser(parents=[common_parser])
train_parser.add_argument('--rate', default=1e-5, type=float)
train_parser.add_argument('--batch-size', default=64, type=int)
train_parser.add_argument('--small-model', default=False, action='store_true')
train_parser.add_argument('--big-model', default=False, action='store_true')
train_parser.add_argument('--big-model-regular', default=False, action='store_true')
train_parser.add_argument('--small-2gauss', default=False, action='store_true')
train_parser.add_argument('--multi-gauss', default=False, action='store_true')


# parse commands for plotting (plot.py)
plot_parser = argparse.ArgumentParser(parents=[common_parser])
plot_parser.add_argument('--model', default='simple_dnn.h5')
plot_parser.add_argument('--copy-to-cernbox', default=False, action='store_true')
plot_parser.add_argument('--path', default='')
