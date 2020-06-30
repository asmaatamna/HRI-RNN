import numpy as np
import sys
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Single run of our SED recurrent architecture')
parser.add_argument('epochs', type=int, default=10) # Number of training epochs
parser.add_argument('tau', type=int, default=5) # tau data parameter
parser.add_argument('eta', type=int, default=2) # tau data parameter
parser.add_argument('n_folds', type=int, default=3) # Train data fold number
parser.add_argument('lr', type=float, default=1e-3) # Optimizer learning rate
parser.add_argument('weight_decay', type=float, default=0.) # L2 regularization weight for optimizer
# parser.add_argument('projection_dim', type=int) # Dimension of projection space before GRU cells
parser.add_argument('hidden_dims', nargs='+', type=int) # Dimensions of RNN's hidden states
parser.add_argument('architecture', type=str, default='SimpleRNN') # Dimensions of RNN's hidden states

args = parser.parse_args()

# Get number of training epochs
epochs = args.epochs

# Get tau and eta values
tau = args.tau
eta = args.eta

# Number of cross validation folds
n_folds = args.n_folds

# Get learning rate and weight decay for optimizer
lr = args.lr
weight_decay = args.weight_decay

# Dimensions of RNN hidden states
hidden_dims = args.hidden_dims

# Dimension of projection space before GRU cells
# projection_dim = args.projection_dim

# Select an architecture
architecture = args.architecture # 'HriRNN' or 'TransformerRNN' for now

# User data directory
data_dir = './' # str(Path.home()) + '/' + 'User-engagement-decrease-detection' + '/'
performance_dir = 'Output/performance-measures/'

robot_speaking_only = False # If True, load data where all sequences contain at least 1 robot audio

if robot_speaking_only:
    fname = '_robot_speaking_only'
else:
    fname = ''

data = np.loadtxt(data_dir + performance_dir + 'Test-f1-scores_' + architecture + '_tau_' + str(tau) + '_eta_' + str(eta) + '_' + str(n_folds) + '-fold_cv_' + str(epochs) + '_epochs_hdim_' + str(hidden_dims) +
                  # '_projection_dim_' + str(projection_dim) +
                  '_lr_' + str(lr) + '_weight_decay_' + str(weight_decay) + fname + '.txt')
print("F1 score: {:.2f} ± {:.3f}".format(np.mean(100 * data), np.std(100 * data)))

data = np.loadtxt(data_dir + performance_dir + 'Test-recalls_' + architecture + '_tau_' + str(tau) + '_eta_' + str(eta) + '_' + str(n_folds) + '-fold_cv_' + str(epochs) + '_epochs_hdim_' + str(hidden_dims) +
                  # '_projection_dim_' + str(projection_dim) +
                  '_lr_' + str(lr) + '_weight_decay_' + str(weight_decay) + fname + '.txt')
print("Recall: {:.2f} ± {:.3f}".format(np.mean(100 * data), np.std(100 * data)))

data = np.loadtxt(data_dir + performance_dir + 'Test-precisions_' + architecture + '_tau_' + str(tau) + '_eta_' + str(eta) + '_' + str(n_folds) + '-fold_cv_' + str(epochs) + '_epochs_hdim_' + str(hidden_dims) +
                  # '_projection_dim_' + str(projection_dim) +
                  '_lr_' + str(lr) + '_weight_decay_' + str(weight_decay) + fname + '.txt')
print("Precision: {:.2f} ± {:.3f}".format(np.mean(100 * data), np.std(100 * data)))

data = np.loadtxt(data_dir + performance_dir + 'Test-roc-auc-scores_' + architecture + '_tau_' + str(tau) + '_eta_' + str(eta) + '_' + str(n_folds) + '-fold_cv_' + str(epochs) + '_epochs_hdim_' + str(hidden_dims) +
                  # '_projection_dim_' + str(projection_dim) +
                  '_lr_' + str(lr) + '_weight_decay_' + str(weight_decay) + fname + '.txt')
print("ROC AUC score: {:.2f} ± {:.3f}".format(np.mean(100 * data), np.std(100 * data)))

data = np.loadtxt(data_dir + performance_dir + 'Test-accuracies_' + architecture + '_tau_' + str(tau) + '_eta_' + str(eta) + '_' + str(n_folds) + '-fold_cv_' + str(epochs) + '_epochs_hdim_' + str(hidden_dims) +
                  # '_projection_dim_' + str(projection_dim) +
                  '_lr_' + str(lr) + '_weight_decay_' + str(weight_decay) + fname + '.txt')
print("Accuracy: {:.2f} ± {:.3f}".format(np.mean(100 * data), np.std(100 * data)))