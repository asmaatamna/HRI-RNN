import torch
import numpy as np
import sklearn.metrics
import argparse
import modules
import hri_dataset

# Parse command line arguments
parser = argparse.ArgumentParser(description='Single run of our SED recurrent architecture')
parser.add_argument('epochs', type=int, default=10) # Number of training epochs
parser.add_argument('tau', type=int, default=5) # tau data parameter
parser.add_argument('eta', type=int, default=2) # tau data parameter
parser.add_argument('n_folds', type=int, default=10) # Train data fold number
parser.add_argument('lr', type=float, default=1e-3) # Optimizer learning rate
parser.add_argument('weight_decay', type=float, default=0.) # L2 regularization weight for optimizer
parser.add_argument('hidden_dims', nargs='+', type=int) # Dimensions of RNN's hidden states
parser.add_argument('architecture', type=str) # Dimensions of RNN's hidden states

args = parser.parse_args()

# Get number of training epochs
epochs = args.epochs

# Get tau and eta values
tau = args.tau
eta = args.eta

# Number of cross validation folds (3-cv for now)
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
res_dir = './' # str(Path.home()) + '/' + 'User-engagement-decrease-detection' + '/'
performance_dir = 'Output/performance-measures/'
data_dir = './HRI-data/'

# Performance lists
accuracies = []
precisions = []
recalls = []
F1_scores = [] # Default average: 'binary'
F1_scores_all_classes = [] # average: None
F1_scores_macro = [] # average: 'macro'
roc_auc_scores = [] # Default average: 'macro'

rebalance = False # Oversample test set (SED class)
robot_speaking_only = False # If True, load data where all sequences contain at least 1 robot audio

if robot_speaking_only:
    fname = '_robot_speaking_only'
else:
    fname = ''

# If true, robot audio is masked in SimpleRNN
user_data_only = False
if not user_data_only and architecture == 'SimpleRNN':
    fname_udata = '_all_data'
else:
    fname_udata = ''

attention_on = 0 # 0: no attention; 1: SimpleAttention; 2: MatchingAttention
if attention_on == 1:
    attention = '_SimpleAttention'
elif attention_on == 2:
    attention = '_MatchingAttention'
else:
    attention = ''

for fold in np.arange(1, n_folds + 1):
    
    # Load test data
    X_test = np.load(data_dir + 'X_test' + fname + '_tau_' + str(tau) + '_eta_' + str(eta) + '_fold_' + str(fold) + '.npy',
                      allow_pickle=True)
    Y_test = np.load(data_dir + 'Y_test' + fname + '_tau_' + str(tau) + '_eta_' + str(eta) + '_fold_' + str(fold) + '.npy',
                      allow_pickle=True)

    # Set batch size
    batch_size = len(Y_test)

    # Model parameters
    input_dim = X_test[0].shape[-1]
    
    # (PyTorch) dataset
    test_dataset = hri_dataset.HRIDataset(X_test, Y_test)

    # Data loader and (possibly) over-/down-sample test data
    labels = np.array([0., 1.])

    # Compute weights for over-/down-sampling
    if rebalance:
        counts = []
        for l in labels:
            counts.append(np.count_nonzero(Y_test == l))
        weights = 1. - torch.tensor(counts).double() / sum(counts)

        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights[test_dataset.labels],
                                                                 num_samples=len(test_dataset), replacement=True)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=sampler)
    else:
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    # Load trained model for evaluation
    model = modules.ClassificationModule(input_dim, hidden_dims, architecture=architecture, attend_over_context=attention_on)
    model.double()

    model.load_state_dict(torch.load(res_dir + 'Output/Model_' + architecture + attention + '_tau_' + str(tau) + '_eta_' + str(eta) +
           '_' + str(epochs) + '_epochs_hdim_' + str(hidden_dims) +
           '_lr_' + str(lr) + '_weight_decay_' + str(weight_decay) + fname + fname_udata + '_' + str(fold), map_location=torch.device('cpu')))
        
    model.eval()

    test_data, test_target = next(iter(test_dataloader))
    test_output = model(test_data.permute(1, 0, 2))

    # Accuracy
    accuracies.append(sklearn.metrics.accuracy_score(test_target.numpy(), torch.round(test_output).detach().numpy()))

    # Different test F1 scores
    F1_scores.append(sklearn.metrics.f1_score(test_target.numpy(), torch.round(test_output).detach().numpy()))
    F1_scores_all_classes.append(sklearn.metrics.f1_score(test_target.numpy(), torch.round(test_output).detach().numpy(), average=None))
    F1_scores_macro.append(sklearn.metrics.f1_score(test_target.numpy(), torch.round(test_output).detach().numpy(), average='macro'))

    # Precision and recall
    precisions.append(sklearn.metrics.precision_score(test_target.numpy(), torch.round(test_output).detach().numpy()))
    recalls.append(sklearn.metrics.recall_score(test_target.numpy(), torch.round(test_output).detach().numpy()))

    # ROC AUC scores
    roc_auc_scores.append(sklearn.metrics.roc_auc_score(test_target.numpy(), test_output.detach().numpy()))

# Print important metrics (mean & standard deviation)
F1_scores = np.array(F1_scores)
recalls = np.array(recalls)
precisions = np.array(precisions)
roc_auc_scores = np.array(roc_auc_scores)
accuracies = np.array(accuracies)

print("---------- {:}'s performance statistics, hidden dims: {:}, tau: {:}, eta: {:} ----------".format(architecture, hidden_dims, tau, eta))
print("F1 score: {:.2f} ± {:.3f}".format(np.mean(100 * F1_scores), np.std(100 * F1_scores))) # F1 score
print("Recall: {:.2f} ± {:.3f}".format(np.mean(100 * recalls), np.std(100 * recalls))) # Recall
print("Precision: {:.2f} ± {:.3f}".format(np.mean(100 * precisions), np.std(100 * precisions))) # Precision
print("ROC AUC score: {:.2f} ± {:.3f}".format(np.mean(100 * roc_auc_scores), np.std(100 * roc_auc_scores))) # ROC AUC
print("Accuracy: {:.2f} ± {:.3f}".format(np.mean(100 * accuracies), np.std(100 * accuracies))) # Accuracy