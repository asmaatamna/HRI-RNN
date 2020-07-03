import torch
# Seed RNG for reproducibility
torch.manual_seed(0)
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
import numpy as np
import sklearn.metrics
import modules
import hri_dataset
import os
import argparse

# Limit number of threads created to parallelize CPU operations to 1
torch.set_num_threads(1)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Single run of our SED recurrent architecture')
parser.add_argument('epochs', type=int, default=10) # Number of training epochs
parser.add_argument('tau', type=int, default=5) # tau data parameter
parser.add_argument('eta', type=int, default=2) # tau data parameter
parser.add_argument('fold', type=int, default=1) # Train data fold number
parser.add_argument('lr', type=float, default=1e-3) # Optimizer learning rate
parser.add_argument('weight_decay', type=float, default=0.) # L2 regularization weight for optimizer
parser.add_argument('hidden_dims', nargs='+', type=int) # Dimensions of RNN's hidden states
parser.add_argument('architecture', type=str) # Selected model
parser.add_argument('--gpu_id', default='0', type=str)

args = parser.parse_args()

# Get number of training epochs
epochs = args.epochs

# Get tau and eta values
tau = args.tau
eta = args.eta

# Train data fold number
fold = args.fold

# Get learning rate and weight decay for optimizer
lr = args.lr
weight_decay = args.weight_decay

# Dimensions of RNN hidden states
hidden_dims = args.hidden_dims

# Select an architecture
architecture = args.architecture # 'HriRNN' or 'SimpleRNN' for now

# Set GPU index
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# User data directory
data_dir = './HRI-data/' # str(Path.home()) + '/' + 'User-engagement-decrease-detection' + '/'
res_dir = './'

# Whether to use attention in the models or not
attention_on = 0 # 0: no attention, 1: SimpleAttention, 2: MatchingAttention

# Set filenames accordingly
if attention_on == 1:
    attention = '_SimpleAttention'
elif attention_on == 2:
    attention = '_MatchingAttention'
else:
    attention = ''

# If true, robot audio is masked in SimpleRNN
user_data_only = False
if not user_data_only and architecture == 'SimpleRNN':
    fname_udata = 'all_data_'
else:
    fname_udata = ''

X_train = np.load(data_dir + 'X_train' + '_tau_' + str(tau) + '_eta_' + str(eta) + '_fold_' + str(fold) + '.npy', allow_pickle=True)
Y_train = np.load(data_dir + 'Y_train' + '_tau_' + str(tau) + '_eta_' + str(eta) + '_fold_' + str(fold) + '.npy', allow_pickle=True)

X_val = np.load(data_dir + 'X_validation' + '_tau_' + str(tau) + '_eta_' + str(eta) + '_fold_' + str(fold) + '.npy', allow_pickle=True)
Y_val = np.load(data_dir + 'Y_validation' + '_tau_' + str(tau) + '_eta_' + str(eta) + '_fold_' + str(fold) + '.npy', allow_pickle=True)

# Model parameters
input_dim = X_train[0].shape[-1]

# Hyperparameters
batch_size = 5000

# Save training/validation losses
save_training_info = False

# Lists to save train & test losses
if save_training_info:
    train_losses = []
    validation_losses = []

# Create (PyTorch) train & validation data sets
train_dataset = hri_dataset.HRIDataset(X_train, Y_train)
validation_dataset = hri_dataset.HRIDataset(X_val, Y_val)

# Get all labels
# Proper way to do it. Returns a sorted array of unique elements of the array
# labels = np.unique(np.concatenate(Y_train))
labels = np.array([0., 1.]) # Quicker solution

# Compute weights for over-/down-sampling
counts = []
for l in labels:
    counts.append(np.count_nonzero(Y_train == l))
weights = 1. - torch.tensor(counts).double() / sum(counts)

gpu_available = torch.cuda.is_available()
if gpu_available:
    device = torch.device('cuda:0')
    pin_memory = False
else:
    device = torch.device('cpu')
    pin_memory = False

# Create train & validation data loaders
# sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights[train_dataset.labels], num_samples=len(train_dataset), replacement=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=pin_memory, shuffle=True)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=len(validation_dataset), pin_memory=pin_memory)

# Set model parameters for the training 
model = modules.ClassificationModule(input_dim, hidden_dims, architecture=architecture, user_data_only=user_data_only,
                                     use_gpu=gpu_available, attend_over_context=attention_on)
model.to(device)
model.double()

criterion = F.binary_cross_entropy
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Load validation data once
validation_data, validation_target = next(iter(validation_dataloader))

# Validation parameters
validate_every = 1 # Change for longer experiments
best_validation_f1_score = 0.

print("-------------------- Architecture: {:}, hidden dim.: {:}, tau: {:}, eta: {:}, fold: {:} --------------------".format(architecture, hidden_dims, tau, eta, fold))

# Training loop
for epoch in range(epochs):
    # Set train mode
    model.train()

    # Initialize train_loss
    if save_training_info:
        train_loss = 0

    # Load training data batch
    for data, target in train_dataloader:
        # Clear gradients
        model.zero_grad()

        # First, permute data axes so that it can be processed by the RNN,
        # then run forward pass
        output = model(data.to(device).permute(1, 0, 2))

        # Compute loss, gradients, and update model parameters
        loss = criterion(output, target.double().view(-1, 1), weight=weights[target.long()].double().view(-1, 1))
        # loss = criterion(output, target.to(device).double().view(-1, 1))

        loss.backward()
        optimizer.step()

        if save_training_info:
            # Update training loss
            train_loss += loss.item() * data.size(0)

    if save_training_info:
        # Compute average loss
        train_loss = train_loss / len(train_dataloader.dataset)

        # Save current train loss in a list
        train_losses.append(train_loss)

    # Evaluate model on validation data
    model.eval()
    validation_output = model(validation_data.to(device).permute(1, 0, 2))

    if save_training_info:
        validation_loss = criterion(validation_output, validation_target.to(device).double().view(-1, 1))
        validation_losses.append(validation_loss.item())

    # Save model if improved validation performance every validate_every epoch
    if epoch % validate_every == 0:
        validation_f1_score = sklearn.metrics.f1_score(validation_target.numpy(), torch.round(validation_output).detach().numpy())
        if validation_f1_score > best_validation_f1_score:
            best_validation_f1_score = validation_f1_score

            # Save learned model
            torch.save(model.state_dict(), res_dir + 'Output/Model_' + architecture + attention + '_tau_' + str(tau) + '_eta_' + str(eta) +
                       '_' + str(epochs) + '_epochs_hdim_' + str(hidden_dims) + '_lr_' + str(lr) + '_weight_decay_' + str(weight_decay) +
                       '_' + fname_udata + str(fold))

            print("Saved model at epoch {:}. Validation F1 score: {:.2f}".format(epoch + 1, best_validation_f1_score))

if save_training_info:
    np.savetxt(res_dir + 'Train-losses_' + architecture + attention + '_tau_' + str(tau) + '_eta_' + str(eta) + '_' + str(epochs) +
               '_epochs_hdim_' + str(hidden_dims) + '_lr_' + str(lr) + '_weight_decay_' + str(weight_decay) +
               '_' + fname_udata + str(fold) + '.txt', train_losses)
    np.savetxt(res_dir + 'Validation-losses_' + architecture + attention + '_tau_' + str(tau) + '_eta_' + str(eta) + '_' + str(epochs) +
               '_epochs_hdim_' + str(hidden_dims) + '_lr_' + str(lr) + '_weight_decay_' + str(weight_decay) +
               '_' + fname_udata + str(fold) + '.txt', validation_losses)