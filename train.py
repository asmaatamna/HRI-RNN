import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
import numpy as np
import sklearn.metrics
import modules
import hri_dataset
import os
import argparse


def train_model(model,
                train_dataloader,
                validation_dataloader,
                criterion,
                optimizer,
                trained_model_name,
                epochs=50,
                weights=None,
                save_learning_curves=False,
                device=torch.device('cpu'),
                res_dir='./Output/',
                validate_every=1):
    """
    :param model: Model to train
    :param train_dataloader: Loads training data
    :param validation_dataloader: Loads validation data we use to monitor training & save the best model
    :param criterion: Loss function used for training
    :param optimizer: Optimizer used for training (Adam is usually the preferred choice)
    :param trained_model_name: Trained model is saved under this name
    :param epochs: Number of training epochs
    :param weights: Class weights. Used with imbalanced datasets
    :param save_learning_curves: Whether to save train and validation losses or not
    :param device: CPU or GPU
    :param res_dir: Directory where to store results (model and learning curves)
    :param validate_every: Evaluate performance (F1 score) on validation set every validate_every epochs
    :return: No return. This function saves the best model (on validation data) & learning curves (if the option is on)
    """

    # Load validation data at once
    validation_data, validation_target = next(iter(validation_dataloader))

    # Lists to save train & test losses
    if save_learning_curves:
        train_losses = []
        validation_losses = []

    # Initialize best validation F1 score
    best_validation_f1_score = 0.

    print("Training {:}".format(trained_model_name))

    # Training loop
    for epoch in range(epochs):
        # Set train mode
        model.train()

        # Initialize train_loss
        if save_learning_curves:
            train_loss = 0

        # Load training data batch
        for data, target in train_dataloader:
            # Clear gradients
            model.zero_grad()

            # Permute data axes so that it can be processed by the RNN, then run forward pass
            output = model(data.to(device).permute(1, 0, 2))

            # Compute loss, gradients, and update model parameters
            if weights is None:
                loss = criterion(output, target.to(device).double().view(-1, 1))
            else:
                loss = criterion(output, target.to(device).double().view(-1, 1),
                                 weight=weights[target.long()].double().view(-1, 1))

            loss.backward()
            optimizer.step()

            if save_learning_curves:
                # Update training loss
                train_loss += loss.item() * data.size(0)

        if save_learning_curves:
            # Compute average loss
            train_loss = train_loss / len(train_dataloader.dataset)

            # Save current train loss in a list
            train_losses.append(train_loss.item())

        # Evaluate model on validation data
        model.eval()
        validation_output = model(validation_data.to(device).permute(1, 0, 2))

        if save_learning_curves:
            validation_loss = criterion(validation_output, validation_target.to(device).double().view(-1, 1))
            validation_losses.append(validation_loss.item())

        # Save trained model every validate_every epoch if validation performance (F1 score) has improved
        if epoch % validate_every == 0:
            validation_f1_score = sklearn.metrics.f1_score(validation_target.numpy(),
                                                           torch.round(validation_output).detach().numpy())
            if validation_f1_score > best_validation_f1_score:
                best_validation_f1_score = validation_f1_score

                # Save learned model
                torch.save(model.state_dict(), res_dir + 'Model_' + trained_model_name)

                print("Model saved at epoch {:}. Validation F1 score: {:.2f}".format(epoch + 1,
                                                                                     best_validation_f1_score))

    print("End of training")

    if save_learning_curves:
        np.savetxt(res_dir + 'Train-losses_' + trained_model_name + '.txt', train_losses)
        np.savetxt(res_dir + 'Validation-losses_' + trained_model_name + '.txt', validation_losses)


def main():

    # Seed the random numbers generator for reproducibility
    torch.manual_seed(0)

    # Limit the number of threads created to parallelize CPU operations to 1
    torch.set_num_threads(1)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train selected model on one fold of HRI data for SED detection')
    parser.add_argument('--architecture', type=str, default='HriRNN')  # Model to train (HriRNN or SimpleRNN)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[32, 32])  # Size of RNN's hidden states
    parser.add_argument('--attention_on', type=int, default=0)  # Whether to use attention or not
    # 0: no attention, 1: SimpleAttention, 2: MatchingAttention
    parser.add_argument('--tau', type=int, default=5)  # Length in sec of HRI sequences
    parser.add_argument('--eta', type=int, default=2)  # Backward time horizon in sec used to label data
    parser.add_argument('--fold', type=int, default=1)  # Data folds number used for training & validation
    parser.add_argument('--epochs', type=int, default=50)  # Number of training epochs
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-3)  # Optimizer learning rate
    parser.add_argument('--weight_decay', type=float, default=0.)  # L2 regularization weight
    parser.add_argument('--save_learning_curves', type=int, default=0)  # Whether to save learning curves
    # (train & validation losses)
    parser.add_argument('--gpu_id', type=str, default='0')

    args = parser.parse_args()

    # Set GPU index
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # User data directory
    data_dir = './HRI-data/'
    res_dir = './Output/'

    # If true, robot audio is masked in SimpleRNN
    user_data_only = False

    X_train = np.load(data_dir + 'X_train' + '_tau_' + str(args.tau) + '_fold_' + str(args.fold) + '.npy',
                      allow_pickle=True)
    Y_train = np.load(data_dir + 'Y_train' + '_tau_' + str(args.tau) + '_eta_' + str(args.eta) + '_fold_' +
                      str(args.fold) + '.npy', allow_pickle=True)

    X_val = np.load(data_dir + 'X_validation' + '_tau_' + str(args.tau) + '_fold_' + str(args.fold) + '.npy',
                    allow_pickle=True)
    Y_val = np.load(data_dir + 'Y_validation' + '_tau_' + str(args.tau) + '_eta_' + str(args.eta) + '_fold_' +
                    str(args.fold) + '.npy', allow_pickle=True)

    # Model parameters
    input_dim = X_train[0].shape[-1]

    # Create (PyTorch) train & validation data sets
    train_dataset = hri_dataset.HRIDataset(X_train, Y_train)
    validation_dataset = hri_dataset.HRIDataset(X_val, Y_val)

    # Get all labels
    # Proper way to do it. Returns a sorted array of unique elements of the array
    # labels = np.unique(np.concatenate(Y_train))
    labels = np.array([0., 1.])  # Quicker solution

    # Class weights for loss function (equivalent to over/down sampling)
    counts = []
    for l in labels:
        counts.append(np.count_nonzero(Y_train == l))
    weights = 1. - torch.tensor(counts).double() / sum(counts)

    gpu_available = torch.cuda.is_available()
    device = torch.device('cuda:0') if gpu_available else torch.device('cpu')

    # Create train & validation data loaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=len(validation_dataset))

    # Set model parameters for the training
    model = modules.ClassificationModule(input_dim, args.hidden_dims, architecture=args.architecture,
                                         user_data_only=user_data_only, use_gpu=gpu_available,
                                         attend_over_context=args.attention_on)
    model.to(device)
    model.double()

    criterion = F.binary_cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Set filenames for saved data (model & learning curves)
    if args.attention_on == 1:
        attention = '_SimpleAttention'
    elif args.attention_on == 2:
        attention = '_MatchingAttention'
    else:
        attention = ''

    if user_data_only and args.architecture == 'SimpleRNN':
        udata = 'user_data_only_'
    else:
        udata = ''

    model_name = args.architecture + attention + '_tau_' + str(args.tau) + '_eta_' + str(args.eta) + '_hdim_' + \
                 str(args.hidden_dims) + '_lr_' + str(args.lr) + '_weight_decay_' + str(args.weight_decay) + '_' + \
                 str(args.epochs) + '_epochs_' + udata + str(args.fold)

    train_model(model, train_dataloader, validation_dataloader, criterion, optimizer, model_name, args.epochs,
                weights, args.save_learning_curves, device, res_dir, validate_every=1)


if __name__ == "__main__":
    main()
