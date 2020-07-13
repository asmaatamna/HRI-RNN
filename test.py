import torch
import numpy as np
import sklearn.metrics
import argparse
import modules
import hri_dataset


def test_model(model, test_dataloader):
    """
    Evaluates model on test data loaded by test_dataloader
    :param model: Model to evaluate
    :param test_dataloader: Loads test data
    :return: Test accuracy, F1 score, precision, recall, and ROC AUC score
    """

    model.eval()

    test_data, test_target = next(iter(test_dataloader))
    test_output = model(test_data.permute(1, 0, 2))

    # Accuracy
    accuracy = sklearn.metrics.accuracy_score(test_target.numpy(), torch.round(test_output).detach().numpy())

    # F1 score
    F1_score = sklearn.metrics.f1_score(test_target.numpy(), torch.round(test_output).detach().numpy())

    # Precision and recall
    precision = sklearn.metrics.precision_score(test_target.numpy(), torch.round(test_output).detach().numpy())
    recall = sklearn.metrics.recall_score(test_target.numpy(), torch.round(test_output).detach().numpy())

    # ROC AUC score
    roc_auc_score = sklearn.metrics.roc_auc_score(test_target.numpy(), test_output.detach().numpy())

    return accuracy, F1_score, precision, recall, roc_auc_score


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate selected model on HRI test data folds')
    parser.add_argument('--architecture', type=str, default='HriRNN')  # Model to train (HriRNN or SimpleRNN)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[32, 32])  # Size of RNN's hidden states
    parser.add_argument('--attention_on', type=int, default=0)  # Whether to use attention or not
    # 0: no attention, 1: SimpleAttention, 2: MatchingAttention
    parser.add_argument('--tau', type=int, default=5)  # Length in sec of HRI sequences
    parser.add_argument('--eta', type=int, default=2)  # Backward time horizon in sec used to label data
    parser.add_argument('--n_folds', type=int, default=5)  # Number of cross-validation folds. Default: 5
    parser.add_argument('--epochs', type=int, default=50)  # Number of training epochs
    parser.add_argument('--lr', type=float, default=1e-3)  # Optimizer learning rate
    parser.add_argument('--weight_decay', type=float, default=0.)  # L2 regularization weight

    args = parser.parse_args()

    # User data directory
    data_dir = './HRI-data/'
    res_dir = './Output/'

    # Prepare names of files to load
    user_data_only = False  # If true, robot audio is masked in SimpleRNN
    if not user_data_only and args.architecture == 'SimpleRNN':
        udata = ''
    else:
        udata = 'user_data_only_'

    if args.attention_on == 1:
        attention = '_SimpleAttention'
    elif args.attention_on == 2:
        attention = '_MatchingAttention'
    else:
        attention = ''

    # Performance arrays
    accuracies = np.array([])
    precisions = np.array([])
    recalls = np.array([])
    F1_scores = np.array([])
    roc_auc_scores = np.array([])

    for fold in np.arange(1, args.n_folds + 1):
        # Load test data
        X_test = np.load(
            data_dir + 'X_test' + '_tau_' + str(args.tau) + '_eta_' + str(args.eta) + '_fold_' + str(fold) +
            '.npy', allow_pickle=True)
        Y_test = np.load(
            data_dir + 'Y_test' + '_tau_' + str(args.tau) + '_eta_' + str(args.eta) + '_fold_' + str(fold) +
            '.npy', allow_pickle=True)

        # Set batch size
        batch_size = len(Y_test)

        # Model parameters
        input_dim = X_test[0].shape[-1]

        # (PyTorch) dataset
        test_dataset = hri_dataset.HRIDataset(X_test, Y_test)

        # Prepare test data loader
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

        # Load trained model for evaluation
        model = modules.ClassificationModule(input_dim, args.hidden_dims, architecture=args.architecture,
                                             attend_over_context=args.attention_on)
        model.double()

        model_name = args.architecture + attention + '_tau_' + str(args.tau) + '_eta_' + str(args.eta) + '_hdim_' +\
                     str(args.hidden_dims) + '_lr_' + str(args.lr) + '_weight_decay_' + str(args.weight_decay) +\
                     '_' + str(args.epochs) + '_epochs_' + udata

        model.load_state_dict(torch.load(res_dir + 'Model_' + model_name + str(args.fold),
                                         map_location=torch.device('cpu')))

        # Evaluate model on current test data fold
        accuracy, F1_score, precision, recall, roc_auc_score = test_model(model, test_dataloader)

        accuracies = np.append(accuracies, accuracy)
        F1_scores = np.append(F1_scores, F1_score)
        precisions = np.append(precisions, precision)
        recalls = np.append(recalls, recall)
        roc_auc_scores = np.append(roc_auc_scores, roc_auc_score)

    # Save test performance (mean & standard deviation) in a text file
    file = open("Test_performance_" + model_name + ".txt", "w")
    file.write("F1 score: {:.2f} ± {:.3f}".format(np.mean(100 * F1_scores), np.std(100 * F1_scores)))
    file.write("Recall: {:.2f} ± {:.3f}".format(np.mean(100 * recalls), np.std(100 * recalls)))
    file.write("Precision: {:.2f} ± {:.3f}".format(np.mean(100 * precisions), np.std(100 * precisions)))
    file.write("ROC AUC score: {:.2f} ± {:.3f}".format(np.mean(100 * roc_auc_scores), np.std(100 * roc_auc_scores)))
    file.write("Accuracy: {:.2f} ± {:.3f}".format(np.mean(100 * accuracies), np.std(100 * accuracies)))
    file.close()


if __name__ == "__main__":
    main()
