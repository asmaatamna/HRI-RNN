import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import argparse


def ratio_positive_class(labels):
    """
    :param labels: Array of labels (0 and 1)
    :return: In the case of binary classification with labels 0 and 1, returns the ratio of labels 1
    """
    return 100 * np.count_nonzero(labels == 1) / len(labels)


def sed_ratio_over_all_data_folds(tau=5, eta=2, n_folds=5, data_dir='./HRI-data/'):
    """
    Prints the ratio of SED labels (1) in HRI data with parameters tau and eta (resp. interaction sequences length and
    labeling parameter) for each train, test, and validation set of the n_folds cross-validation folds

    :param tau: Length of data sequences in seconds
    :param eta: Length of backward time horizon used for SED annotation (in seconds)
    :param n_folds: Number of cross-validation data folds
    :param data_dir: directory containing HRI data
    :return: Prints SED ratio in train, test, and validation data for each data fold
    """
    for fold in np.arange(1, n_folds + 1):
        Y_train = np.load(data_dir + 'Y_train_tau_' + str(tau) + '_eta_' + str(eta) + '_fold_' + str(fold) + '.npy',
                          allow_pickle=True)
        Y_test = np.load(data_dir + 'Y_test_tau_' + str(tau) + '_eta_' + str(eta) + '_fold_' + str(fold) + '.npy',
                         allow_pickle=True)
        Y_validation = np.load(data_dir + 'Y_validation_tau_' + str(tau) + '_eta_' + str(eta) + '_fold_' + str(fold) +
                               '.npy', allow_pickle=True)

        # Print ratio of SED data in train, test, and validation data sets
        print("SED ratio in HRI data with tau = {:} and eta = {:}, fold {:}:".format(tau, eta, fold))
        print("     Train data: {:.2f}".format(ratio_positive_class(Y_train)))
        print("     Test data: {:.2f}".format(ratio_positive_class(Y_test)))
        print("     Validation data: {:.2f}".format(ratio_positive_class(Y_validation)))


def sed_ratio_vs_tau(tau_list=[5, 10, 20, 30, 40], eta_list=[2, 4, 8, 12, 16]):
    """
    :param tau_list: List of HRI datasets to process
    :param eta_list: List of SED labeling parameters
    :return: Prints the % of SED labels in HRI (overall) datasets with sequence lengths defined in tau_list
    """
    for tau, eta in zip(tau_list, eta_list):
        Y_all_users = np.concatenate(np.load('./HRI-data/Y_all_users_tau_' + str(tau) + '_eta_' + str(eta) + '.npy',
                                             allow_pickle=True))
        print("SED ratio in HRI data with tau = {:} and eta = {:}: {:.2f} %".format(tau, eta,
                                                                                    ratio_positive_class(Y_all_users)))


def ratio_sequences_where_robot_speaks(hri_data):
    """
    :param hri_data: Array of HRI sequences of size (batch, seq_len, dim)
    :return: Ratio of sequences where the robot speaks at least once
    """
    mask = (hri_data[:, :, -1] > 0) * 1  # The last feature indicates whether the robot is speaking (> 0)
    res = np.max(mask, axis=1)  # Contains 1 if the corresponding sequence contains robot's audio, 0 otherwise
    return 100 * sum(res) / len(res)


def ratio_sequences_where_robot_speaks_over_all_data_folds(tau=5, n_folds=5, data_dir='./HRI-data/'):
    """
    :param tau: Length of data sequences in seconds
    :param n_folds: Number of cross-validation data folds
    :param data_dir: directory containing HRI data
    :return: Ratio of sequences where the robot speaks at least once in train, test, and validation data for each
    cross-validation fold
    """
    train_ratios = []
    test_ratios = []
    val_ratios = []

    for fold in np.arange(1, n_folds + 1):
        X_train = np.load(data_dir + 'X_train_tau_' + str(tau) + '_fold_' + str(fold) + '.npy',
                          allow_pickle=True)
        X_test = np.load(data_dir + 'X_test_tau_' + str(tau) + '_fold_' + str(fold) + '.npy',
                         allow_pickle=True)
        X_val = np.load(data_dir + 'X_validation_tau_' + str(tau) + '_fold_' + str(fold) + '.npy',
                        allow_pickle=True)

        train_ratios.append(ratio_sequences_where_robot_speaks(X_train))
        test_ratios.append(ratio_sequences_where_robot_speaks(X_test))
        val_ratios.append(ratio_sequences_where_robot_speaks(X_val))

    return train_ratios, test_ratios, val_ratios


def plot_ratio_sequences_where_robot_speaks_vs_tau(tau_list=[5, 10, 20, 30, 40], n_folds=None):
    """
    :param tau_list: Sequence length (in sec) defining HRI datasets to go through
    :param n_folds: Number of cross-validation folds
    :return: Plots the % of sequences where the robot speaks for all train, test, and validation data folds and all
    values in tau_list if n_folds. Otherwise, it does it for the non-split HRI datasets
    """
    list_ratios = []

    if n_folds:
        for tau in tau_list:
            train_ratio, test_ratio, val_ratio = ratio_sequences_where_robot_speaks_over_all_data_folds(tau,
                                                                                                        n_folds=n_folds)
            list_ratios.append(np.transpose(np.array([train_ratio, test_ratio, val_ratio])))

        list_ratios = np.stack(list_ratios, axis=-1)

        plt.figure(figsize=(10, 10))
        axes = []
        for fold, data in enumerate(list_ratios):
            axes.append(plt.subplot(n_folds, 1, fold + 1))
            axes[-1].plot(tau_list, data[0], label="Train", marker="o")
            axes[-1].plot(tau_list, data[1], label="Test", marker="v")
            axes[-1].plot(tau_list, data[2], label="Val.", marker="s")
            axes[-1].title.set_text("Fold {:}".format(fold + 1))
            axes[-1].set_ylabel("% seq. where robot speaks")
            axes[-1].grid(True, which='both')
            axes[-1].set_ylim(50, 100)
        plt.xlabel(r'$\tau$' + " (sec)")
        plt.savefig("Ratio_sequences_where_robot_speaks_vs_tau_all_data_folds.pdf")
        plt.show()
    else:
        for tau in tau_list:
            X_all_users = np.concatenate(np.load('./HRI-data/X_all_users_tau_' + str(tau) + '.npy', allow_pickle=True))
            list_ratios.append(ratio_sequences_where_robot_speaks(X_all_users))
        plt.plot(tau_list, list_ratios, marker="o")
        plt.xlabel(r'$\tau$' + " (sec)")
        plt.ylabel("% seq. where robot speaks")
        plt.grid(True, which='both')
        plt.savefig("Ratio_sequences_where_robot_speaks_vs_tau.pdf")
        plt.show()


def distribution_robot_speaking_duration(hri_data):
    """
    :param hri_data: Array of HRI sequences of size (batch, seq_len, dim)
    :return: Robot's speech duration (in sec) in each input HRI data sequence (distribution)
    """
    mask = (hri_data[:, :, -1] > 0) * 1  # The last feature indicates whether the robot is speaking (> 0)
    res = np.sum(mask, axis=1) / 2  # Contains the robot's speech duration (in sec) in each HRI sequence
    # We divide by 2 to recover duration in sec (feature vectors are extracted every 0.5 sec)
    return res


def plot_distribution_robot_speaking_duration_vs_tau(tau_list=[5, 10, 20, 30, 40], n_folds=None):
    """
    :param tau_list: Sequence length (in sec) defining HRI datasets to go through
    :param eta: SED labeling parameter
    :param n_folds: Number of cross-validation folds
    :return: Plots the distribution of the robot's speaking duration in one HRI sequence for all the datasets indicated
    in tau_list
    """
    list_ecdfs = []

    if n_folds:
        pass  # TODO: To be implemented
    else:
        for tau in tau_list:
            X_all_users = np.concatenate(np.load('./HRI-data/X_all_users_tau_' + str(tau) + '.npy', allow_pickle=True))
            list_ecdfs.append(ECDF(distribution_robot_speaking_duration(X_all_users)))

        for i, tau in enumerate(tau_list):
            plt.plot(list_ecdfs[i].x, list_ecdfs[i].y, label=r'$\tau = {}$'.format(tau))
        plt.xlabel("Cumulative distrib. of robot speaking duration")
        plt.grid(True, which='both')
        plt.legend(loc='best', fancybox=True, framealpha=0.5)
        plt.savefig("ECDF_robot_speaking_duration_vs_tau.pdf")
        plt.show()


def distribution_speaker_changes(hri_data):
    """
    :param hri_data: Array of HRI sequences of size (batch, seq_len, dim)
    :return: List containing the number of times the speaker changes (user, robot) in each input HRI data sequence
    (distribution of speaker changes)

    TODO: itertools.groupby isn't the most efficient way to go. Improve
    """
    mask = (hri_data[:, :, -1] > 0) * 1  # The last feature indicates whether the robot is speaking (> 0)
    return [len(list(groupby(m, lambda x: x > 0))) - 1 for m in mask]


def plot_distribution_speaker_changes_vs_tau(tau_list=[5, 10, 20, 30, 40], n_folds=None):
    """
    :param tau_list: Sequence length (in sec) defining HRI datasets to go through
    :param eta: SED labeling parameter
    :param n_folds: Number of cross-validation folds
    :return: Plots the distribution of speaker changes in one HRI sequence for all the datasets indicated
    in tau_list
    """
    list_ecdfs = []

    if n_folds:
        pass  # TODO: To be implemented
    else:
        for tau in tau_list:
            X_all_users = np.concatenate(np.load('./HRI-data/X_all_users_tau_' + str(tau) + '.npy', allow_pickle=True))
            list_ecdfs.append(ECDF(distribution_speaker_changes(X_all_users)))

        for i, tau in enumerate(tau_list):
            plt.plot(list_ecdfs[i].x, list_ecdfs[i].y, label=r'$\tau = {}$'.format(tau))
        plt.xlabel("Empirical cumulative distrib. of the number of speaker changes")
        plt.grid(True, which='both')
        plt.legend(loc='best', fancybox=True, framealpha=0.5)
        plt.savefig("ECDF_speaker_changes_vs_tau.pdf")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Functions for computing various statistics on HRI data')
    parser.add_argument('--selection', type=int, default=1)
    parser.add_argument('--tau', type=int, default=5)
    parser.add_argument('--eta', type=int, default=2)
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--data_dir', type=str, default='./HRI-data/')

    #  TODO: To be completed

    sed_ratio_vs_tau()


if __name__ == "__main__":
    main()
