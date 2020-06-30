import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import sys

# Get tau and eta values
tau = int(sys.argv[1])
eta = int(sys.argv[2])
n_folds = int(sys.argv[3]) # Number of cross validation folds. Ideal is 10. 5 is ok as well

# User data directory
data_dir = './HRI-data/'

# Load users data
X_all_users = np.load(data_dir + 'X_all_users_tau_' + str(tau) + '_eta_' + str(eta) + '.npy', allow_pickle=True)
Y_all_users = np.load(data_dir + 'Y_all_users_tau_' + str(tau) + '_eta_' + str(eta) + '.npy', allow_pickle=True)

# If True, only select data sequences where robot speaks at least once
robot_speaking_only = False

# Determines whether the splitting (in particular fot the test set) is user-dependent
user_dependent = False

# Model parameters
input_dim = X_all_users[0].shape[-1]
max_label = int(max([max(Y) for Y in Y_all_users]))
out = max_label + 1 if max_label > 1 else 1

# Select train & test data (test user IDs & train user IDs)
fold = 0

if user_dependent:
    X_all_users = np.concatenate(X_all_users)
    Y_all_users = np.concatenate(Y_all_users)

# Asma: Very specific treatment that consists in summing 2 originally last features
# (RobotSpeakDur & RobotListenDur) and swiping the IsListeningToRobot feature to last position
if user_dependent:
    for i in range(len(X_all_users)):
        tmp = np.copy(X_all_users[i][:, -3])
        X_all_users[i][:, -3] = X_all_users[i][:, -1] + X_all_users[i][:, -2]
        X_all_users[i][:, -2] = tmp
    X_all_users = X_all_users[:, :, :-1]  # Discard last dimension (redundant)
else:
    for j in range(len(X_all_users)):
        for i in range(len(X_all_users[j])):
            tmp = np.copy(X_all_users[j][i][:, -3])
            X_all_users[j][i][:, -3] = X_all_users[j][i][:, -1] + X_all_users[j][i][:, -2]
            X_all_users[j][i][:, -2] = tmp
        X_all_users[j] = X_all_users[j][:, :, :-1]  # Discard last dimension (redundant)

kf = KFold(n_splits=n_folds, shuffle=True, random_state=5)

for train_idx, test_idx in kf.split(X_all_users):
    fold += 1
    # Create train, test, and validation data sets, then reshape them as (total_seq_nb x seq_length x nb_feat)

    # Set 10% of training data for validation
    train_idx_bis, val_idx = train_test_split(train_idx, test_size=0.1, random_state=5) # test_size was: 0.2

    if user_dependent:
        X_train, X_test, X_val = X_all_users[train_idx_bis], X_all_users[test_idx], X_all_users[val_idx]
        Y_train, Y_test, Y_val =  Y_all_users[train_idx_bis], Y_all_users[test_idx], Y_all_users[val_idx]
    else:
        X_train, X_test, X_val = np.concatenate(X_all_users[train_idx_bis]), np.concatenate(X_all_users[test_idx]), np.concatenate(X_all_users[val_idx])
        Y_train, Y_test, Y_val = np.concatenate(Y_all_users[train_idx_bis]), np.concatenate(Y_all_users[test_idx]), np.concatenate(Y_all_users[val_idx])

    # Replace missing values with mean of corresponding feature & normalize data
    # (steps 1 & 2 below)
    #
    # 1. Fit SimpleImputer on training data (after collapsing first dimension)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X_train.reshape(-1, X_train.shape[-1]))

    # Apply trained imputer to train, test, and validation data
    X_train = np.asarray([imp.transform(a) for a in X_train])
    X_test = np.asarray([imp.transform(a) for a in X_test])
    X_val = np.asarray([imp.transform(a) for a in X_val])

    # 2. Fit StandardScaler on imputed training data (after collapsing first dimension)
    scaler = preprocessing.StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))

    # Apply scaler to train, test, and validation data (after imputation)
    X_train = np.asarray([scaler.transform(a) for a in X_train])
    X_test = np.asarray([scaler.transform(a) for a in X_test])
    X_val = np.asarray([scaler.transform(a) for a in X_val])

    # Select sequences where the robot speaks at least once
    if robot_speaking_only:
        selected_idx = []
        for i in range(len(X_train)):
            if len(np.where(X_train[i][:, -3] > 0)[0]) > 0:
                selected_idx.append(i)
        X_train = X_train[selected_idx]
        Y_train = Y_train[selected_idx]

        selected_idx = []
        for i in range(len(X_test)):
            if len(np.where(X_test[i][:, -3] > 0)[0]) > 0:
                selected_idx.append(i)
        X_test = X_test[selected_idx]
        Y_test = Y_test[selected_idx]

        selected_idx = []
        for i in range(len(X_val)):
            if len(np.where(X_val[i][:, -3] > 0)[0]) > 0:
                selected_idx.append(i)
        X_val = X_val[selected_idx]
        Y_val = Y_val[selected_idx]

    # Save processed data
    if user_dependent:
        fname = '_user_dependent'
    else:
        fname = ''

    if robot_speaking_only:
        fname += '_robot_speaking_only'

    np.save(data_dir + 'X_train' + fname + '_tau_' + str(tau) + '_eta_' + str(eta) + '_fold_' + str(fold), X_train)
    np.save(data_dir + 'Y_train' + fname + '_tau_' + str(tau) + '_eta_' + str(eta) + '_fold_' + str(fold), Y_train)

    np.save(data_dir + 'X_test' + fname + '_tau_' + str(tau) + '_eta_' + str(eta) + '_fold_' + str(fold), X_test)
    np.save(data_dir + 'Y_test' + fname + '_tau_' + str(tau) + '_eta_' + str(eta) + '_fold_' + str(fold), Y_test)

    np.save(data_dir + 'X_validation' + fname + '_tau_' + str(tau) + '_eta_' + str(eta) + '_fold_' + str(fold), X_val)
    np.save(data_dir + 'Y_validation' + fname + '_tau_' + str(tau) + '_eta_' + str(eta) + '_fold_' + str(fold), Y_val)