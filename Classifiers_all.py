
# #### Import libraries
import numpy as np
import h5py
import csv
import sys

# observations: 1533 positive and 1375227 negative for 1376760 total

# FILE_ROOT = '/home/local/GACL/rpsander/Python'
FILE_ROOT = '/scrfs/storage/rpsander/home'
DATA_DIR = 'correct_data'
DATA_EXT = 'split'

TIME_AGG = sys.argv[1]
SEQ_LEN = sys.argv[2]
PAT_SAMPLE = 'pat_sample_all'
# WEIGHT = {0: 1.0, 1: 897.08}
data_file_name = '{0}_{1}_{2}'.format(PAT_SAMPLE, TIME_AGG, SEQ_LEN)

with h5py.File('{0}/{1}/{2}_{3}.hdf5'.format(FILE_ROOT, DATA_DIR, data_file_name, DATA_EXT), 'r') as f:
    num_samples = f['y_train'].shape[0]
    num_pos = np.sum(f['y_train'])
WEIGHT = {0: 1.0, 1: (num_samples - num_pos) / num_pos}


def run_classifiers():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
    model_name = 'Classifiers_correctdata'
    feature_importance = []
    for kfold in range(5):

        rf = RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1, n_jobs=-1, class_weight=WEIGHT)

        with h5py.File('{0}/{1}/{2}_{3}.hdf5'.format(FILE_ROOT, DATA_DIR, data_file_name, DATA_EXT), 'r') as f:
            X_train = f['fold_{0}'.format(str(kfold))]['X_val'][:]
            y_train = f['fold_{0}'.format(str(kfold))]['y_val'][:]
        y_train = np.array(y_train)
        X_train = np.array([x.ravel() for x in X_train])
        num_train = y_train.shape[0]

        rf.fit(X_train, y_train)
        del X_train, y_train

        with h5py.File('{0}/{1}/{2}_{3}.hdf5'.format(FILE_ROOT, DATA_DIR, data_file_name, DATA_EXT), 'r') as f:
            y_test = f['y_test'][:num_train]
            X_test = f['X_test'][:num_train]
        y_test1 = np.array(y_test)
        X_test = np.array([x.ravel() for x in X_test])
        y_pred1 = rf.predict(X_test)

        with h5py.File('{0}/{1}/{2}_{3}.hdf5'.format(FILE_ROOT, DATA_DIR, data_file_name, DATA_EXT), 'r') as f:
            X_test = f['X_test'][num_train:]
            y_test = f['y_test'][num_train:]
        y_test = np.concatenate((y_test1, np.array(y_test)), axis=0)
        X_test = np.array([x.ravel() for x in X_test])
        y_pred = np.concatenate((y_pred1, rf.predict(X_test)), axis=0)
        del y_pred1, y_test1, X_test

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        feature_importance.append(rf.feature_importances_)
        print(rf.feature_importances_)

        with open('{0}/{1}_kscores.csv'.format(FILE_ROOT, model_name), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['classifier', 'FOLD', 'TIME_AGG', 'SEQ_LEN', 'accuracy', 'recall', 'precision', 'f1', 'auc'])
            writer.writerow([rf, kfold, TIME_AGG, SEQ_LEN, accuracy, recall, precision, f1, auc])

        del rf

    with open('{0}/{1}_Kfeatureimportance_{2}_{3}.csv'.format(FILE_ROOT, model_name, TIME_AGG, SEQ_LEN), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['TIME_AGG', TIME_AGG])
        writer.writerow(['SEQ_LEN', SEQ_LEN])
        writer.writerow(['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4'])
        for feat in range(len(feature_importance[0])):
            writer.writerow([feature_importance[0][feat],
                             feature_importance[1][feat],
                             feature_importance[2][feat],
                             feature_importance[3][feat],
                             feature_importance[4][feat]])


if __name__ == "__main__":
    run_classifiers()