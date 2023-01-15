import pandas as pd
import numpy as np
import seaborn as sn
import scipy.stats as stats
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score, KFold


def dataset_cleanup(X):
    #Datatable cleanup
    for i, item in enumerate(X['forward_bps_var']):
        match = re.findall(r"(?=[E+]).*", item)
        if match != []:
            item = item.replace(match[0], "")
        X['forward_bps_var'][i] = float(item)
    X['forward_bps_var'] = pd.to_numeric(X['forward_bps_var'])

    print(X.info())
    return X


def decision_tree_classifier(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.3)
    clf = DecisionTreeClassifier(random_state=0, max_depth=2)
    clf.fit(X_train, Y_train)
    return clf, X_train, X_test, Y_train, Y_test


def performance_evaluation(clf, X_test, Y_test, X_train, Y_train, uniques):
    cv = KFold(n_splits=10, random_state=0, shuffle=True)
    accuracy = clf.score(X_test, Y_test)
    KFold10_accuracy = cross_val_score(clf, X_train, Y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    print("KFold accuracy: ", KFold10_accuracy)
    predict = clf.predict(X_test)
    cm = confusion_matrix(Y_test, predict)
    precision = precision_score(Y_test, predict, average='weighted', labels=np.unique(predict))
    recall = recall_score(Y_test, predict, average='weighted', labels=np.unique(predict))
    f1scoreMacro = f1_score(Y_test, predict, average='macro', labels=np.unique(predict))
    print(classification_report(Y_test, predict, target_names=uniques))
    return cm


def find_the_10_most_important_features(clf):
    importance = clf.feature_importances_
    important_feature_dict = {}
    for idx, val in enumerate(importance):
        important_feature_dict[idx] = val
    important_feature_list = sorted(important_feature_dict, key=important_feature_dict.get, reverse=True)
    print(f'10 most important features: {important_feature_list[:10]}')


def plot_decision_tree_and_confusion_matrix(clf, unique, cm):
    fn = ['tp_src', 'tp_dst', 'nw_proto',
    'forward_pc', 'forward_bc', 'forward_pl', 'forward_piat', 'forward_pps',
    'forward_bps', 'forward_pl_mean', 'forward_piat_mean',
    'forward_pps_mean', 'forward_bps_mean', 'forward_pl_var',
    'forward_piat_var', 'forward_pps_var', 'forward_bps_var',
    'forward_pl_q1', 'forward_pl_q3', 'forward_piat_q1', 'forward_piat_q3',
    'forward_pl_max', 'forward_pl_min', 'forward_piat_max',
    'forward_piat_min', 'forward_pps_max', 'forward_pps_min',
    'forward_bps_max', 'forward_bps_min', 'forward_duration',
    'forward_size_packets', 'forward_size_bytes', 'reverse_pc',
    'reverse_bc', 'reverse_pl', 'reverse_piat', 'reverse_pps',
    'reverse_bps', 'reverse_pl_mean', 'reverse_piat_mean',
    'reverse_pps_mean', 'reverse_bps_mean', 'reverse_pl_var',
    'reverse_piat_var', 'reverse_pps_var', 'reverse_bps_var',
    'reverse_pl_q1', 'reverse_pl_q3', 'reverse_piat_q1', 'reverse_piat_q3',
    'reverse_pl_max', 'reverse_pl_min', 'reverse_piat_max',
    'reverse_piat_min', 'reverse_pps_max', 'reverse_pps_min',
    'reverse_bps_max', 'reverse_bps_min', 'reverse_duration',
    'reverse_size_packets', 'reverse_size_bytes']
    la = ['WWW', 'DNS', 'FTP', 'ICMP', 'P2P', 'VOIP']
    plt.figure(1, dpi=300)
    fig = tree.plot_tree(clf, filled=True, feature_names=fn, class_names=la)
    plt.title("Decision Tree")
    plt.show()
    labels = unique
    plt.figure(2, figsize=(5, 2))
    plt.title("Confusion Matrix", fontsize=10)
    cmnew = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sn.heatmap(cmnew, annot=True, cmap="YlGnBu", fmt=".2f", xticklabels=labels, yticklabels=labels)
    plt.show()


def main():
    dataset = pd.read_csv("dataset/SDN_traffic.csv")

    print(dataset.head())
    print(dataset.info())
    print(dataset.describe())
    print(dataset.duplicated())

    X = dataset[['tp_src', 'tp_dst', 'nw_proto',
    'forward_pc', 'forward_bc', 'forward_pl', 'forward_piat', 'forward_pps',
    'forward_bps', 'forward_pl_mean', 'forward_piat_mean',
    'forward_pps_mean', 'forward_bps_mean', 'forward_pl_var',
    'forward_piat_var', 'forward_pps_var', 'forward_bps_var',
    'forward_pl_q1', 'forward_pl_q3', 'forward_piat_q1', 'forward_piat_q3',
    'forward_pl_max', 'forward_pl_min', 'forward_piat_max',
    'forward_piat_min', 'forward_pps_max', 'forward_pps_min',
    'forward_bps_max', 'forward_bps_min', 'forward_duration',
    'forward_size_packets', 'forward_size_bytes', 'reverse_pc',
    'reverse_bc', 'reverse_pl', 'reverse_piat', 'reverse_pps',
    'reverse_bps', 'reverse_pl_mean', 'reverse_piat_mean',
    'reverse_pps_mean', 'reverse_bps_mean', 'reverse_pl_var',
    'reverse_piat_var', 'reverse_pps_var', 'reverse_bps_var',
    'reverse_pl_q1', 'reverse_pl_q3', 'reverse_piat_q1', 'reverse_piat_q3',
    'reverse_pl_max', 'reverse_pl_min', 'reverse_piat_max',
    'reverse_piat_min', 'reverse_pps_max', 'reverse_pps_min',
    'reverse_bps_max', 'reverse_bps_min', 'reverse_duration',
    'reverse_size_packets', 'reverse_size_bytes']]

    X = dataset_cleanup(X)

    Y = dataset[["category"]]
    Y = Y.to_numpy()
    Y = Y.ravel()

    labels, uniques = pd.factorize(Y)
    Y = labels
    Y = Y.ravel()

    X = stats.zscore(X)
    X = np.nan_to_num(X)

    # Decision Tree AI
    dt_clf, X_train, X_test, Y_train, Y_test = decision_tree_classifier(X, Y)
    cm = performance_evaluation(dt_clf, X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, uniques=uniques)
    find_the_10_most_important_features(dt_clf)
    plot_decision_tree_and_confusion_matrix(dt_clf, uniques, cm)

if __name__ == "__main__":
    main()