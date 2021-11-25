from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def lgr_classifier(counts_train, lab_train):
    """
    Logistic Regression Model
    :param counts_train: count the number of times each term appears in a document and transform each doc into a count vector
    :param lab_train: ratings for training data
    :return: gridsearch
    """
    clf = LogisticRegression(solver='liblinear')
    LGR_grid = [{'penalty': ['l1', 'l2'], 'C': [0.5, 1, 1.5, 2, 3, 5, 10]}]
    gridsearchLGR = GridSearchCV(clf, LGR_grid, cv=5)
    return gridsearchLGR.fit(counts_train, lab_train)


def rf_classifier(counts_train, lab_train):
    """
    Random Forest Model
    :param counts_train: count the number of times each term appears in a document and transform each doc into a count vector
    :param lab_train: ratings for training data
    :return: gridsearch
    """
    clf = RandomForestClassifier(random_state=150, max_depth=600, min_samples_split=160)
    RF_grid = [{'n_estimators': [50, 100, 150, 200, 300, 500, 800, 1200, 1600, 2100],
                'criterion': ['gini', 'entropy'], 'max_features': ['auto', 'sqrt', 'log2']}]
    gridsearchRF = GridSearchCV(clf, RF_grid, cv=5)
    return gridsearchRF.fit(counts_train, lab_train)


def knn_classifier(counts_train, lab_train):
    """
    K nearest neighbors Model
    :param counts_train: count the number of times each term appears in a document and transform each doc into a count vector
    :param lab_train: ratings for training data
    :return: gridsearch
    """
    clf = KNeighborsClassifier()
    KNN_grid = [{'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17],
                 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'brute']}]
    gridsearchKNN = GridSearchCV(clf, KNN_grid, cv=5)
    return gridsearchKNN.fit(counts_train, lab_train)


def dt_classifier(counts_train, lab_train):
    """
    Decision Tree Model
    :param counts_train: count the number of times each term appears in a document and transform each doc into a count vector
    :param lab_train: ratings for training data
    :return: gridsearch
    """
    clf = DecisionTreeClassifier()
    DT_grid = [{'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random']}]
    gridsearchDT = GridSearchCV(clf, DT_grid, cv=5)
    return gridsearchDT.fit(counts_train, lab_train)


def nb_classifier(counts_train, lab_train):
    """
    Native Bayes Model
    :param counts_train: count the number of times each term appears in a document and transform each doc into a count vector
    :param lab_train: ratings for training data
    :return: gridsearch
    """
    clf = MultinomialNB()
    NB_grid = [{'alpha': [0.0001, 0.001, 0.01, 0.1, 0.8, 1, 10], 'fit_prior': [True, False]}]
    gridsearchNB = GridSearchCV(clf, NB_grid, cv=5)
    return gridsearchNB.fit(counts_train, lab_train)


def svm_classifier(counts_train, lab_train):
    """
    Support Vector Machine Model
    :param counts_train: count the number of times each term appears in a document and transform each doc into a count vector
    :param lab_train: ratings for training data
    :return: gridsearch
    """
    clf = svm.SVC()
    SVM_grid = [{'C': [0.0001, 0.001, 0.01, 0.1, 0.8, 1, 10], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}]
    gridsearchSVM = GridSearchCV(clf, SVM_grid, cv=5)
    return gridsearchSVM.fit(counts_train, lab_train)
