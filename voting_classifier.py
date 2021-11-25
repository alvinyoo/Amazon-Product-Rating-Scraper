from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier


def vt(predictors, counts_test, counts_train, lab_train):
    """
    Voting Classifier with different classification algorithms
    :param predictors: different classification algorithms
    :param counts_test: the transformed testing data
    :return: the accuracy score
    """
    VT = VotingClassifier(predictors)
    VT.fit(counts_train, lab_train)
    predicted = VT.predict(counts_test)
    return accuracy_score(predicted, lab_test)
