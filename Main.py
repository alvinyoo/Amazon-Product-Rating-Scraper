from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from rate_models import lgr_classifier
from rate_models import rf_classifier
from rate_models import knn_classifier
from rate_models import dt_classifier
from rate_models import nb_classifier
from rate_models import svm_classifier
from nltk.corpus import stopwords
from Data_Writer import loadData
from voting_classifier import vt
from Data_Writer import Filter
from sraper import scrape
from sklearn import svm
from time import time


def app(link1, link2):

    all_start = time()

    print('start scraping...')
    start = time()
    scrape(link1, name='train')
    scrape(link2, name='test')
    print('scraping finished')
    print(f"scrape time: {time() - start}")

    start = time()
    print('start training...')

    rev_train, lab_train = loadData('/train.csv')
    rev_test, lab_test = loadData('/test.csv')

    # remove the noise
    rev_train = Filter(rev_train)
    rev_test = Filter(rev_test)

    # Build a counter based on the training dataset
    counter = CountVectorizer(stop_words=stopwords.words('english'))
    counter.fit(rev_train)

    # count the number of times each term appears in a document and transform each doc into a count vector
    counts_train = counter.transform(rev_train)  # transform the training data
    counts_test = counter.transform(rev_test)  # transform the testing data

    # fit the models
    lgr_time = time()
    lgr_classifier(counts_train, lab_train)
    print(f"Logistic regression finished, run time: {time() - lgr_time}")

    rf_time = time()
    rf_classifier(counts_train, lab_train)
    print(f"Random Forest finished, run time: {time() - rf_time}")

    knn_time = time()
    knn_classifier(counts_train, lab_train)
    print(f"KNN finished, run time: {time() - knn_time}")

    dt_time = time()
    dt_classifier(counts_train, lab_train)
    print(f"Decision tree finished, run time: {time() - dt_time}")

    nb_time = time()
    nb_classifier(counts_train, lab_train)
    print(f"Naive Bayes finished, run time: {time() - nb_time}")

    svm_time = time()
    svm_classifier(counts_train, lab_train)
    print(f"SVM finished, run time: {time() - svm_time}")

    predictors = [('lreg', LogisticRegression()), ('rf', RandomForestClassifier()), ('knn', KNeighborsClassifier()),
                  ('dt', DecisionTreeClassifier()), ('nb', MultinomialNB()), ('svm', svm.SVC())]

    score = vt(predictors, counts_test, counts_train, lab_train)

    print(f"all finished, run time: {time() - start}")  # 949.3523089885712
    print(f"accuracy: {score}")  # 0.8833333333333333

    print(f"all finished, run time: {time() - all_start}")


if __name__ == "__main__":
    url1 = 'https://www.amazon.com/Sennheiser-Momentum-Cancelling-Headphones-Functionality/dp/B07VW98ZKG'
    url2 = 'https://www.amazon.com/Sennheiser-Momentum-Cancelling-Headphones-Functionality/dp/B07VW98ZKG'

    app(url1, url2)
