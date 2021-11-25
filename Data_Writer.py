import pandas as pd
import nltk
import re


def loadData(file):
    """
    read the reviews and their polarities from a given file
    while rate is more than 3 in 5, it would be a good review
    :param file: the path of review file
    :return: reviews and labels
    """
    reviews, labels = [], []
    f = pd.read_csv(file, header=None)
    for i in range(len(f)):
        reviews.append(f[0][i].replace('\n', ''))
        # f[1][i] --> rating
        rate = f[1][i].strip()
        rate = int(rate[0])
        if rate >= 3:
            labels.append(1)
        else:
            labels.append(0)
    return reviews, labels


def Filter(reviews):
    """
    decrease the dimension of dataset
    :param reviews: reviews from dataset
    :return: reviews without stop words
    """
    ans = []
    for review in reviews:
        temp = []
        # review = re.sub(r'[^\w\s]', ' ', review)
        review = re.sub('[^a-z]', ' ', review)  # replace all non-letter characters

        ps = nltk.stem.porter.PorterStemmer()

        new_review = []
        for word in review.split():
            word = ps.stem(word)
            if word == '':
                continue  # ignore empty words and stopwords
            else:
                new_review.append(word)
        temp.append(' '.join(new_review))
        ans += temp
    return ans
