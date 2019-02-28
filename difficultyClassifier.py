# import csv
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

def get_encoder(filepath, isY):
    with open(filepath) as tsv:
        data = tsv.readlines()
    out = []
    for line in data:
        line = line.strip()
        if isY:
            line = line.split()
            label = line[-1]
            out.append(label)
        else:
            line = line.split('\t')
            out.extend(line)
    encoder = preprocessing.LabelEncoder()
    encoder.fit(out)
    # transformed_train_vocab = encoder.transform(encoder.classes_)  # do we need this?
    return encoder


def get_data(filepath, encoder, isY):
    with open(filepath) as tsv:
        data = tsv.readlines()
    out = []
    for line in data:
        line = line.strip()
        if isY:
            line = line.split()
            label = [line[-1]]
            label = encoder.transform(label)
            out.extend(label)
        else:
            line = line.split('\t')
            vectored_line = encoder.transform(line)
            out.append(list(vectored_line))
    return out


def difficulty_classifier():
    dat = 'difficultyN'

    # getting x_train
    isY = False
    trainingFile = 'train_data/{0}/{0}.tsv'.format(dat)
    enc = get_encoder(trainingFile, isY)
    x_train = get_data(trainingFile, enc, isY)

    # getting y_train
    isY = True
    trainingLabels = 'train_data/{0}/{0}Answers.train'.format(dat)
    enc = get_encoder(trainingLabels, isY)
    y_train = get_data(trainingLabels, enc, isY)

    # getting x_test
    isY = False
    testingFile = 'test_data/{0}/{0}_test.tsv'.format(dat)
    enc = get_encoder(testingFile, isY)
    x_test = get_data(testingFile, enc, isY)

    # getting y_test
    isY = True
    testingLabels = 'test_data/{0}/{0}Answers.test'.format(dat)
    enc = get_encoder(testingLabels, isY)
    y_test = get_data(testingLabels, enc, isY)

    # print(x_train)
    # print(y_train)
    # print(x_test)
    # print(y_test)

    # interpreting encoded labels
    print('og  labels: ', enc.classes_)
    print('num labels: ', enc.transform(enc.classes_))

    # Multinomial NB classifier
    # clf = MultinomialNB(alpha=3.0)
    # clf.fit(x_train, y_train)
    # y_pred = clf.predict(x_test)

    # KNN classifier
    # clf = KNeighborsClassifier(n_neighbors=19)
    # clf.fit(x_train, y_train)
    # y_pred = clf.predict(x_test)

    # svm classifier
    # clf = svm.SVC()
    clf = svm.SVC(kernel='rbf', gamma='scale', class_weight='balanced', C=1, decision_function_shape='ovo')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print(set(y_pred))
    print('Did not predict: ', set(y_test) - set(y_pred))

    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    print('Precision: ', metrics.precision_score(y_test, y_pred, average='weighted'))
    print('Recall: ', metrics.recall_score(y_test, y_pred, average='weighted'))


def main():
    difficulty_classifier()


if __name__ == main():
    main()
