# import csv
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics


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


def arm_classifier():
    dat = 'armN'

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

    # svm classifier
    clf = svm.SVC(kernel='linear', decision_function_shape='ovo')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    print('Precision: ', metrics.precision_score(y_test, y_pred, average='weighted'))
    print('Recall: ', metrics.recall_score(y_test, y_pred, average='weighted'))

