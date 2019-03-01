# import csv
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


'''
:param filepath: file to the tsv file of features
:param isY: flag telling if this is a label file
:return: numericized features and encoder
'''
def encode(filepath, isY):
    with open(filepath) as tsv:
        data = tsv.readlines()
    out = []
    for line in data:
        line = line.strip()
        if isY:
            line = line.split()
            label = line[-1]
            out.append([label])
        else:
            line = line.split('\t')
            out.append(line)
    if isY:
        enc = preprocessing.LabelEncoder()
    else:
        # enc = preprocessing.OneHotEncoder(n_values=shape)
        enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
    out = enc.fit_transform(out)
    return out, enc


'''
:param filepath: file to the tsv file of features
:param encoder: the encoder to encode data in filepath
:param isY: flag telling if this is a label file
:return: numericized test features
'''
def encode_test(filepath, encoder, isY):
    with open(filepath) as tsv:
        data = tsv.readlines()
    out = []
    for line in data:
        line = line.strip()
        if isY:
            line = line.split()
            label = [line[-1]]
            # label = encoder.transform(label)
            out.extend(label)
        else:
            line = line.split('\t')
            # vectored_line = encoder.transform(line)
            out.append(line)
    out = encoder.transform(out)
    return out


def arm_classifier():
    dat = 'armN'

    # getting x_train
    isY = False
    trainingFile = 'train_data/{0}/{0}.tsv'.format(dat)
    x_train, enc_data = encode(trainingFile, isY)
    # x_train = get_data(trainingFile, enc, isY)
    # x_train = list(preprocessing.scale(x_train))

    # getting y_train
    isY = True
    trainingLabels = 'train_data/{0}/{0}Answers.train'.format(dat)
    y_train, enc_label = encode(trainingLabels, isY)
    # y_train = get_data(trainingLabels, enc, isY)
    # y_train = list(preprocessing.scale(y_train))

    # getting x_test
    isY = False
    testingFile = 'test_data/{0}/{0}_test.tsv'.format(dat)
    x_test = encode_test(testingFile, enc_data, isY)
    # x_test = get_data(testingFile, enc, isY)
    # x_test = list(preprocessing.scale(x_test))

    # getting y_test
    isY = True
    testingLabels = 'test_data/{0}/{0}Answers.test'.format(dat)
    y_test = encode_test(testingLabels, enc_label, isY)
    # y_test = get_data(testingLabels, enc, isY)
    # y_test = list(preprocessing.scale(y_test))

    # print('x_train: \n', x_train)
    # print('y_train: \n', y_train)
    # print(x_test)
    # print(y_test)

    # interpreting encoded labels
    # print('og  labels: ', enc.categories_)
    # print('num labels: ', enc.transform(enc.categories_))

    # svm SVC classifier
    # clf= svm.SVC()
    # clf = svm.SVC(kernel='sigmoid', gamma='scale', C=3, class_weight='balanced', decision_function_shape='ovo')
    clf = svm.SVC(kernel='rbf', gamma='scale', C=4)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # svm Linear SVC classifier
    # clf = svm.LinearSVC()
    # clf = svm.LinearSVC(loss='hinge', max_iter=5000000)
    # clf.fit(x_train, y_train)
    # y_pred = clf.predict(x_test)

    # svm Nu SVC classifier
    # clf = svm.NuSVC(nu=0.9999)
    # clf.fit(x_train, y_train)
    # y_pred = clf.predict(x_test)

    print(set(y_pred))
    print('Did not predict: ', set(y_test) - set(y_pred))

    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    print('Precision: ', metrics.precision_score(y_test, y_pred, average='weighted'))
    print('Recall: ', metrics.recall_score(y_test, y_pred, average='weighted'))


def main():
    arm_classifier()


if __name__ == main():
    main()

