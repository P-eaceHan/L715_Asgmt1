# import csv
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report
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
            out.extend(label)
        else:
            line = line.split('\t')
            out.append(line)
    out = encoder.transform(out)
    return out


def arm_classifier():
    dat = 'armN'

    # getting x_train
    isY = False
    trainingFile = 'train_data/{0}/{0}.tsv'.format(dat)
    x_train, enc_data = encode(trainingFile, isY)

    # getting y_train
    isY = True
    trainingLabels = 'train_data/{0}/{0}Answers.train'.format(dat)
    y_train, enc_label = encode(trainingLabels, isY)

    # getting x_test
    isY = False
    testingFile = 'test_data/{0}/{0}_test.tsv'.format(dat)
    x_test = encode_test(testingFile, enc_data, isY)

    # getting y_test
    isY = True
    testingLabels = 'test_data/{0}/{0}Answers.test'.format(dat)
    y_test = encode_test(testingLabels, enc_label, isY)

    # print('x_train: \n', x_train)
    # print('y_train: \n', y_train)
    # print(x_test)
    # print(y_test)

    # interpreting encoded labels
    print('labels: ')
    [print(x) for x in enc_label.classes_]
    # print('num labels: ', enc_label.transform(enc_label.classes_))
    print('===============================================')

    # rbf kernel
    print('rbf kernel results')
    # clf= svm.SVC(kernel='rbf')
    clf = svm.SVC(kernel='rbf', gamma='scale', C=4, class_weight='balanced', decision_function_shape='ovo')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Did not predict: ', set(y_test) - set(y_pred))

    print(classification_report(y_test, y_pred))
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    print('===============================================')

    # linear kernel
    print('linear kernel results')
    # clf= svm.SVC(kernel='linear)
    clf = svm.SVC(kernel='linear', gamma='scale', C=4, class_weight='balanced', decision_function_shape='ovo')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Did not predict: ', set(y_test) - set(y_pred))

    print(classification_report(y_test, y_pred))
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    print('===============================================')

    # sigmoid? poly? kernel
    print('sigmoid kernel results')
    # clf= svm.SVC(kernel='poly)
    clf = svm.SVC(kernel='sigmoid', gamma='scale', C=4, class_weight='balanced', decision_function_shape='ovo')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Did not predict: ', set(y_test) - set(y_pred))

    print(classification_report(y_test, y_pred))
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    print('===============================================')

    # poly kernel
    print('poly kernel results')
    # clf= svm.SVC(kernel='poly')
    clf = svm.SVC(kernel='poly', gamma='scale', C=4, class_weight='balanced', decision_function_shape='ovo')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Did not predict: ', set(y_test) - set(y_pred))

    print(classification_report(y_test, y_pred))
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    print('===============================================')


def main():
    arm_classifier()


if __name__ == main():
    main()

