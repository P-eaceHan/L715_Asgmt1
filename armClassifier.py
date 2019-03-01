# import csv
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report
import featureEncoder as fe


def arm_classifier():
    dat = 'armN'

    # getting x_train
    isY = False
    trainingFile = 'train_data/{0}/{0}.tsv'.format(dat)
    x_train, enc_data = fe.encode(trainingFile, isY)

    # getting y_train
    isY = True
    trainingLabels = 'train_data/{0}/{0}Answers.train'.format(dat)
    # y_train, enc_label = fe.encode(trainingLabels, isY)
    y_train, enc_label = fe.label_encoder(trainingLabels)

    # getting x_test
    isY = False
    testingFile = 'test_data/{0}/{0}_test.tsv'.format(dat)
    x_test = fe.encode_test(testingFile, enc_data, isY)

    # getting y_test
    isY = True
    testingLabels = 'test_data/{0}/{0}Answers.test'.format(dat)
    # y_test = fe.encode_test(testingLabels, enc_label, isY)
    y_test, enc_label = fe.label_encoder(testingLabels, enc_label)

    # print('x_train: \n', x_train)
    # print('y_train: \n', y_train)
    # print(x_test)
    # print(y_test)

    # interpreting encoded labels
    print('labels: ')
    [print(v, ' - ', k) for k, v in enc_label.items()]

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

