from sklearn import preprocessing

'''
:param filepath: file to the tsv file of features
:param isY: flag telling if this is a label file
:return: numericized features and encoder
'''


def encode_data(filepath):
    with open(filepath) as tsv:
        data = tsv.readlines()
    out = []
    for line in data:
        line = line.strip()
        line = line.split('\t')
        out.append(line)
    enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
    out = enc.fit_transform(out)
    return out, enc


'''
:param filepath: file to the tsv file of features
:param encoder: the encoder to encode data in filepath
:param isY: flag telling if this is a label file
:return: numericized test features
'''


def encode_test(filepath, encoder):
    with open(filepath) as tsv:
        data = tsv.readlines()
    out = []
    for line in data:
        line = line.strip()
        line = line.split('\t')
        out.append(line)
    out = encoder.transform(out)
    return out


'''
:param filepath: file to the label file
:param d: the "label encoder" is a dictionary
:return: labels: the transformed (numercized) labels from file
:return: d: the potentially updated "encoder"
'''


def label_encoder(filepath, d=dict()):
    with open(filepath) as tsv:
        data = tsv.readlines()
    labels = []
    for line in data:
        line = line.strip()
        line = line.split()
        label = line[-1]
        if label not in d.keys():
            d[label] = len(d)
        labels.append(d[label])
    return labels, d

