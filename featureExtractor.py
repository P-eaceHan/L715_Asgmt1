'''
extracting features from training data:
[w-2   w-1   w+1   w+2   w-2%w-1   w-1%w+1   w+1%w+2   class]
'''
import regex as re
import xml.etree.ElementTree as ET


def pluralize(word):
    if word[-1] == 'y':
        word = word[-1]
        word += 'ies'
    else:
        word += 's'


def extractFeatures(word, pos):
    dat = word + pos
    inputFile = 'train_data/{0}/{0}.train'.format(dat)
    root = ET.parse(inputFile).getroot()
    out = open('train_data/{0}/{0}.tsv'.format(dat), 'w')
    out2 = open('train_data/{0}/{0}_withClass.tsv'.format(dat), 'w')

    for instance in root.findall('instance'):
        answer = instance.find('answer').attrib
        sense = answer.get('senseid')
        context = ET.tostring(instance.find('context'))
        context = context.decode('UTF-8')
        print(context)
        # 2 words before and after word
        # punctuation = word
        pattern2 = "(?:\S+\s+){,2}<head>.*<\/head>(?:\s+\S+){,2}"
        res2 = re.findall(pattern2, context)
        justWords = []
        if res2:
            [justWords.append(x) for x in res2]
            print(justWords)
            if len(justWords) > 1:
                break  # to catch duplicate contexts in training data
            # [out.write(x + '\n') for x in res2]
            contextFragment = justWords[0]
            vectorList = contextFragment.split()
            if len(vectorList) < 5:  # to detect potential errors
                print(vectorList)
                break
            vectorList.pop(2)  # get rid of the target word
            finalVector = '\t'.join(vectorList)
            for i in range(len(vectorList) - 1):  # combining features
                finalVector += '\t' + vectorList[i] + '%' + vectorList[i + 1]
            print(contextFragment)
            print(finalVector)
            # finalVector += '\n'
            out.write(finalVector + '\n')
            finalVector += '\t' + sense
            out2.write(finalVector + '\n')
    out.close()
    out2.close()

