'''
extracting features from training data:
[w-2   w-1   w+1   w+2   w-2%w-1   w-1%w+1   w+1%w+2   class]
'''
import regex as re
import xml.etree.ElementTree as ET

inputFile = 'train_data/armN.train'

root = ET.parse(inputFile).getroot()
out = open('armN.tsv', 'w')
out2 = open('armN_withClass.tsv', 'w')

for instance in root.findall('instance'):
    answer = instance.find('answer').attrib
    inst = answer.get('instance')
    sense = answer.get('senseid')
    context = ET.tostring(instance.find('context'))
    context = context.decode('UTF-8')
    print(context)
    # 2 words before and after word
    # punctuation = word
    pattern2 = "(?:\S+\s+){,2}<head>arms?<\/head>(?:\s+\S+){,2}"
    res2 = re.findall(pattern2, context)
    justWords = []
    if res2:
        [justWords.append(x) for x in res2]
        # print(justWords)
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
'''
# 2 words before and after word
# punctuation not count as word
pattern = "(?:\w+\W+){,2}<head>arms?</head>(?:\W+\w+){,2}"

# 2 words before and after word
# punctuation = word
pattern2 = "(?:\S+\s+){,2}<head>arms?<\/head>(?:\s+\S+){,2}"

out = open("armNVector.train", 'w')
out2 = open('arnmN.tsv', 'w')
justWords = []

# what to do with instance 00078497? it's two instances of arms in one context
with open(inputFile) as file:
    for line in file:
        res = re.findall(pattern, line)
        res2 = re.findall(pattern2, line)
        # if res:
        #     print(res)
        if res2:
            # print(res2)
            [justWords.append(x) for x in res2]
            # [out.write(x + '\n') for x in res2]

    for tempVec in justWords:
        tempVecList = tempVec.split()
        if len(tempVecList) < 5:  # to detect potential errors
            out.write(' '.join(tempVecList))
        tempVecList.pop(2)
        tempVec = '\t'.join(tempVecList)
        for i in range(len(tempVecList)-1):
            tempVec += '\t' + tempVecList[i] + '%' + tempVecList[i+1]
        print(tempVec)
        tempVec += '\n'
        out2.write(tempVec)



# print(justWords)

'''

