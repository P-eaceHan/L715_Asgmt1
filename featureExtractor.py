'''
extracting features from training data:
[w-2   w-1   w+1   w+2   w-2%w-1   w-1%w+1   w+1%w+2   class]
'''
import regex as re
import xml.etree.ElementTree as ET

inputFile = 'train_data/armN.train'

# 2 words before and after word
# punctuation not count as word
pattern = "(?:\w+\W+){,2}<head>arms?</head>(?:\W+\w+){,2}"

# 2 words before and after word
# punctuation = word
pattern2 = "(?:\S+\s+){,2}<head>arms?<\/head>(?:\s+\S+){,2}"

out = open("armNVector.train", 'w')
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
        tempVec = ' '.join(tempVecList)
        for i in range(len(tempVecList)-1):
            tempVec += ' ' + tempVecList[i] + '%' + tempVecList[i+1]
        print(tempVec)


print(justWords)

