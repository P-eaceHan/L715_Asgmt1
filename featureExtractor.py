'''
extracting features from training data:
[w-2   w-1   w+1   w+2   w-2%w-1   w-1%w+1   w+1%w+2   class]
'''
import xml.etree.ElementTree as ET
import re
inputFile = 'train_data/armN.train'

# with open(inputFile) as file:
#     for line in file:
#         print(line)

# e = ET.parse(inputFile).getroot()
# # print(e)
#
# # for atype in e:
# #     print(atype.tag, atype.attrib)
# out = []
#
# for instance in e.findall('instance'):
#     print(instance.attrib)
#     # out.append(instance.get('answer'))
#     # print(instance.find('answer'))
#     ans = instance.find('answer').attrib.get('senseid')
#     print(instance.find('context').text)
#     # print(ans)

# regex approach
# pattern = "(.+?)<head>(.+?)</head>(.+?)"
pattern = "(?:\w+\W+){,2}<head>arms?</head>(?:\W+\w+){,2}"
# pattern2 = "<head>(?:\W+\w+){,2}"
# pattern = "(?P<w-2>"
with open(inputFile) as file:
    for line in file:
        res = re.findall(pattern, line)
        # res2 = re.findall(pattern2, line)
        if res:
            print(res)
        # if res2:
        #     print(res2)

