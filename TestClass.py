
from nltk.corpus import brown
import operator

sents = brown.tagged_sents(tagset='universal')
first = sents[0]
tagMap = {'a': 1000, 'b': 3000, 'c': 100}
print(max(tagMap.items(), key=operator.itemgetter(1))[0])
print(max(tagMap.items(),key=operator.itemgetter(1))[1])
# probMatrix = [[[0] * 2] * 3] * 4
probMatrix=[]


col=[]
col.append([232, "nn"])
col.append([222, "v"])
col.append([23342, "adj"])
col.append([232, "nn"])
col.append([222, "v"])
col.append([23342, "adj"])
probMatrix.append(col)
probMatrix.append(col)
probMatrix.append(col)
fal=[]
fal.append([100, "n22n"])
fal.append([222, "v"])
fal.append([23342, "adj"])
fal.append([232, "nn"])
fal.append([222, "vaa"])
fal.append([23342, "adj"])
probMatrix.append(fal)

print(probMatrix)

print(probMatrix[-1][4][1])



test=[]
test.append([2,3,4,5])
test.append([1,2,3,4])
print(test)
words = [w for (w,_) in first]

tags = [t for (_,t) in first]

# print("----------")
# print("Words")
# print(words)
# print("----------")
#
# print("Sents")
# print(sents)
# print("----------")
#
# print("Tags")
# print(tags)
# print("----------")
#
# def show_sent(sent):
#     print(sent)
#
# for sent in sents[0:10]:
#     show_sent(sent)