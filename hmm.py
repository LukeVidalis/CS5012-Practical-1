
from nltk.corpus import brown

sents = brown.tagged_sents(tagset='universal')
first = sents[0]

words = [w for (w,_) in first]

tags = [t for (_,t) in first]

print("----------")
print("Words")
print(words)
print("----------")

print("First")
print(first)
print("----------")

print("Tags")
print(tags)
print("----------")

def show_sent(sent):
    print(sent)

for sent in sents[0:10]:
    show_sent(sent)