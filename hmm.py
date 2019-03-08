from nltk import FreqDist, WittenBellProbDist
from nltk.corpus import brown

class HMM:

        def __init__(self, corpus, tagset=""):
            self.corpus = corpus
            self.taggedSents, self.sents = self.getSentences(tagset)
            self.trainSize = int(len(self.taggedSents) * 0.95)
            self.testingSize = 500
            self.trainSents, self.testSents = self.splitTrainingTesting()
            self.words, self.tags = self.splitWordsTags()
            self.tagsDistribution = FreqDist(self.tags)
            self.uniqueTags = self.getUniqueTags()
            self.wordsDist, self.tagsDist = self.setProbDistributions()
            self.output()

        def setProbDistributions(self):
            tagMap={}
            wordMap={}

            for t in self.uniqueTags:
                tagList=[]
                wordList=[]
                for i in range(len(self.tags)):
                    if self.tags[i] == t:
                        wordList.append(self.words[i])
                        if i < (len(self.tags)-1):
                            tagList.append(self.tags[i+1])
                tagMap[t] = WittenBellProbDist(FreqDist(tagList), bins=1e5)
                wordMap[t] = WittenBellProbDist(FreqDist(wordList), bins=1e5)

            return wordMap, tagMap

        def getSentences(self, selected_tagset):
            taggedSents = self.corpus.tagged_sents(tagset=selected_tagset)
            sents = self.corpus.sents()
            return taggedSents, sents

        def splitTrainingTesting(self):
            train_sents = self.taggedSents[:self.trainSize]
            test_sents = self.sents[self.trainSize:self.trainSize + self.testingSize]
            return train_sents, test_sents

        def output(self):
            print("Training Data: "+str(self.trainSize)+" Sentences")
            print("Testing Data: "+str(self.testingSize)+" Sentences")
            print(self.testSents)

        def splitWordsTags(self):
            words = []
            tags = []
            startDelimeter = ["<s>"]
            endDelimeter = ["</s>"]
            for sentances in self.trainSents:
                words += startDelimeter + [w for (w, _) in sentances] + endDelimeter
                tags += startDelimeter + [t for (_, t) in sentances] + endDelimeter
            return words, tags

        def getUniqueTags(self):
            tagSet = set(self.tags)
            tagList = list(tagSet)
            return tagList


def main():
    corpus = brown
    tagset = "universal"
    hmm = HMM(corpus, tagset)


if __name__ == '__main__':
    main()


