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

            #self.output()

        def getSentences(self, selected_tagset):
            tagged_sents = self.corpus.tagged_sents(tagset=selected_tagset)
            sents = self.corpus.sents()
            return tagged_sents, sents

        def splitTrainingTesting(self):
            train_sents = self.taggedSents[:self.trainSize]
            test_sents = self.sents[self.trainSize:self.trainSize + self.testingSize]
            return train_sents, test_sents

        def output(self):
            print("Training Data: "+str(self.trainSize)+" Sentences")
            print("Testing Data: "+str(self.testingSize)+" Sentences")
            print(self.words)

        def splitWordsTags(self):
            words = []
            tags = []
            startDelimeter = ["<s>"]
            endDelimeter = ["</s>"]
            for sentances in self.trainSents:
                words += startDelimeter + [w for (w, _) in sentances] + endDelimeter
                tags += startDelimeter + [t for (_, t) in sentances] + endDelimeter
            return words, tags



def main():
    corpus = brown
    tagset = "universal"
    hmm = HMM(corpus, tagset)


if __name__ == '__main__':
    main()


