from nltk import FreqDist, WittenBellProbDist
from nltk.corpus import brown
import operator

class HMM:

        def __init__(self, corpus, tagset=""):
            self.corpus = corpus
            self.taggedSents, self.sents = self.getSentences(tagset)
            self.trainSize = int(len(self.taggedSents) * 0.90)
            self.testingSize = 5
            self.trainSents, self.testSents = self.splitTrainingTesting()
            self.words, self.tags = self.splitWordsTags()
            self.check_sents = self.taggedSents[self.trainSize:self.trainSize + self.testingSize]

            self.testingWords, self.testingTags = self.splitWordsTagsTesting()
            self.testingWordsNoDelim, self.testingTagsNoDelim = self.splitWordsTagsTestingNoDelim()

            self.tagsDistribution = FreqDist(self.tags)
            self.uniqueTags, self.uniqueTagsNoDelim = self.getUniqueTags()
            self.wordsDist, self.tagsDist = self.setProbDistributions()
            #self.finalTags = self.setTags()
            self.finalTags = self.viterbi()
            self.output()

        def viterbi(self):
            finalTags = []
            probMatrix = []

            for w in self.testingWordsNoDelim:
                col = []
                for t in self.uniqueTagsNoDelim:

                    
                    if w == self.testingWordsNoDelim[0]:
                        pT = self.tagsDist['<s>'].prob(t)
                        pW = self.wordsDist[t].prob(w)
                        col.append([pW*pT, "q0"])
                        print("col: ", col)
                    else:
                        tagMap = {}
                        for pp in range(0, len(self.uniqueTagsNoDelim) - 1):
                            pT = self.tagsDist[self.uniqueTagsNoDelim[pp]].prob(t)
                            pW = self.wordsDist[t].prob(w)
                            print("pm: ", probMatrix)
                            print("pp: ", pp)
                            tagMap[self.uniqueTagsNoDelim[pp]] = pT * pW * probMatrix[-1][pp][0]

                        prevBestTag = max(tagMap.items(), key=operator.itemgetter(1))[0]
                        value = max(tagMap.items(), key=operator.itemgetter(1))[1]
                        col.append([value, prevBestTag])
                print("col2: ", col)
                probMatrix.append(col)

            finalTags = self.getTagsFromMatrix(probMatrix)
            return finalTags

        def getTagsFromMatrix(self, matrix):
            finalTags=[]
            pointer = ""
            pointerID=0
            for i in range(len(self.testingWords), 1):
                max=0
                maxID=0
                if i == range(len(self.testingWords)):
                    for j in range(0, len(self.uniqueTags) - 1):
                        if matrix[-i][j][0]>max:
                            max=matrix[-i][j][0]
                            maxID = j
                    finalTags.append(self.uniqueTags[maxID])
                    pointer = matrix[-i][maxID][1]
                else:
                    pointerID=self.uniqueTags.index(pointer)
                    finalTags.append(self.uniqueTags[pointerID])
                    pointer = matrix[-i][pointerID][1]

            finalTags.reverse()
            print("finalTags: ")

            print(finalTags)
            return finalTags

        def setTags(self):
            finalTags = []

            for i in self.testingWords:
                tagMap = {}
                if i == "<s>":
                    finalTags.append("<s>")
                elif i == "</s>":
                    finalTags.append("</s>")
                else:

                    for tag in self.uniqueTags:
                        pT = self.tagsDist[finalTags[-1]].prob(tag)
                        pW = self.wordsDist[tag].prob(i)
                        tagMap[tag] = pT*pW
                    finalTags.append(max(tagMap.items(), key=operator.itemgetter(1))[0])

            return finalTags

        def setProbDistributions(self):
            tagMap = {}
            wordMap = {}

            for t in self.uniqueTags:
                tagList=[]
                wordList=[]
                for i in range(len(self.tags)-1):
                    if self.tags[i] == t:
                        wordList.append(self.words[i])
                        if i < (len(self.tags)-2):
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
            print(self.testingWordsNoDelim)
            print("--------------------------")
            print(self.finalTags)
            print("--------------------------")
            # commonList = [i for i, j in zip(self.testingTags, self.finalTags) if i == j]
            # percent = (len(commonList)/len(self.testingTags))*100
            # print(str(percent)+"% Accuracy")

        def splitWordsTags(self):
            words = []
            tags = []
            startDelimeter = ["<s>"]
            endDelimeter = ["</s>"]

            for s in self.trainSents:
                words += startDelimeter + [w for (w, _) in s] + endDelimeter
                tags += startDelimeter + [t for (_, t) in s] + endDelimeter
            return words, tags

        def splitWordsTagsTesting(self):
            words = []
            tags = []
            startDelimeter = ["<s>"]
            endDelimeter = ["</s>"]

            for s in self.check_sents:
                words += startDelimeter + [w for (w, _) in s] + endDelimeter
                tags += startDelimeter + [t for (_, t) in s] + endDelimeter
            return words, tags

        def splitWordsTagsTestingNoDelim(self):
            words = []
            tags = []


            for s in self.check_sents:
                words += [w for (w, _) in s]
                tags += [t for (_, t) in s]
            return words, tags

        def getUniqueTags(self):
            tagSet = set(self.tags)
            tagList = list(tagSet)

            noDelim = tagList.copy()

            noDelim.remove('<s>')
            noDelim.remove('</s>')

            return tagList, noDelim


def main():
    corpus = brown
    tagset = "universal"
    hmm = HMM(corpus, tagset)


if __name__ == '__main__':
    main()


