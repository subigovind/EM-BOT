import csv
import numpy as np
import sys
Training_data = list(csv.reader(open('editeddataset2.csv')))
#print(Training_data)
class NaiveBayes:
    #4
    def __init__(self,data,vocab):
        self._vocab=vocab
        labelArray=[]
        for i in range(1,len(data)):labelArray.append(data[i][1])
        self._label=np.array(labelArray)
        docArray=[]
        #6
        for i in range(1,len(data)):
            docArray.append(self.map_doc_to_vocab(data[i][0].split()))
        self._doc=np.array(docArray)
        self.calc_prior_prob().calc_cond_probs()
    def calc_prior_prob(self):
        sum=0
        for i in self._label:
            if ("-".__eq__(i)): sum+=1;
        self._priorProb=sum/len(self._label)
        return self
    def calc_cond_probs(self):
        pProbNum=np.ones(len(self._doc[0])); nProbNum=np.ones(len(self._doc[0]))
        pProbDenom=len(self._vocab); nProbDenom=len(self._vocab)
        for i in range(len(self._doc)):
            if "-".__eq__(self._label[i]):
                nProbNum+=self._doc[i]
                nProbDenom += sum(self._doc[i])
            else:
                pProbNum +=self._doc[i]
                pProbDenom +=sum(self._doc[i])
        self._negProb=np.log(nProbNum/nProbDenom)
        self._posProb=np.log(pProbNum/pProbDenom)
        return self
    def classify(self,doc):
        sentiment="negative"
        nLogSums=doc @ self._negProb +np.log(self._priorProb)
        pLogSums=doc @ self._posProb+np.log(1.0-self._priorProb)
        if pLogSums > nLogSums: sentiment = "positive"
        return "Thanks for your "+sentiment+" review"  
    #5        
    def map_doc_to_vocab(self,doc):
        mappedDoc=[0]*len(self._vocab)
        for d in doc:
            counter=0
            for v in self._vocab:
                if(d.__eq__(v)):mappedDoc[counter]+=1
                counter+=1
        return mappedDoc

def handle_command_line(nb):
    entry=sys.argv[1]
    print(nb.classify(np.array(nb.map_doc_to_vocab(entry.lower().split()))))
#1
def prepare_data():
    data=[]
    for i in range(0,len(Training_data)):
        data.append([Training_data[i][0].lower(),Training_data[i][1]])
    print(data)
    return data
#2
def prepare_vocab(data):
    vocabSet=set([])
    for i in range(1,len(data)):
        for word in data[i][0].split(): vocabSet.add(word)
    return list(vocabSet)
#3
data=prepare_data()

handle_command_line(NaiveBayes(data,prepare_vocab(data)))
