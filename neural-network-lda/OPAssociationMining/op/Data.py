import nltk
import codecs
import os
import sys
import Utility
import Tokenizer
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np

class Data:
    data_dir=""
    all_data =[]
    patterns = []
    all_features = []
    all_unigram_features = {}
    all_bigram_features = {}
    features_by_patterns = {}
    unigram_N_by_patterns={}
    bigram_N_by_patterns={}
    words_documents={}
    document_frequency={}
    all_unigram_N={}
    all_bigram_N={}
    unigram_features_by_patterns = {}
    bigram_features_by_patterns = {}
    all_text = "";
    totUni=0
    totBi=0
    stemmer = SnowballStemmer('english')
    s_words = stopwords.words('english')
    s_words.extend(
       ['...', '.', '.✥', '✥', '\'', '\"','\",','.\"', ',', ':', '.**','(\"','\")', '\").','?\"','(', ')', 'e', '.,', '-', '....', ';', '[', ']', '—', '.)',
        'therefor', '’', '([', '])','--','/','],','ff','e','g','bibref','60ksloc','foote2000'])

    def __init__(self,datadir):
        self.data_dir=os.getcwd() + "/"+datadir+"/"
        symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
        try:
            for file in os.listdir(self.data_dir):
                with open(self.data_dir+ file, 'r', encoding="utf-8") as f:
                    temp_text = ""
                    temp_unigram_N=0
                    temp_bigram_N=0
                    for line in f:
                        for i in range(len(symbols)):
                                line = np.str.replace(line, symbols[i], ' ')
                                line = np.str.replace(line, "  ", " ")
                        line= np.str.replace(line, "'", "")
                        line = line.strip()
                        if not line.isspace() and not line.startswith('\n'):
                            temp_text += line;
                            self.all_text += line;

                    filtered_text = [self.stemmer.stem(w) for w in nltk.wordpunct_tokenize(temp_text) if not w.lower() in self.s_words and len(w)>2]

                    # self.unigram_N_by_patterns[file.split('.')[0]] = len(filtered_text)
                    unigram_temp=nltk.FreqDist(filtered_text)
                    filtered_bigram_text = nltk.bigrams(filtered_text)
                    # self.bigram_N_by_patterns[file.split('.')[0]] = len(filtered_bigram_text)
                    bigram_temp = nltk.FreqDist(filtered_bigram_text)
                    uni_bi_temp = unigram_temp
                    uni_bi_temp.update(bigram_temp)
                    self.unigram_features_by_patterns[file.split('.')[0]] = unigram_temp
                    self.bigram_features_by_patterns[file.split('.')[0]] = bigram_temp
                    self.features_by_patterns[file.split('.')[0]] = uni_bi_temp
                    self.patterns.append(file.split('.')[0])
            filtered_text = [self.stemmer.stem(w) for w in nltk.wordpunct_tokenize(self.all_text) if not w.lower() in self.s_words]
            # self.all_unigram_N = len(filtered_text)
            filtered_bigram_text = nltk.bigrams(filtered_text)
            # self.all_bigram_N = len(filtered_bigram_text)
            self.all_unigram_features = nltk.FreqDist(filtered_text)
            self.all_unigram_features = nltk.FreqDist(filtered_text)
            self.all_bigram_features = nltk.FreqDist(filtered_bigram_text)

            for k, v in self.all_unigram_features.items():
                for op in self.unigram_features_by_patterns:
                    for ki, vi in self.unigram_features_by_patterns[op].items():
                        if (k == ki):
                            try:
                                self.words_documents[k].add(op)
                            except:
                                self.words_documents[k] = {op}
            for k, v in self.all_bigram_features.items():
                for op in self.bigram_features_by_patterns:
                    for ki, vi in self.bigram_features_by_patterns[op].items():
                        if (k == ki):
                            try:
                                self.words_documents[k].add(op)
                            except:
                                self.words_documents[k] = {op}

            for i in self.words_documents:
                self.document_frequency[i]=len(self.words_documents[i])

            for i,v in self.document_frequency.items():
                if v>1:
                    self.all_features.append(i);

            # for f in self.all_features:
            #     print(f)
        except IOError:
            type, value, traceback = sys.exc_info()
            print("Some errors occured : " + type + "\n value: " + value + "\n traceback: " + traceback)

    def remove_punctuation(data):
        symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
        for i in range(len(symbols)):
            data = np.char.replace(data, symbols[i], ' ')
            data = np.char.replace(data, "  ", " ")
        data = np.char.replace(data, ',', '')
        return data

    def getUniBigramDataUsingTFIDF(self,tfidfThreshold=0.1):
        data =[]
        data.append(self.patterns)
        print(len(self.all_features))
        for feature in self.all_features:
            term_appears =[]
            for op, fs in self.features_by_patterns.items():
                exists =0
                try:
                    opfeature = fs.get(feature)
                    if opfeature !=None:
                        tf = opfeature
                        df = self.document_frequency[feature]
                        idf = np.log10((len(self.patterns) + 1) / (df + 1))
                        tfidf = tf * idf;
                        if (tfidf > tfidfThreshold):
                            exists=1
                    term_appears.append(exists)
                except:
                    print('Something went wrong in building words to organizational patterns matrix')
                    return
            data.append(term_appears)
        return data


    def getUniBiGramDataUsingTFIDF(self,min_sup_uni=4, max_sup_uni=30, min_sup_bi=2,tfidfThreshold=0.1):
        data =[]
        data.append(self.patterns)
        totalUni=0
        totalBi=0

        for k,v in self.all_unigram_features.items():
            if(v > min_sup_uni and v < max_sup_uni):
                uni_gram_appears =[]
                for op in self.unigram_features_by_patterns:
                    exists = 0;
                    for ki,vi in self.unigram_features_by_patterns[op].items():
                        if(k==ki):
                            tf=vi
                            df=v
                            idf = np.log10((len(self.patterns)+1)/(df+1))
                            tfidf = tf*idf;

                            if(tfidf>tfidfThreshold):
                                #print(op,k,tf,df,round(idf,2),round(tfidf,2))
                                exists = 1;
                    uni_gram_appears.append(exists)
                data.append(uni_gram_appears)
                self.totUni=self.totUni+1
        for k,v in self.all_bigram_features.items():
            if(v > min_sup_bi):
                bi_gram_appears =[]
                for op in self.bigram_features_by_patterns:
                    exists = 0;
                    for ki,vi in self.bigram_features_by_patterns[op].items():
                        if(k==ki):
                            tf=vi
                            df=v
                            idf = np.log10((len(self.patterns)+1)/(df+1))
                            tfidf = tf*idf;
                            if(tfidf>tfidfThreshold):
                                #print(op,k,tf,df,round(idf,2),round(tfidf,2))
                                exists = 1;
                    bi_gram_appears.append(exists)
                data.append(bi_gram_appears)
                self.totBi =self.totBi+1
        return data, self.totUni,self.totBi

    def getUniBiGramData(self,min_sup_uni=4, max_sup_uni=30, min_sup_bi=2):
        data =[]
        data.append(self.patterns)
        totalUni=0
        totalBi=0
        print("Total Uni-grams",totalUni)
        for k,v in self.all_unigram_features.items():
            if(v > min_sup_uni and v < max_sup_uni):
                uni_gram_appears =[]
                for op in self.unigram_features_by_patterns:
                    exists = 0;
                    for ki,vi in self.unigram_features_by_patterns[op].items():
                        if(k==ki):
                            exists=1;
                    uni_gram_appears.append(exists)
                data.append(uni_gram_appears)
                self.totUni=self.totUni+1
        for k,v in self.all_bigram_features.items():
            if(v > min_sup_bi):
                bi_gram_appears =[]
                for op in self.bigram_features_by_patterns:
                    exists = 0;
                    for ki,vi in self.bigram_features_by_patterns[op].items():
                        if(k==ki):
                            exists=1;
                    bi_gram_appears.append(exists)
                data.append(bi_gram_appears)
                self.totBi =self.totBi+1
        return data, self.totUni,self.totBi

    def getUniGramData(self,min_sup_uni=4, max_sup_uni=30):
        data =[]
        data.append(self.patterns)
        for k,v in self.all_unigram_features.items():
            if(v>min_sup_uni and v<max_sup_uni):
                uni_gram_appears =[]
                for op in self.unigram_features_by_patterns:
                    exists = 0;
                    for ki,vi in self.unigram_features_by_patterns[op].items():
                        if(k==ki):
                            exists=1;
                    uni_gram_appears.append(exists)
                data.append(uni_gram_appears)
                self.totUni=self.totUni+1
        return data, self.totUni

    def getBiGramData(self,min_sup_bi=2):
        data =[]
        data.append(self.patterns)
        for k,v in self.all_bigram_features.items():
            if(v>min_sup_bi):
                bi_gram_appears =[]
                for op in self.bigram_features_by_patterns:
                    exists = 0;
                    for ki,vi in self.bigram_features_by_patterns[op].items():
                        if(k==ki):
                            exists=1;
                    bi_gram_appears.append(exists)
                data.append(bi_gram_appears)
                self.totBi=self.totBi+1
        return data, self.totBi

    def prinUniBiGramData(self,min_sup_uni=4, max_sup_uni=30, min_sup_bi=2):
         for k, v in self.all_unigram_features.items():
            if (v > min_sup_uni and v < max_sup_uni):
                print('\n\n', k, v)
                for op in self.unigram_features_by_patterns:
                    for ki, vi in self.unigram_features_by_patterns[op].items():
                        if (k == ki):
                            print(op, vi)
                self.totUni=self.totUni+1
            print(self.totUni)

         for k, v in self.all_bigram_features.items():
            if (v > min_sup_bi):
                print('\n\n', k, v)
                for op in self.bigram_features_by_patterns:
                    for ki, vi in self.bigram_features_by_patterns[op].items():
                        if (k == ki):
                            print(op, vi)
                self.totBi=self.totBi+1
            print(self.totBi)


    def prinUniGramData(self,min_sup_uni=4, max_sup_uni=30):
         for k, v in self.all_unigram_features.items():
            if (v > min_sup_uni and v < max_sup_uni):
                print('\n\n', k, v)
                for op in self.unigram_features_by_patterns:
                    for ki, vi in self.unigram_features_by_patterns[op].items():
                        if (k == ki):
                            print(op, vi)
                self.totUni=self.totUni+1
            print(self.totUni)

    def prinUniGramData(self,min_sup_bi=2):
         for k, v in self.all_bigram_features.items():
            if (v > min_sup_bi):
                print('\n\n', k, v)
                for op in self.bigram_features_by_patterns:
                    for ki, vi in self.bigram_features_by_patterns[op].items():
                        if (k == ki):
                            print(op, vi)
                self.totBi=self.totBi+1
            print(self.totBi)
