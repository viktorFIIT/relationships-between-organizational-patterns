import nltk
import os
import sys
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np

op = opd.CorpusReader("corpus");

class CorpusReader:
    data_dir=""
    all_data =[]
    patterns = []
    all_features = []
    all_unigram_features = {}
    all_bigram_features = {}
    unigram_features_by_patterns={}
    bigram_features_by_patterns={}
    features_by_patterns = {}
    words_documents={}
    document_frequency={}
    all_text = "";
    totUni=0
    totBi=0
    stemmer = SnowballStemmer('english')
    s_words = stopwords.words('english')
    s_words.extend(
       ['...', '.', '.✥', '✥', '\'', '\"','\",','.\"', ',', ':', '.**','(\"','\")', '\").','?\"','(', ')', 'e', '.,', '-', '....', ';', '[', ']', '—', '.)',
        'therefor', '’', '([', '])','--','/','],','ff','e','g','bibref','60ksloc','foote2000'])
    totalcorpuslenght=0;
    def __init__(self,datadir):
        self.data_dir=os.getcwd() + "/"+datadir+"/"
        symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
        try:
            for file in os.listdir(self.data_dir):
                with open(self.data_dir+ file, 'r', encoding="utf-8") as f:
                    temp_text = ""
                    for line in f:
                        self.totalcorpuslenght+= len(line);
                        for i in range(len(symbols)):
                                line = np.str.replace(line, symbols[i], ' ')
                                line = np.str.replace(line, "  ", " ")
                        line= np.str.replace(line, "'", "")
                        line = line.strip()
                        if not line.isspace() and not line.startswith('\n'):
                            temp_text += line
                            self.all_text += line

                    filtered_text = [self.stemmer.stem(w) for w in nltk.wordpunct_tokenize(temp_text) if not w.lower() in self.s_words and len(w)>2]
                    unigram_temp=nltk.FreqDist(filtered_text)
                    filtered_bigram_text = nltk.bigrams(filtered_text)
                    bigram_temp = nltk.FreqDist(filtered_bigram_text)
                    uni_bi_temp = unigram_temp
                    uni_bi_temp.update(bigram_temp)
                    self.unigram_features_by_patterns[file.split('.')[0]] = unigram_temp
                    self.bigram_features_by_patterns[file.split('.')[0]] = bigram_temp
                    self.features_by_patterns[file.split('.')[0]] = uni_bi_temp
                    self.patterns.append(file.split('.')[0])

            filtered_text = [self.stemmer.stem(w) for w in nltk.wordpunct_tokenize(self.all_text) if not w.lower() in self.s_words]
            filtered_bigram_text = nltk.bigrams(filtered_text)
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
                    self.all_features.append(i)
            print("Total Corpus Text with out cleaning ",self.totalcorpuslenght)
            print("Bag-of-Words vocabulary size: ",len(self.all_features))
        except IOError:
            type, value, traceback = sys.exc_info()
            print("Some errors occured : " + type + "\n value: " + value + "\n traceback: " + traceback)

    def getUniBigramDataUsingTFIDF(self,tfidfThreshold=0.1):
        data =[]
        data.append(self.patterns)

        for feature in self.all_features:
            term_appears =[]
            df = self.document_frequency[feature]
            idf = np.log10((len(self.patterns) + 1) / (df + 1))
            for op, fs in self.features_by_patterns.items():
                exists =0
                try:
                    opfeature = fs.get(feature)
                    if opfeature !=None:
                        tf = opfeature
                        tfidf = tf * idf;
                        if (tfidf >= tfidfThreshold):
                            #print(feature,op,tf,df,round(idf,3),round(tfidf,3))
                            exists=1
                    term_appears.append(exists)
                except:
                    print('Something went wrong in building words to organizational patterns matrix')
                    return
            data.append(term_appears)
        return data
    def getWordsStatisticsFor(self, pattern1,pattern2):
        data ={}
        statistics ={}
        p1unigramsize=0
        p1bigramsize=0
        p2unigramsize=0
        p2bigramsize=0
        unigramcommon=0
        bigramcommon=0
        totalcommon=0
        for p1f,p1v in self.features_by_patterns.get(pattern1).items():
            if(type(p1f)==tuple):
                    p1bigramsize+=1
            else:
                    p1unigramsize+=1
            for p2f,p2v in self.features_by_patterns.get(pattern2).items():
                if(p1bigramsize==1 or p1bigramsize==1):
                    if(type(p2f)==tuple):
                        p2bigramsize+=1
                    else:
                        p2unigramsize+=1
                if p1f==p2f:
                    data[p1f]= [p1v,p2v]
                    if(type(p1f)==tuple):
                        bigramcommon+=1
                    else:
                        unigramcommon+=1
        statistics[pattern1,'Unigrams']=p1unigramsize
        statistics[pattern1,'Bigrams']=p1bigramsize
        statistics[pattern1,'Total']=p1unigramsize+p1bigramsize
        statistics[pattern2,'Unigrams']=p2unigramsize
        statistics[pattern2,'Bigrams']=p2bigramsize
        statistics[pattern2,'Total']=p2unigramsize+p2bigramsize
        statistics['Common','Unigrams']=unigramcommon
        statistics['Common','Bigrams']=bigramcommon
        statistics['Common','Total']=unigramcommon+bigramcommon
        return  data,statistics
    def getWordsStatisticsFor(self, pattern1,pattern2,pattern3):
        data ={}
        statistics ={}
        p1unigramsize=0
        p1bigramsize=0
        p2unigramsize=0
        p2bigramsize=0
        p3unigramsize=0
        p3bigramsize=0
        unigramcommon=0
        bigramcommon=0
        for p1f,p1v in self.features_by_patterns.get(pattern1).items():
            if(type(p1f)==tuple):
                    p1bigramsize+=1
            else:
                    p1unigramsize+=1
            for p2f,p2v in self.features_by_patterns.get(pattern2).items():
                if(p1unigramsize==1):
                    if(type(p2f)==tuple):
                        p2bigramsize+=1
                    else:
                        p2unigramsize+=1
                if p1f==p2f:
                    for p3f,p3v in self.features_by_patterns.get(pattern3).items():
                        if(p1unigramsize==1):
                            if(type(p3f)==tuple):
                                p3bigramsize+=1
                            else:
                                p3unigramsize+=1
                        if p2f==p3f:
                            data[p1f]= [p1v,p2v,p3v]
                            if(type(p1f)==tuple):
                                bigramcommon+=1
                            else:
                                unigramcommon+=1

        statistics[pattern1,'Unigrams']=p1unigramsize
        statistics[pattern1,'Bigrams']=p1bigramsize
        statistics[pattern1,'Total']=p1unigramsize+p1bigramsize
        statistics[pattern2,'Unigrams']=p2unigramsize
        statistics[pattern2,'Bigrams']=p2bigramsize
        statistics[pattern2,'Total']=p2unigramsize+p2bigramsize
        statistics[pattern3,'Unigrams']=p3unigramsize
        statistics[pattern3,'Bigrams']=p3bigramsize
        statistics[pattern3,'Total']=p3unigramsize+p3bigramsize
        statistics['Common','Unigrams']=unigramcommon
        statistics['Common','Bigrams']=bigramcommon
        statistics['Common','Total']=unigramcommon+bigramcommon
        return  data,statistics

    def getStateistics(self,patterns,tfidfThreshold):
        data={}
        for feature in self.all_features:
            term_appears =[]
            df = self.document_frequency[feature]
            idf = np.log10((len(self.patterns) + 1) / (df + 1))
            exists = 1
            tempfeature=[]
            for p in patterns:
                try:
                    opfeature = self.features_by_patterns[p].get(feature)
                    if opfeature !=None:
                        tf = opfeature
                        tfidf = tf * idf
                        if (tfidf < tfidfThreshold):
                            exists = 0
                        tempfeature.append(round(tfidf,2))
                except:
                    print('Something went wrong in building words to organizational patterns matrix')
                    return
            if exists and len(tempfeature) > 2:
                data[feature]=tempfeature
        return data

    def getNgramsExistence(self,patterns,tfidfThreshold):
        data={}
        for feature in self.all_features:
            term_appears =[]
            df = self.document_frequency[feature]
            idf = np.log10((len(self.patterns) + 1) / (df + 1))

            tempfeature=[]
            for p in patterns:
                exists = 0
                try:
                    opfeature = self.features_by_patterns[p].get(feature)
                    if opfeature !=None:
                        tf = opfeature
                        tfidf = tf * idf
                        if (tfidf >= tfidfThreshold):
                            exists = 1
                    tempfeature.append(exists)
                except:
                    print('Something went wrong in building words to organizational patterns matrix')
                    return
            data[feature]=tempfeature
        return data
    def getNgramsExistenceStatistics(self,patterns,tfidfThreshold):
        data={}
        for feature in self.all_features:
            term_appears =[]
            df = self.document_frequency[feature]
            idf = np.log10((len(self.patterns) + 1) / (df + 1))
            for p in patterns:
                try:
                    opfeature = self.features_by_patterns[p].get(feature)
                    if opfeature !=None:
                        tf = opfeature
                        tfidf = tf * idf
                        if (tfidf >= tfidfThreshold):
                            try:
                                data[p]=data[p]+1
                            except:
                                data[p]=1
                except:
                    print('Something went wrong in building words to organizational patterns matrix')
                    return
        return data
