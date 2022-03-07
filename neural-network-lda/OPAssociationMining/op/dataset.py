import nltk
import codecs
import os
import sys
import Utility
import Tokenizer
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


try:

    data_dir="../corpus/"
    all_data =[]
    patterns = []

    all_context_features = {}
    all_problem_features = {}
    all_therefore_features = {}
    all_consequences_features = {}

    all_unigram_context_features = {}
    all_unigram_problem_features = {}
    all_unigram_therefore_features = {}
    all_unigram_consequences_features = {}

    all_bigram_context_features = {}
    all_bigram_problem_features = {}
    all_bigram_therefore_features = {}
    all_bigram_consequences_features = {}

    all_uni_bigram_context_features = {}
    all_uni_bigram_problem_features = {}
    all_uni_bigram_therefore_features = {}
    all_uni_bigram_consequences_features = {}


    all_unigram_context_features_by_patterns = {}
    all_unigram_problem_features_by_patterns = {}
    all_unigram_therefore_features_by_patterns = {}
    all_unigram_consequences_features_by_patterns = {}

    all_bigram_context_features_by_patterns = {}
    all_bigram_problem_features_by_patterns = {}
    all_bigram_therefore_features_by_patterns = {}
    all_bigram_consequences_features_by_patterns = {}

    all_uni_bigram_context_features_by_patterns = {}
    all_uni_bigram_problem_features_by_patterns = {}
    all_uni_bigram_therefore_features_by_patterns = {}
    all_uni_bigram_consequences_features_by_patterns = {}

    all_context_text = "";
    all_problem_text = "";
    all_therefore_text = "";
    all_consequences_text = "";

    totUni=0
    totBi=0
    stemmer = SnowballStemmer('english')
    s_words = stopwords.words('english')
    s_words.extend(
       ['...', '.', '.✥', '✥', '\'', '\"', ',', ':', '.**', '(', ')', 'e', '.,', '-', '....', ';', '[', ']', '—', '.)',
        'therefor', '’', '([', '])','--','/','],','ff','e','g','bibref','60ksloc','foote2000'])
    for file in os.listdir(data_dir):
        with open(data_dir + file, 'r', encoding="utf-8") as f:
            sectiondetector=0;
            temp_text = ""
            temp_context=""
            temp_problem=""
            temp_therefore=""
            temp_consequences=""
            for line in f:
                line = line.strip()
                if not line.isspace() and not line.startswith('\n'):
                   if "✥ ✥ ✥" in line or "Therefore" in line:
                       sectiondetector+=1
                       continue
                   if sectiondetector==0:
                       temp_context+=line
                   if sectiondetector==1:
                       temp_problem+=line
                   if sectiondetector==2:
                       temp_therefore+=line
                   if sectiondetector==3:
                       temp_consequences+=line
            patterns.append(file.split('.')[0])

            filtered_context = [stemmer.stem(w) for w in nltk.wordpunct_tokenize(temp_context) if not w.lower() in s_words]
            filtered_problem = [stemmer.stem(w) for w in nltk.wordpunct_tokenize(temp_problem) if not w.lower() in s_words]
            filtered_therefore = [stemmer.stem(w) for w in nltk.wordpunct_tokenize(temp_therefore) if not w.lower() in s_words]
            filtered_consequences = [stemmer.stem(w) for w in nltk.wordpunct_tokenize(temp_consequences) if not w.lower() in s_words]

            all_context_text+=temp_context
            all_problem_text+=temp_problem
            all_therefore_text+=temp_therefore
            all_consequences_text+=temp_consequences

            unigram_context_temp=nltk.FreqDist(filtered_context)
            bigram_context_temp = nltk.FreqDist(nltk.bigrams(filtered_context))
            uni_bi_context_temp = unigram_context_temp
            uni_bi_context_temp.update(bigram_context_temp)

            unigram_problem_temp=nltk.FreqDist(filtered_problem)
            bigram_problem_temp = nltk.FreqDist(nltk.bigrams(filtered_problem))
            uni_bi_problem_temp = unigram_problem_temp
            uni_bi_problem_temp.update(bigram_problem_temp)

            unigram_therefore_temp=nltk.FreqDist(filtered_therefore)
            bigram_therefore_temp = nltk.FreqDist(nltk.bigrams(filtered_therefore))
            uni_bi_therefore_temp = unigram_therefore_temp
            uni_bi_therefore_temp.update(bigram_therefore_temp)

            unigram_consequences_temp=nltk.FreqDist(filtered_consequences)
            bigram_consequences_temp = nltk.FreqDist(nltk.bigrams(filtered_consequences))
            uni_bi_consequences_temp = unigram_consequences_temp
            uni_bi_consequences_temp.update(bigram_consequences_temp)



            all_unigram_context_features_by_patterns[file.split('.')[0]] = unigram_context_temp
            all_unigram_problem_features_by_patterns[file.split('.')[0]] = unigram_problem_temp
            all_unigram_therefore_features[file.split('.')[0]] = unigram_therefore_temp
            all_unigram_context_features_by_patterns[file.split('.')[0]] = unigram_consequences_temp

            all_bigram_context_features_by_patterns[file.split('.')[0]] = bigram_context_temp
            all_bigram_problem_features_by_patterns[file.split('.')[0]] = bigram_problem_temp
            all_bigram_therefore_features[file.split('.')[0]] = bigram_therefore_temp
            all_bigram_context_features_by_patterns[file.split('.')[0]] = bigram_consequences_temp

            all_uni_bigram_context_features_by_patterns[file.split('.')[0]] =uni_bi_context_temp
            all_uni_bigram_problem_features_by_patterns[file.split('.')[0]] = uni_bi_problem_temp
            all_uni_bigram_therefore_features[file.split('.')[0]] = unigram_therefore_temp
            all_uni_bigram_context_features_by_patterns[file.split('.')[0]] = unigram_consequences_temp




            # print('\n'+file.split('.')[0])
            # print("Context")
            # print(temp_context)
            # print("Problem")
            # print(temp_problem)
            # print("Therefore")
            # print(temp_therefore)
            # print("Consequences")
            # print(temp_consequences)

    filtered_context = [stemmer.stem(w) for w in nltk.wordpunct_tokenize(all_context_text) if not w.lower() in s_words]
    filtered_problem = [stemmer.stem(w) for w in nltk.wordpunct_tokenize(all_problem_text) if not w.lower() in s_words]
    filtered_therefore = [stemmer.stem(w) for w in nltk.wordpunct_tokenize(all_therefore_text) if not w.lower() in s_words]
    filtered_consequences = [stemmer.stem(w) for w in nltk.wordpunct_tokenize(all_consequences_text) if not w.lower() in s_words]


    all_unigram_context_features = nltk.FreqDist(filtered_context)
    all_unigram_problem_features= nltk.FreqDist(filtered_problem)
    all_unigram_therefore_features= nltk.FreqDist(filtered_therefore)
    all_unigram_consequences_features = nltk.FreqDist(filtered_consequences)

    all_bigram_context_features = nltk.FreqDist(nltk.bigrams(filtered_context))
    all_bigram_problem_features = nltk.FreqDist(nltk.bigrams(filtered_problem))
    all_bigram_therefore_features = nltk.FreqDist(nltk.bigrams(filtered_therefore))
    all_bigram_consequences_features = nltk.FreqDist(nltk.bigrams(filtered_consequences))

    uni_bi_context_temp = unigram_context_temp
    uni_bi_context_temp.update(bigram_context_temp)
    uni_bi_problem_temp = unigram_problem_temp
    uni_bi_problem_temp.update(bigram_problem_temp)
    uni_bi_therefore_temp = unigram_therefore_temp
    uni_bi_therefore_temp.update(bigram_therefore_temp)
    uni_bi_consequences_temp = unigram_consequences_temp
    uni_bi_consequences_temp.update(bigram_consequences_temp)

    all_uni_bigram_context_features = uni_bi_context_temp
    all_uni_bigram_problem_features = uni_bi_problem_temp
    all_uni_bigram_therefore_features = uni_bi_therefore_temp
    all_uni_bigram_consequences_features = uni_bi_consequences_temp


    data_context = []
    data_problem = []
    data_therefore = []
    data_consequences = []
    totalUni = 0
    totalBi = 0
    for k, v in all_unigram_context_features.items():
        if (v > 2 and v < 20):
            uni_gram_appears = []
            for op in all_unigram_context_features_by_patterns:
                exists = 0;
                for ki, vi in all_unigram_context_features_by_patterns[op].items():
                    if (k == ki):
                        exists = 1;
                uni_gram_appears.append(exists)
            data_context.append(uni_gram_appears)

    for k, v in all_bigram_context_features.items():
        if (v > 2):
            bi_gram_appears = []
            for op in all_bigram_context_features_by_patterns:
                exists = 0;
                for ki, vi in all_bigram_context_features_by_patterns[op].items():
                    if (k == ki):
                        exists = 1;
                bi_gram_appears.append(exists)
            data_context.append(bi_gram_appears)

    for k, v in all_unigram_problem_features.items():
        if (v > 2 and v < 20):
            uni_gram_appears = []
            for op in all_unigram_problem_features_by_patterns:
                exists = 0;
                for ki, vi in all_unigram_problem_features_by_patterns[op].items():
                    if (k == ki):
                        exists = 1;
                uni_gram_appears.append(exists)
            data_problem.append(uni_gram_appears)

    for k, v in all_bigram_problem_features.items():

        if (v > 2):
            bi_gram_appears = []
            for op in all_bigram_problem_features_by_patterns:
                exists = 0;
                for ki, vi in all_bigram_problem_features_by_patterns[op].items():
                    if (k == ki):
                        exists = 1;
                bi_gram_appears.append(exists)
            data_problem.append(bi_gram_appears)

    for k, v in all_unigram_therefore_features.items():
        if (v > 2 and v < 20):
            uni_gram_appears = []
            for op in all_unigram_therefore_features_by_patterns:
                exists = 0;
                for ki, vi in all_unigram_therefore_features_by_patterns[op].items():
                    if (k == ki):
                        exists = 1;
                uni_gram_appears.append(exists)
            data_therefore.append(uni_gram_appears)

    for k, v in all_bigram_therefore_features.items():
        if (v > 2):
            bi_gram_appears = []
            for op in all_bigram_therefore_features_by_patterns:
                exists = 0;
                for ki, vi in all_bigram_therefore_features_by_patterns[op].items():
                    if (k == ki):
                        exists = 1;
                bi_gram_appears.append(exists)
            data_therefore.append(bi_gram_appears)

    for k, v in all_unigram_consequences_features.items():
        if (v > 2 and v < 20):
            uni_gram_appears = []
            for op in all_unigram_consequences_features_by_patterns:
                exists = 0;
                for ki, vi in all_unigram_context_features_by_patterns[op].items():
                    if (k == ki):
                        exists = 1;
                uni_gram_appears.append(exists)
            data_consequences.append(uni_gram_appears)

    for k, v in all_bigram_consequences_features.items():
        if (v > 2):
            bi_gram_appears = []
            for op in all_bigram_consequences_features_by_patterns:
                exists = 0;
                for ki, vi in all_bigram_consequences_features_by_patterns[op].items():
                    if (k == ki):
                        exists = 1;
                bi_gram_appears.append(exists)
            data_consequences.append(bi_gram_appears)


    for i in data_problem:
        print(i)
    for i in data_context:
        print(i)


    data_1 = data_problem
    for i in data_context:
        data_1.append(i)

    data_2 = data_therefore
    for i in data_consequences:
        data_2.append(i)


except IOError:
    type, value, traceback = sys.exc_info()
    print("Some errors occured : " + type + "\n value: " + value + "\n traceback: " + traceback)
