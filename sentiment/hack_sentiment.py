

import numpy as np
import math
from math import isinf
from nltk.stem.porter import PorterStemmer
from scipy import spatial # fast impl of cosine distance
import logging
import pickle
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import plotly.plotly as py
import time
import pandas as pd
from joblib import Parallel, delayed
import os.path
import matplotlib.colors as mcolors


#create global variables
stemmer = PorterStemmer()


#____ cel boundary
#____ cel boundary
#____ cel boundary



# setmming, punct routines

def remove_punct(document1):
    document1 = document1.lower()
    document1 = document1.replace('\n', ' ')
    document1 = document1.replace("\n", ' ')
    document1 = document1.replace("\t", ' ')
    document1 = document1.replace(',', ' ')
    document1 = document1.replace('.', ' ')
    document1 = document1.replace(')', ' ')
    document1 = document1.replace('(', ' ')
    document1 = document1.replace('"', ' ')
    document1 = document1.replace("'", ' ')
    document1 = document1.replace(":", ' ')
    document1 = document1.replace("?", ' ')
    document1 = document1.replace("!", ' ')
    document1 = document1.replace("“", ' ')

    document1 = document1.replace("@", ' ')
    document1 = document1.replace("[", ' ')
    document1 = document1.replace("]", ' ')
    document1 = document1.replace("/", ' ')
    document1 = document1.replace("‘", ' ')
    document1 = document1.replace(";", ' ')
    document1 = document1.replace(":", ' ')
    document1 = document1.replace("’", ' ')
    document1 = document1.replace("”", ' ')
    document1 = document1.replace("   ", ' ')
    document1 = document1.replace("  ", ' ')

    return (document1)


def get_stemmed(list_of_words, make_unique_and_sort=True):
    stemmed = []
    for w in list_of_words:
        w = stemmer.stem(w)  # use Porter's stemmer
        if len(w) < 3:  # remove short tokens
            continue
        stemmed.append(w)
    if make_unique_and_sort:
        unique = list(set(stemmed))
        unique.sort()
        return unique
    else:
        return stemmed


def pre_process(non_stemmed_text):
    t = remove_punct(non_stemmed_text)
    t = get_stemmed(t.split(' '), make_unique_and_sort=False)
    t = " ".join(t)
    return t


#____ cel boundary




class SentimentProcessor(object):
    def __init__(self, afinn_filename, debug_mode=0):
        self.load_afinn(afinn_filename, debug_mode)
        self.scores_by_term = {}

    def load_afinn(self, afinn_filename, debug_mode=0):
        afinnfile = open(afinn_filename)
        print(afinn_filename, 'loaded')

        # load the hashmap of sentiment terms
        self.scores_by_term = {}  # initialize an empty dictionary
        for line in afinnfile:
            term, score = line.split("\t")  # The file is tab-delimited
            term = pre_process(term)
            self.scores_by_term[term] = int(score)  # Convert the score to an integer.
        if debug_mode >= 2:
            print(self.scores_by_term.items())  # Print every (term, score) pair in the dictionary

    def cal_term_weight_on_full_corpus(self, list_of_documents, debug_mode=0):
        self.counts_by_term = {}  # initialize an empty dictionary
        self.total_sentiment_by_word = {}
        self.sum_sentiment = 0
        self.max_sentiment = 0
        self.min_sentiment = 0
        self.min_text = ''
        self.max_text = ''
        for doc in list_of_documents:
            doc_sentiment = 0
            text = pre_process(doc)
            words = text.split()
            for word in words:
                if word in self.scores_by_term:
                    doc_sentiment = doc_sentiment + self.scores_by_term[word]
                if word in self.counts_by_term:
                    self.counts_by_term[word] = self.counts_by_term[word] + 1
                else:
                    self.counts_by_term[word] = 1
            if debug_mode >= 2 & doc_sentiment != 0:
                print(str(doc_sentiment) + ' |   ' + text.replace('\n', ' ').replace('\r', ' '))
            # add the doc sentiment to all the words in the doc
            for word in words:
                if word in self.total_sentiment_by_word:
                    self.total_sentiment_by_word[word] = self.total_sentiment_by_word[word] + doc_sentiment
                else:
                    self.total_sentiment_by_word[word] = doc_sentiment
            self.sum_sentiment = self.sum_sentiment + doc_sentiment
            if (doc_sentiment > self.max_sentiment):
                self.max_sentiment = doc_sentiment
                self.max_text = text
            if (doc_sentiment < self.min_sentiment):
                self.min_sentiment = doc_sentiment
                self.min_text = text
            if debug_mode == 2:
                print(str(doc_sentiment))

    def calc_term_total_sent(self, debug_mode=0):
        term_total_sentiment = 0
        for word in self.total_sentiment_by_word:
            if word in self.scores_by_term:
                term_total_sentiment = term_total_sentiment + self.total_sentiment_by_word[word]

        sum_key_sent_contibution = 0.0;
        for word in self.scores_by_term:
            if word in self.counts_by_term:
                sum_key_sent_contibution = sum_key_sent_contibution + (
                self.scores_by_term[word] * self.counts_by_term[word])

        self.ratio = term_total_sentiment / float(sum_key_sent_contibution)

        if debug_mode == 2:
            print('Total Dict Sentiment ', str(sum_key_sent_contibution))
            print('Total File Sentiment ', str(term_total_sentiment))
            print('Ratio ', str(self.ratio))

        if debug_mode == 2:
            for word in scores_by_term:
                if word in total_sentiment_by_word:
                    print(word.encode('utf-8'),
                          str(scores_by_term[word]) + ' ' + str(self.total_sentiment_by_word[word]),
                          str(self.total_sentiment_by_word[word] / float(self.ratio)))

        # print the outputs, each word, and the scaled amount it might contribute to the sentiment if it were in the doctionary.
        if debug_mode >= 1:
            for word in self.total_sentiment_by_word:
                print(word, str(self.total_sentiment_by_word[word] / float(self.ratio)))

    def get_doc_sentiment(self, text, debug_mode=0):
        doc_sentiment = 0
        text = pre_process(text)
        words = text.split()
        for word in words:
            if word in self.total_sentiment_by_word:
                doc_sentiment = doc_sentiment + self.total_sentiment_by_word[word]
        if debug_mode >= 2:
            print('raw doc sentiment', doc_sentiment)
            print('doc_sentiment', str(doc_sentiment / self.ratio))
            print('norm tweet_sentiment', (doc_sentiment / self.ratio / len(words)))
        return (doc_sentiment / self.ratio / len(words))


#____ cel boundary

## unit tests sentiment

afinn_filename = './AFINN-111.txt'

sp = SentimentProcessor(afinn_filename)
sp.cal_term_weight_on_full_corpus(['abhor abandoned archaeological',' happy apple archaeological '])
sp.calc_term_total_sent()



print('happy has a doc normalised value of:',sp.get_doc_sentiment('happy'))
print('archaeological has a doc normalised value of:',sp.get_doc_sentiment('archaeological'))
assert(sp.get_doc_sentiment('archaeolog') < 0) # with 2 neg, and 1 +ve word
assert(abs(sp.get_doc_sentiment('happy') - 0.8571428571428571)<max_diff)
assert(sp.get_doc_sentiment('archaeolog') < 0) # with 2 neg, and 1 +ve word



#____ cel boundary

#build sentiment model for all words - unstemmed

print ('started at:',time.strftime("%Y-%m-%d %H:%M"))
list_of_all_text = []
for hi in range(len(ds.stances)):
    headline = remove_punct(ds.stances[hi]['Headline'],)
    list_of_all_text.append(headline)
for i, bi_key in enumerate(ds.articles.keys()):
    body = remove_punct(ds.articles[bi_key])
    list_of_all_text.append(body)

afinn_filename = './AFINN-111.txt'
sp_all_raw = SentimentProcessor(afinn_filename)
sp_all_raw.cal_term_weight_on_full_corpus(list_of_all_text)
sp_all_raw.calc_term_total_sent()
save_model(sp_all_raw,'./', 'sent_model_unstemmed.pickle')
print ('completed at:',time.strftime("%Y-%m-%d %H:%M"))



#____ cel boundary


# Calc headline and body sent for each headline
if os.path.isfile('./sent_by_stance.pickle'):
    # load and continue
    print('loading from pickle')
    sentiment_by_stance2 = load_model('./', 'sent_by_stance.pickle')
else:
    print('started at:', time.strftime("%Y-%m-%d %H:%M"))
    sentiment_by_stance2 = {'agree': [], 'unrelated': [], 'discuss': [], 'disagree': []}
    for i in range(len(ds.stances)):
        headline = ds.stances[i]['Headline']
        key = ds.stances[i]['Body ID']
        bs = body_sents[key]
        stance = ds.stances[i]['Stance']
        body = ds.articles[key]
        s1 = sp_all_raw.get_doc_sentiment(headline)
        s2 = sp_all_raw.get_doc_sentiment(body)
        # print(stance, '\t', int(s1), '\t', int(s2), '\t', int(abs(s1-s2)) )
        d = s1 - s2
        sentiment_by_stance2[stance].append(d)
        if i % 1000 == 0:
            print(time.strftime("%Y-%m-%d %H:%M"), 100 * (i / len(ds.stances)), '% completed')

    print('completed at:', time.strftime("%Y-%m-%d %H:%M:%S"))
    save_model(sentiment_by_stance2, './', 'sent_by_stance.pickle')



#____ cel boundary

print ('bodies started at:' ,time.strftime("%Y-%m-%d %H:%M:%S"))

body_sentiment = [0] * len(ds.articles.keys())
for i ,key in enumerate(ds.articles.keys()):
    body = pre_process(ds.articles[key])
    s1 = sp_all_raw.get_doc_sentiment(body)
    body_sentiment[i] = s1
    if i % 800 == 0:
        print (time.strftime("%Y-%m-%d %H:%M:%S") ,i ,'processed')
        # print(s1)

print ('headers started at:' ,time.strftime("%Y-%m-%d %H:%M:%S"))

headline_sentiment = [0] * len(ds.stances)
for i in range(0 ,len(ds.stances)):
    headline = pre_process(ds.stances[i]['Headline'])
    s1 = sp_all_raw.get_doc_sentiment(headline)
    headline_sentiment[i] = s1
    if i % 5000 == 0:
        print (time.strftime("%Y-%m-%d %H:%M:%S") ,i ,'processed')

        # print(s1)

print ('completed at:' ,time.strftime("%Y-%m-%d %H:%M:%S"))


#____ cel boundary

body_sentiment = body_sentiment - np.min(body_sentiment)
body_sentiment = body_sentiment / np.max(body_sentiment)
headline_sentiment = headline_sentiment - np.min(headline_sentiment)
headline_sentiment = headline_sentiment / np.max(headline_sentiment)


print(np.mean(body_sentiment))
print(np.mean(headline_sentiment))
print(np.min(body_sentiment))
print(np.min(headline_sentiment))
print(np.max(body_sentiment))
print(np.max(headline_sentiment))

print ('started at:',time.strftime("%Y-%m-%d %H:%M"))
norm_sentiment_by_stance = {'agree':[],'unrelated':[], 'discuss':[],'disagree':[]}
norm_sentiment_diff = [0] * len(ds.stances)
for i in range(len(ds.stances)):
    headline = ds.stances[i]['Headline']
    key = ds.stances[i]['Body ID']
    stance = ds.stances[i]['Stance']
    s1 = headline_sentiment[i]
    body_index = list(ds.articles.keys()).index(key)
    s2 = body_sentiment[body_index]
    d = s1 - s2
    norm_sentiment_diff[i] = d
    norm_sentiment_by_stance[stance].append(d)
    if i % 5000 == 0:
        print (time.strftime("%Y-%m-%d %H:%M"),100*(i/len(ds.stances)),'% completed')

print ('completed at:',time.strftime("%Y-%m-%d %H:%M:%S"))







def plot_sent_v2(sentiment_by_stance):


    fig = plt.figure()
    for stance in ['disagree']:
        estimated_pdf_via_hist = np.histogram(sentiment_by_stance[stance], bins=100)
        y = estimated_pdf_via_hist[0] / max(estimated_pdf_via_hist[0])
        plt.plot(estimated_pdf_via_hist[1][1:], y, label=stance)
        #

    for stance in ['agree']:
        estimated_pdf_via_hist = np.histogram(sentiment_by_stance[stance], bins=100)
        y = estimated_pdf_via_hist[0] / max(estimated_pdf_via_hist[0])
        plt.plot(estimated_pdf_via_hist[1][1:], y, label=stance)
        # plt.xlabel(stance)

    for stance in ['discuss']:
        estimated_pdf_via_hist = np.histogram(sentiment_by_stance[stance], bins=100)
        y = estimated_pdf_via_hist[0] / max(estimated_pdf_via_hist[0])
        plt.plot(estimated_pdf_via_hist[1][1:], y, label=stance)
        # plt.xlabel(stance)

    for stance in ['unrelated']:
        estimated_pdf_via_hist = np.histogram(sentiment_by_stance[stance], bins=100)
        y = estimated_pdf_via_hist[0] / max(estimated_pdf_via_hist[0])
        plt.plot(estimated_pdf_via_hist[1][1:], y, label=stance)
        # plt.xlabel(stance)
    plt.xlabel('Sentiment')
    plt.legend()
    plt.title('Sentiment Anaylsis for the 4 Stances')
    plt.savefig('sentiment_by_stance.png')
    plt.show()

plot_sent_v2(sentiment_by_stance2)
