import os
import pickle
import logging

from sellibrary.sentiment.stemmer import Stemmer


class SentimentProcessor(object):

    # set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def __init__(self, debug_mode=0):
        self.counts_by_term = {}  # initialize an empty dictionary
        self.total_sentiment_by_word = {}
        self.sum_sentiment = 0
        self.max_sentiment = 0
        self.min_sentiment = 0
        self.min_text = ''
        self.max_text = ''
        self.scores_by_term = {}
        # __
        self.ratio = 0.0

    def save_model(self, output_filename):
        struct = {}
        struct['counts_by_term'] = self.counts_by_term
        struct['total_sentiment_by_word'] = self.total_sentiment_by_word
        struct['sum_sentiment'] = self.sum_sentiment
        struct['max_sentiment'] = self.max_sentiment
        struct['min_sentiment'] = self.min_sentiment
        struct['min_text'] = self.min_text
        struct['max_text'] = self.max_text
        struct['scores_by_term'] = self.scores_by_term
        struct['ratio'] = self.ratio
        self.logger.info('About to write %s', output_filename)
        with open(output_filename, 'wb') as handle:
            pickle.dump(struct, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info('file written = %s', output_filename)

    def load_model(self, input_filename):
        if os.path.isfile(input_filename):
            self.logger.info('loading sentiment from %s', input_filename)
            with open(input_filename, 'rb') as handle:
                struct = pickle.load(handle)
                self.counts_by_term = struct['counts_by_term']
                self.total_sentiment_by_word = struct['total_sentiment_by_word']
                self.sum_sentiment = struct['sum_sentiment']
                self.max_sentiment = struct['max_sentiment']
                self.min_sentiment = struct['min_sentiment']
                self.min_text = struct['min_text']
                self.max_text = struct['max_text']
                self.scores_by_term = struct['scores_by_term']
                self.ratio = struct['ratio']

            self.logger.info('loaded')
        else:
            self.logger.warning('file not found %s', input_filename)

    def remove_punct(self, document1):
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

        return document1

    def pre_process(self, non_stemmed_text):
        t = self.remove_punct(non_stemmed_text)
        stemmer = Stemmer()
        t = stemmer.get_stemmed(t.split(' '), make_unique_and_sort=False)
        t = " ".join(t)
        return t

    def get_ngram(self, text, n, center_word_start_index, center_word_end_index):
        text = self.remove_punct(text)
        i = 0
        p_index = center_word_start_index
        while i <= n and p_index > -1:
            p_index = text.rfind(' ', 0, p_index)
            i += 1
        i = 0
        n_index = center_word_end_index
        while i <= n and n_index >= center_word_end_index:
            n_index = text.find(' ', n_index)
            i += 1
            n_index += 1
        if p_index < 0:
            p_index = 0
        if n_index <= 0:
            n_index = len(text)
        return text[p_index:n_index].strip()

    def load_afinn(self, afinn_filename, debug_mode=0):
        if os.path.isfile(afinn_filename):
            afinnfile = open(afinn_filename)
            print(afinn_filename, 'loaded')

            # load the hashmap of sentiment terms
            self.scores_by_term = {}  # initialize an empty dictionary
            for line in afinnfile:
                term, score = line.split("\t")  # The file is tab-delimited
                term = self.pre_process(term)
                self.scores_by_term[term] = int(score)  # Convert the score to an integer.
            if debug_mode >= 2:
                print(self.scores_by_term.items())  # Print every (term, score) pair in the dictionary
        else:
            self.logger.warning('afinn file not found %s',afinn_filename)


    def cal_term_weight_on_full_corpus(self, afinn_filename, list_of_ngrams, debug_mode=0):


        # Builds a dictionary of sentiment by add sentiment to all words
        # in the 'document'
        self.counts_by_term = {}  # initialize an empty dictionary
        self.total_sentiment_by_word = {}
        self.sum_sentiment = 0
        self.max_sentiment = 0
        self.min_sentiment = 0
        self.min_text = ''
        self.max_text = ''

        self.load_afinn(afinn_filename, debug_mode)


        for doc in list_of_ngrams:
            doc_sentiment = 0
            text = self.pre_process(doc)
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
            if doc_sentiment > self.max_sentiment:
                self.max_sentiment = doc_sentiment
                self.max_text = text
            if doc_sentiment < self.min_sentiment:
                self.min_sentiment = doc_sentiment
                self.min_text = text
            if debug_mode == 2:
                print(str(doc_sentiment))

        self.__calc_term_total_sent()


    def __calc_term_total_sent(self, debug_mode=0):

        self.logger.info('calculating term total sent')
        # sum up the sentiment over all words
        term_total_sentiment = 0
        for word in self.total_sentiment_by_word:
            if word in self.scores_by_term:
                term_total_sentiment = term_total_sentiment + self.total_sentiment_by_word[word]

        # sum up contributions over all terms
        sum_key_sent_contibution = 0.0
        for word in self.scores_by_term:
            if word in self.counts_by_term:
                sum_key_sent_contibution = sum_key_sent_contibution + (
                    self.scores_by_term[word] * self.counts_by_term[word])

        if sum_key_sent_contibution != 0:
            self.ratio = term_total_sentiment / float(sum_key_sent_contibution)
        else:
            self.logger.warning('sum_key_sent_contibution == 0. Why?')
            self.ratio = 0


        if debug_mode == 2:
            print('Total Dict Sentiment ', str(sum_key_sent_contibution))
            print('Total File Sentiment ', str(term_total_sentiment))
            print('Ratio ', str(self.ratio))

        if debug_mode == 2:
            for word in self.scores_by_term:
                if word in self.total_sentiment_by_word:
                    print(word.encode('utf-8'),
                          str(self.scores_by_term[word]) + ' ' + str(self.total_sentiment_by_word[word]),
                          str(self.total_sentiment_by_word[word] / float(self.ratio)))

        # print the outputs, each word, and the scaled amount it might contribute to the sentiment if it were in the doctionary.
        if debug_mode >= 1:
            for word in self.total_sentiment_by_word:
                print(word, str(self.total_sentiment_by_word[word] / float(self.ratio)))

    def get_doc_sentiment(self, text, debug_mode=0):
        doc_sentiment = 0
        text = self.pre_process(text)
        words = text.split()
        for word in words:
            if word in self.total_sentiment_by_word:
                doc_sentiment = doc_sentiment + self.total_sentiment_by_word[word]
        if debug_mode >= 2:
            print('raw doc sentiment', doc_sentiment)
            print('doc_sentiment', str(doc_sentiment / self.ratio))
            print('norm tweet_sentiment', (doc_sentiment / self.ratio / len(words)))
        return doc_sentiment / self.ratio / len(words)


    def get_doc_simple_sentiment(self, text):
        doc_sentiment = 0
        text = self.pre_process(text)
        words = text.split()
        for word in words:
            if word in self.scores_by_term:
                doc_sentiment = doc_sentiment + self.scores_by_term[word]
        if len(words) == 0:
            return 0
        else:
            t = doc_sentiment / len(words)
            t = t + 5.0
            t = t / 10.0
            return t



    def get_doc_prop_pos_prob_neg(self, text):
        count_positive = 0
        count_negative = 0
        text = self.pre_process(text)
        words = text.split()
        if len(words) == 0:
            print('zero length ['+text+']')
            return 0.0, 0.0
        for word in words:
            if word in self.scores_by_term:
                score = self.scores_by_term[word]
                if score >= 0:
                    count_positive += 1
                else:
                    count_negative += 1
        return count_positive / len(words), count_negative / len(words)