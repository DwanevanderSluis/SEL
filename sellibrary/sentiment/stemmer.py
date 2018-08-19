from nltk.stem.porter import PorterStemmer


class Stemmer():

    def __init__(self, stemmer = PorterStemmer()):
        self.stemmer = stemmer

    def get_stemmed(self, list_of_words, make_unique_and_sort = True):
        stemmed = []
        for w in list_of_words:
            w = self.stemmer.stem(w)  # use Porter's stemmer
            if len(w) < 3:  # remove short tokens
                continue
            stemmed.append(w)
        if make_unique_and_sort:
            unique = list(set(stemmed))
            unique.sort()
            return unique
        else:
            return stemmed