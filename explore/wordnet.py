from nltk.corpus import wordnet as wn


def explore_word(word1,word2):
    synonyms = []
    antonyms = []

    for syn in wn.synsets(word1):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    print(set(synonyms))
    print(set(antonyms))


explore_word('good','bad')
