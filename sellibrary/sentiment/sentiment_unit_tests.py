
from sellibrary.sentiment.sentiment import SentimentProcessor


def unit_test1():
    afinn_filename = '../resources/AFINN-111.txt'
    max_diff = 0.0001

    sp = SentimentProcessor(afinn_filename)
    sp.cal_term_weight_on_full_corpus(['abhor abandoned archaeological',' happy apple archaeological '])
    sp.__calc_term_total_sent()

    print('happy has a doc normalised value of:',sp.get_doc_sentiment('happy'))
    print('archaeological has a doc normalised value of:',sp.get_doc_sentiment('archaeological'))
    assert(sp.get_doc_sentiment('archaeolog') < 0) # with 2 neg, and 1 +ve word
    assert(abs(sp.get_doc_sentiment('happy') - 0.8571428571428571)<max_diff)
    assert(sp.get_doc_sentiment('archaeolog') < 0) # with 2 neg, and 1 +ve word


def unit_test_2():
    afinn_filename = '../resources/AFINN-111.txt'
    sp = SentimentProcessor(afinn_filename)
    x = sp.get_ngram('a b c d e f g h i j k',3, 8, 9)
    assert(x == 'b c d e f g h')
    x = sp.get_ngram('a b c d e f g h i j k', 4, 8, 9)
    assert(x == 'a b c d e f g h i')
    x = sp.get_ngram('a b c d e f g h i j k', 6, 8, 9)
    assert(x == 'a b c d e f g h i j k')




if __name__ == "__main__":
    unit_test_2()