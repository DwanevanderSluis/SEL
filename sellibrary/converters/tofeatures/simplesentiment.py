import numpy as np


class SimpleSentiment:

    def __init__(self, sentiment_processor, n_gram_length = 20):
        self.sentiment_processor = sentiment_processor
        self.half_n_gram_length = int(n_gram_length/2.0)

    def get_feature_list(self, sentiment_processor, phrase):
        sent = sentiment_processor.get_doc_simple_sentiment(phrase)
        feture_list = [sent]
        feture_list.extend(sentiment_processor.get_doc_prop_pos_prob_neg(phrase))
        return feture_list

    # we want this interface the same on all tofeature converters
    # returns a dictionary of feature lists keyed by entity id
    #
    # Features
    #      sentiment_simple_feature_names = ['title_total_sentiment_dv_wc','title_pos_words_div_word_count','title_neg_words_div_word_count',
    #                                  'body_total_sentiment_dv_wc','body_pos_words_div_word_count','body_neg_words_div_word_count']

    def get_features(self, body, body_entities,
                          title, title_entities):

        features_by_entity_id = {}

        # for title
        title_entity_id_set = set()
        title_count_by_entity_id = {}
        title_features_by_entity_id = {}
        for e in title_entities:
            title_entity_id_set.add(e.entity_id)
            title_count_by_entity_id[e.entity_id] = 0
        for e in title_entities:
            n_gram = self.sentiment_processor.get_ngram(title, self.half_n_gram_length, e.start_char, e.end_char)
            feature_list = self.get_feature_list(self.sentiment_processor, n_gram)
            if e.entity_id in title_features_by_entity_id:
                title_features_by_entity_id[e.entity_id] = title_features_by_entity_id[e.entity_id] + np.array(
                    feature_list)
                title_count_by_entity_id[e.entity_id] = 1
            else:
                title_features_by_entity_id[e.entity_id] = np.array(feature_list)
                title_count_by_entity_id[e.entity_id] += 1

        # for body
        body_entity_id_set = set()
        body_count_by_entity_id = {}
        body_features_by_entity_id = {}
        for e in body_entities:
            body_entity_id_set.add(e.entity_id)
            body_count_by_entity_id[e.entity_id] = 0
        for e in body_entities:
            n_gram = self.sentiment_processor.get_ngram(body, self.half_n_gram_length, e.start_char, e.end_char)
            feature_list = self.get_feature_list(self.sentiment_processor, n_gram)
            if e.entity_id in body_features_by_entity_id:
                body_features_by_entity_id[e.entity_id] = body_features_by_entity_id[e.entity_id] + np.array(
                    feature_list)
                body_count_by_entity_id[e.entity_id] = 1
            else:
                body_features_by_entity_id[e.entity_id] = np.array(feature_list)
                body_count_by_entity_id[e.entity_id] += 1

        # join together
        for entity_id in body_entity_id_set.union(title_entity_id_set):
            if entity_id in title_features_by_entity_id:
                title_features = title_features_by_entity_id[entity_id]
                title_features = title_features / title_count_by_entity_id[entity_id]
            else:
                title_features = np.zeros([1, 3])[0]
                title_features[0] = 0.5


            if entity_id in body_features_by_entity_id:
                body_features = body_features_by_entity_id[entity_id]
                body_features = body_features / body_count_by_entity_id[entity_id]
            else:
                body_features = np.zeros([1, 3])[0]
                body_features[0] = 0.5

            f = list(title_features) + list(body_features)
            features_by_entity_id[entity_id] = f
        return features_by_entity_id
