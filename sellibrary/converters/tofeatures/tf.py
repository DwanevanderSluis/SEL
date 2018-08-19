
class DocToTermFreqConverter:

    def __init__(self):
        self.tf_feature_names = ['term_frequency_in_body_and_title',
                                 'term_frequency_in_title',
                                 'term_frequency_in_body'
                                 ]


    # we want this interface the same on all tofeature converters
    # returns a dictionary of feature lists keyed by entity id

    def get_feature_names(self):
        return self.tf_feature_names

    def get_features(self, body, body_entities,
                          title, title_entities):

        features_by_entity_id = {}
        title_count_by_entity_id = {}
        body_count_by_entity_id = {}
        entity_id_set = set()

        # count the entity occurrences across title and body
        for e in title_entities:
            entity_id_set.add(e.entity_id)
            if e.entity_id not in title_count_by_entity_id:
                title_count_by_entity_id[e.entity_id] = 1
            else:
                title_count_by_entity_id[e.entity_id] += 1

        for e in body_entities:
            entity_id_set.add(e.entity_id)
            if e.entity_id not in body_count_by_entity_id:
                body_count_by_entity_id[e.entity_id] = 1
            else:
                body_count_by_entity_id[e.entity_id] += 1

        # convert to list of (1) features
        entities_in_body = sum(list(body_count_by_entity_id.values()))
        entities_in_title = sum(list(title_count_by_entity_id.values()))

        for entity_id in entity_id_set:
            body_count = 0
            title_count = 0
            if entity_id in body_count_by_entity_id:
                body_count = body_count_by_entity_id[entity_id]
            if entity_id in title_count_by_entity_id:
                title_count = title_count_by_entity_id[entity_id]
            doc_count = title_count + body_count
            if (entities_in_body+entities_in_title) == 0:
                doc_v = 0
            else:
                doc_v = doc_count / (entities_in_body+entities_in_title)
            if entities_in_title == 0:
                title_v = 0.0
            else:
                title_v = title_count / entities_in_title
            if entities_in_body == 0:
                body_v = 0.0
            else:
                body_v = body_count / entities_in_body
            features_by_entity_id[entity_id] = [doc_v, title_v, body_v]

        return features_by_entity_id
