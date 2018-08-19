import gzip
import json
import logging
import os
import pickle
import sys
import threading

import numpy as np
import scipy.sparse as sparse
from sellibrary.locations import FileLocations

class WikipediaDataset:

    # class variables - shared across all instances
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    # ___
    wikipedia_link_graph_sparse = None
    wikipedia_link_graph_sparse_csc = None
    wikipedia_link_graph_sparse_csr = None

    # ____
    wikititle_marisa_trie = None
    text_marisa_trie = None

    #_____
    wikititle_id_by_id = None

    wikititle_by_id = None


    def __init__(self):
        pass

    def get_dexter_dataset(self, path=None, filename='short.json'):
        if path is None:
            path = FileLocations.get_dropbox_wikipedia_path()
        with open(path + filename) as f:
            content = f.readlines()
        return content

    def extract_graph_from_compressed(self, wikititle_to_id_filename=None):
        self.logger.warning('running extract_graph_from_compressed().')
        self.logger.warning('[this takes about 2hr 20 min on Dwane\'s home machine]')
        input_file = gzip.open(FileLocations.get_dropbox_wikipedia_path() + 'wikipedia-dump.json.gz', 'rt', encoding='utf-8')
        if wikititle_to_id_filename is not None:
            fn = wikititle_to_id_filename
        else:
            fn = FileLocations.get_dropbox_wikipedia_path() + 'wikititle_marisa_trie.case_insensitive.15910478.pickle'
        self.logger.warning(' %s needs to be complete for these results to make most sense', fn)
        self.get_wikititle_case_insensitive_marisa_trie()

        count = 0
        line = '{}'
        from_list = []
        to_list = []
        value_list = []
        max_id = 0

        while count < 25000000 and line is not None and line != '':
            count += 1
            early_log = count <= 50000 and count % 10000 == 0
            late_log = count > 50000 and count % 1000000 == 0
            if early_log or late_log:
                self.logger.info('%d lines processed', count)
                output_filename = FileLocations.get_temp_path() + 'wikipedia_link_graph_sparse.deduped.' + str(count) + '.pickle'
                self.logger.info('saving file %s', output_filename)
                row = np.array(from_list)
                col = np.array(to_list)
                data = np.array(value_list)
                mtx = sparse.coo_matrix((data, (row, col)), shape=(max_id + 1, max_id + 1))
                self.logger.info('About to write %s', output_filename)
                with open(output_filename, 'wb') as handle:
                    pickle.dump(mtx, handle, protocol=pickle.HIGHEST_PROTOCOL)
                self.logger.info('file written = %s', output_filename)

            line = input_file.readline()
            if line != '':
                try:
                    data = json.loads(line)
                except json.decoder.JSONDecodeError as e:
                    self.logger.warning("type error decoding json: json = %s, error = %s", line, str(e))
                    break
                # pprint.pprint(data)
                if 'links' in data:
                    fid = data['wid']
                    if self.get_wikititle_id_from_id(fid)[0][0] != fid:
                        self.logger.info('%s -> %s ',fid,self.get_wikititle_id_from_id(fid)[0][0])
                        fid = self.get_wikititle_id_from_id(fid)[0][0]

                    if fid > max_id:
                        max_id = fid
                    for link in data['links']:
                        link_name = link['id']  # this is not numeric, has underscores, matches WikiTitle

                        if link_name in self.wikititle_marisa_trie:
                            link_list = self.wikititle_marisa_trie[link_name]
                            link_cid = link_list[0][0]

                            if link_cid > max_id:
                                max_id = link_cid

                            # d['type'] = link['type'] # do we care about link type? assuming no
                            from_list.append(fid)
                            to_list.append(link_cid)
                            value_list.append(1)

        self.logger.info('%d lines processed', count)
        output_filename = FileLocations.get_temp_path() + 'wikipedia_link_graph_sparse.deduped.' + str(count) + '.pickle'
        self.logger.info('saving file %s', output_filename)
        row = np.array(from_list)
        col = np.array(to_list)
        data = np.array(value_list)
        mtx = sparse.coo_matrix((data, (row, col)), shape=(max_id, max_id))
        self.logger.info('About to write %s', output_filename)
        with open(output_filename, 'wb') as handle:
            pickle.dump(mtx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info('file written = %s', output_filename)


    def synchronized_at_class_level(func):
        func.__lock__ = threading.Lock()
        def synced_func(*args, **kws):
            with func.__lock__:
                return func(*args, **kws)
        return synced_func

    @synchronized_at_class_level
    def load_wikipeadia_link_graph(self, link_graph_filename=None):
        if self.wikipedia_link_graph_sparse is None:
            if link_graph_filename is None:
                link_graph_filename = FileLocations.get_dropbox_wikipedia_path() + 'wikipedia_link_graph_sparse.deduped.15910478.pickle'
            if os.path.isfile(link_graph_filename):
                self.logger.info('loading wikipedia_link_graph_sparse from %s', link_graph_filename)
                with open(link_graph_filename, 'rb') as handle:
                    self.wikipedia_link_graph_sparse = pickle.load(handle)
                self.logger.info('loaded')

    def get_entity_in_degree(self, curid):
        self.load_wikipeadia_link_graph()  # loads the graph from disk if needed
        if self.wikipedia_link_graph_sparse.shape[1] <= curid:
            self.logger.warning('graph bounds exceeded returning 0')
            return 0 #
        in_degree = self.wikipedia_link_graph_sparse.getcol(curid).sum()
        return in_degree

    def get_entity_out_degree(self, curid):
        self.load_wikipeadia_link_graph()  # loads the graph from disk if needed
        if self.wikipedia_link_graph_sparse.shape[0] <= curid:
            self.logger.warning('graph bounds exceeded returning 0')
            return 0 #
        out_degree = self.wikipedia_link_graph_sparse.getrow(curid).sum()
        return out_degree

    def get_wikipedia_link_graph_sparse(self):
        return self.wikipedia_link_graph_sparse

    def get_entity_degree(self, curid):
        # this routine is slow, avoid calling if possible
        self.load_wikipeadia_link_graph()  # loads the graph from disk if needed
        degree = self.get_entity_in_degree(curid) + self.get_entity_out_degree(curid)
        return degree

    @synchronized_at_class_level
    def convert_link_graph_to_csr_and_csc(self):

        self.load_wikipeadia_link_graph()
        self.logger.info('converting to csr')
        csr = self.wikipedia_link_graph_sparse.tocsr()
        self.logger.info('converting to csc')
        csc = self.wikipedia_link_graph_sparse.tocsc()

        output_filename = FileLocations.get_dropbox_wikipedia_path() + 'wikipedia_link_graph_sparse_csr.deduped.15910478.pickle'
        self.logger.info('About to write %s', output_filename)
        with open(output_filename, 'wb') as handle:
            pickle.dump(csr, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info('file written = %s', output_filename)

        output_filename = FileLocations.get_dropbox_wikipedia_path() + 'wikipedia_link_graph_sparse_csc.deduped.15910478.pickle'
        self.logger.info('About to write %s', output_filename)
        with open(output_filename, 'wb') as handle:
            pickle.dump(csc, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info('file written = %s', output_filename)

    @synchronized_at_class_level
    def get_wikipedia_link_graph_sparse_csr(self):
        filename = FileLocations.get_dropbox_wikipedia_path() + 'wikipedia_link_graph_sparse_csr.deduped.15910478.pickle'
        self.logger.info('Loading %s', filename)
        with open(filename, 'rb') as handle:
            self.wikipedia_link_graph_sparse_csr = pickle.load(handle)
        self.logger.info('Loaded %s', filename)
        return self.wikipedia_link_graph_sparse_csr

    @synchronized_at_class_level
    def get_wikipedia_link_graph_sparse_csc(self):
        filename = FileLocations.get_dropbox_wikipedia_path() + 'wikipedia_link_graph_sparse_csc.deduped.15910478.pickle'
        self.logger.info('Loading %s', filename)
        with open(filename, 'rb') as handle:
            self.wikipedia_link_graph_sparse_csc = pickle.load(handle)
        self.logger.info('Loaded %s', filename)
        return self.wikipedia_link_graph_sparse_csc

    def get_links_to(self, entity_id):
        if self.wikipedia_link_graph_sparse_csc is None:
            self.get_wikipedia_link_graph_sparse_csc()
        col = self.wikipedia_link_graph_sparse_csc.getcol(entity_id)
        return (col.indices)

    def get_links_from(self, entity_id):
        if self.wikipedia_link_graph_sparse_csr is None:
            self.get_wikipedia_link_graph_sparse_csr()
        row = self.wikipedia_link_graph_sparse_csr.getrow(entity_id)
        return (row.indices)

    #______________________________

    @synchronized_at_class_level
    def load_wikititle_case_insensitive_marisa_trie(self, filename=None):
        if self.wikititle_marisa_trie is None:
            if filename is None:
                filename = FileLocations.get_dropbox_wikipedia_path() + 'wikititle_marisa_trie.case_insensitive.15910478.pickle'
            self.logger.info('Loading %s', filename)
            with open(filename, 'rb') as handle:
                self.wikititle_marisa_trie = pickle.load(handle)
            self.logger.info('Loaded %s', filename)

    def get_wikititle_case_insensitive_marisa_trie(self, filename=None):
        if self.wikititle_marisa_trie is None:
            self.load_wikititle_case_insensitive_marisa_trie(filename)
        return self.wikititle_marisa_trie

    def get_id_from_wiki_title(self, wiki_title):
        wiki_title = wiki_title.lower()
        id = self.get_wikititle_case_insensitive_marisa_trie()[wiki_title][0][0]
        id = self.get_wikititle_id_from_id(id)
        return id


    @synchronized_at_class_level
    def load_anchor_text_case_insensitive_marisa_trie(self, filename=None):
        if self.text_marisa_trie is None:
            if filename is None:
                filename = self.get_path() + 'text_marisa_trie.body.case_insensitive.15000000.pickle'
            self.logger.info('Loading %s', filename)
            with open(filename, 'rb') as handle:
                self.text_marisa_trie = pickle.load(handle)
            self.logger.info('Loaded %s', filename)

    def get_anchor_text_case_insensitive_marisa_trie(self, filename=None):
        if self.text_marisa_trie is None:
            self.load_anchor_text_case_insensitive_marisa_trie(self, filename)
        return self.text_marisa_trie

    # ______________________________


    @synchronized_at_class_level
    def load_wikititle_by_id(self, filename=None):
        if self.wikititle_by_id is None:
            if filename is None:
                filename = FileLocations.get_dropbox_wikipedia_path() + 'wikititle_by_id.pickle'
            self.logger.info('Loading %s', filename)
            with open(filename, 'rb') as handle:
                self.wikititle_by_id = pickle.load(handle)
            self.logger.info('Loaded %s', filename)

    def get_wikititle_by_id(self, filename=None):
        if self.wikititle_by_id is None:
            self.load_wikititle_by_id(filename)
        return self.wikititle_by_id


    #______________________________

    @synchronized_at_class_level
    def load_wikititle_id_by_id(self, filename=None):
        if self.wikititle_id_by_id is None:
            if filename is None:
                filename = FileLocations.get_dropbox_wikipedia_path() + 'wikititle_id_by_id.case_insensitive.15910478.pickle'
            self.logger.info('Loading %s', filename)
            with open(filename, 'rb') as handle:
                self.wikititle_id_by_id = pickle.load(handle)
            self.logger.info('Loaded %s', filename)

    def get_wikititle_id_from_id(self, entity_id, filename=None):
        if self.wikititle_id_by_id is None:
            self.load_wikititle_id_by_id(filename)
        if entity_id in self.wikititle_id_by_id:
            x = self.wikititle_id_by_id[entity_id]
            # self.logger.info('%s', x)
            return x[0][0]
        else:
            self.logger.debug('Key error %d not found in get_wikititle_id_from_id',entity_id)
            return entity_id


