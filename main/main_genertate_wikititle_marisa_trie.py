import marisa_trie
import gzip
import json
import logging
import os
import pickle

import gc
import sys

from sellibrary.wiki.wikipedia_datasets import WikipediaDataset

class ExtractWikiTitle:
    def __init__(self):
        # Set up logging
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(handler)
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)

    def get_intermediate_path(self):
        if sys.platform == 'win32':
            path = 'C:\\temp\\'
        else:
            path = '/Users/dsluis/Data/intermediate/'
        # self.logger.info(path)
        return path


    # take about 1.5 hours to run
    def cretae_wikititle_marisa_trie(self, case_insensitive=True):
        input_file = gzip.open("E:\\tmp\\" + 'wikipedia-dump.json.gz', 'rt', encoding='utf-8')

        unique_wiki_titles = []
        unique_wiki_wids = []
        unique_wiki_title_set = set()

        fname_prefix = self.get_intermediate_path()+'wikititle_marisa_trie.'
        if case_insensitive:
            fname_prefix = fname_prefix + 'case_insensitive.'


        count = 1
        line = ''
        while count < 25000000 and line is not None:  # TODO check termination and remove magic number
            log_progress = count < 50000 and count % 10000 == 0
            if log_progress:
                self.logger.info('starting gc ')
                gc.collect()  # have no real reason to think this is needed or will help the memory issue
                self.logger.info('%d lines processed', count)

            save_progress = count % 1000000 == 0 or count == 10
            if save_progress:
                self.logger.info('%d lines processed', count)
                marisa_trie_filename = fname_prefix+str(count)+'.pickle'
                # t = marisa_trie.Trie(keys)
                # see http://marisa-trie.readthedocs.io/en/latest/tutorial.html
                fmt = "<Lb"  # one long unsign 32 bit integer. # see https://docs.python.org/3/library/struct.html#format-strings
                t2 = marisa_trie.RecordTrie(fmt, zip(unique_wiki_titles, unique_wiki_wids))
                self.logger.info('about to save to %s', marisa_trie_filename)
                with open(marisa_trie_filename, 'wb') as handle:
                    pickle.dump(t2, handle, protocol=pickle.HIGHEST_PROTOCOL)
                self.logger.info('written  %s', marisa_trie_filename)

            line = input_file.readline()
            if  line is not None and line != '':
                data = json.loads(line)
                # pprint.pprint(data)
                if case_insensitive:
                    wikititle = data['wikiTitle'].lower()
                else:
                    wikititle = data['wikiTitle']

                wid = data['wid']
                if (wikititle not in unique_wiki_title_set):
                    unique_wiki_titles.append(wikititle)
                    unique_wiki_wids.append((wid,0))
                    unique_wiki_title_set.add(wikititle)
            else:
                break

            count += 1

        self.logger.info('%d lines processed', count)
        marisa_trie_filename = fname_prefix + str(count) + '.pickle'
        # t = marisa_trie.Trie(keys)
        # see http://marisa-trie.readthedocs.io/en/latest/tutorial.html
        fmt = "<Lb"  # one long unsign 32 bit integer. # see https://docs.python.org/3/library/struct.html#format-strings
        t2 = marisa_trie.RecordTrie(fmt, zip(unique_wiki_titles, unique_wiki_wids))
        self.logger.info('about to save to %s', marisa_trie_filename)
        with open(marisa_trie_filename, 'wb') as handle:
            pickle.dump(t2, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info('written  %s', marisa_trie_filename)

        # trie.items("fo")   give all this thing below this
        # trie["foo"] returns the key of this item
        # key in trie2 # returns true / false



    # take about 1.5 hours to run
    def check_for_wikititle_collisions(self, case_insensitive=True):
        input_file = gzip.open("E:\\tmp\\" + 'wikipedia-dump.json.gz', 'rt', encoding='utf-8')

        wd = WikipediaDataset()
        wikititle_mt = wd.get_wikititle_case_insensitive_marisa_trie()
        wikititle_id_by_id = {}
        fname_prefix = self.get_intermediate_path()+'wikititle_id_by_id.'
        if case_insensitive:
            fname_prefix = fname_prefix + 'case_insensitive.'

        count = 1
        collision_count = 1
        line = ''

        duplicate_ids_by_wikititle = {}

        while count < 25000000 and line is not None:  # TODO check termination and remove magic number
            log_progress = count < 50000 and count % 10000 == 0
            if log_progress:
                self.logger.info('starting gc ')
                gc.collect()  # have no real reason to think this is needed or will help the memory issue
                self.logger.info('%d lines processed', count)

            save_progress = count % 1000000 == 0 or count == 10
            if save_progress:
                self.logger.info('%d lines processed', count)
                wikititle_by_id_filename = fname_prefix + str(count) + '.pickle'
                self.logger.info('about to save to %s', wikititle_by_id_filename)
                with open(wikititle_by_id_filename, 'wb') as handle:
                    pickle.dump(wikititle_id_by_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
                self.logger.info('written  %s', wikititle_by_id_filename)

            line = input_file.readline()
            if  line is not None and line != '':
                data = json.loads(line)
                # pprint.pprint(data)
                if case_insensitive:
                    wikititle = data['wikiTitle'].lower()
                else:
                    wikititle = data['wikiTitle']

                wt_id = wikititle_mt[wikititle]
                wid = data['wid']
                wikititle_id_by_id[wid] = wt_id

            else:
                break

            count += 1

        self.logger.info('%d lines processed', count)
        wikititle_by_id_filename = fname_prefix + str(count) + '.pickle'
        self.logger.info('about to save to %s', wikititle_by_id_filename)
        with open(wikititle_by_id_filename, 'wb') as handle:
            pickle.dump(wikititle_id_by_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info('written  %s', wikititle_by_id_filename)

            # trie.items("fo")   give all this thing below this
        # trie["foo"] returns the key of this item
        # key in trie2 # returns true / false




if __name__ == "__main__":
    ds = ExtractWikiTitle()
    #ds.cretae_wikititle_marisa_trie()
    ds.check_for_wikititle_collisions()


