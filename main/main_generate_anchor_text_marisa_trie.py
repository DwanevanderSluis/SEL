import marisa_trie
import gzip
import json
import logging
import pickle
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset
# import pprint

import gc
import sys


class ExtractAnchorText:
    def __init__(self):
        # Set up logging
        self.wikiDS = WikipediaDataset()
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

    # take about 3.5 hours to run
    # link type BODY or LINK
    def create_anchor_text_marisa_trie(self, case_insensitive=True, link_type='BODY'):
        wikititle_trie = self.wikiDS.get_wikititle_case_insensitive_marisa_trie()

        input_file = gzip.open("E:\\tmp\\" + 'wikipedia-dump.json.gz', 'rt', encoding='utf-8')

        unique_text = []
        unique_wids = []
        unique_text_set = set()

        fname_prefix = self.get_intermediate_path() + 'text_marisa_trie.' + link_type.lower() + '.'
        if case_insensitive:
            fname_prefix = fname_prefix + 'case_insensitive.'

        count = 1
        cache_hits = 0
        cache_misses = 0
        total_link_count = 0
        specific_link_count = 0
        line = ''
        while count < 25000000 and line is not None:  # TODO check termination and remove magic number
            log_progress = (count < 50000 and count % 1000 == 0)
            if log_progress:
                self.logger.info('starting gc ')
                gc.collect()  # have no real reason to think this is needed or will help the memory issue
                self.logger.info(
                    '%d lines processed. links processed = %d, total_links = %d, percentage=%f, cache_hits = %d, '
                    'cache_misses = %d',
                    count, specific_link_count, total_link_count, (specific_link_count / float(total_link_count)),
                    cache_hits, cache_misses)

            save_progress = count % 1000000 == 0 or count == 10
            if save_progress:
                self.logger.info('starting gc ')
                gc.collect()  # have no real reason to think this is needed or will help the memory issue
                self.logger.info(
                    "%d lines processed. links processed = %d, total_links = %d, percentage=%f, "
                    "cache_hits = %d, cache_misses = %d",
                    count, specific_link_count, total_link_count, (specific_link_count / float(total_link_count)),
                    cache_hits, cache_misses)
                marisa_trie_filename = fname_prefix + str(count) + '.pickle'
                # t = marisa_trie.Trie(keys)
                # see http://marisa-trie.readthedocs.io/en/latest/tutorial.html
                fmt = "<Lb"
                # one long unsign 32 bit integer.
                #  see https://docs.python.org/3/library/struct.html#format-strings
                t2 = marisa_trie.RecordTrie(fmt, zip(unique_text, unique_wids))
                self.logger.info('about to save to %s', marisa_trie_filename)
                with open(marisa_trie_filename, 'wb') as handle:
                    pickle.dump(t2, handle, protocol=pickle.HIGHEST_PROTOCOL)
                self.logger.info('written  %s', marisa_trie_filename)

            line = input_file.readline()
            if line is not None and line != '':
                data = json.loads(line)
                # wikititle has underscores, 'title' has spaces
                # wid = data['wid']
                # title = data['title']
                # wikititle = data['wikiTitle']

                if 'links' in data:
                    links = data['links']
                    # pprint.pprint(links)
                    for link in links:
                        total_link_count += 1
                        if 'anchor' in link:
                            anchor_text = link['anchor']  # text
                            wikititle = link['id']  # text - matches wikititle
                            link_type = link['type']
                            if link_type == link_type:
                                specific_link_count += 1
                                if case_insensitive:
                                    wikititle = wikititle.lower()
                                    anchor_text = anchor_text.lower()

                                if wikititle in wikititle_trie:
                                    value_list = wikititle_trie[wikititle]
                                    curid = value_list[0][0]

                                    # if anchor_text not in unique_text_set:
                                    unique_text_set.add(anchor_text)
                                    unique_text.append(anchor_text)
                                    unique_wids.append((curid, 0))
                                    cache_hits += 1

                                else:
                                    # TODO change the below to a warning
                                    self.logger.debug('wikititle %s not found in curid_by_wikititle_trie', wikititle)
                                    cache_misses += 1
            else:
                break

            count += 1

        self.logger.info('%d lines processed', count)
        marisa_trie_filename = fname_prefix + str(count) + '.pickle'
        # t = marisa_trie.Trie(keys)
        # see http://marisa-trie.readthedocs.io/en/latest/tutorial.html
        fmt = "<Lb"
        # one long unsign 32 bit integer.
        #  see https://docs.python.org/3/library/struct.html#format-strings
        t2 = marisa_trie.RecordTrie(fmt, zip(unique_text, unique_wids))
        self.logger.info('about to save to %s', marisa_trie_filename)
        with open(marisa_trie_filename, 'wb') as handle:
            pickle.dump(t2, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info('written  %s', marisa_trie_filename)

        # trie.items("fo")   give all this thing below this
        # trie["foo"] returns the key of this item
        # key in trie2 # returns true / false


if __name__ == "__main__":
    ds = ExtractAnchorText()
    ds.create_anchor_text_marisa_trie()
