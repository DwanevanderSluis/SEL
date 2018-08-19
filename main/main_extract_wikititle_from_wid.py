import logging
import pickle

from sellibrary.wiki.wikipedia_datasets import WikipediaDataset
from sel.file_locations import FileLocations

# dense to sparse

# set up logging
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    ds = WikipediaDataset()
    wikititle_marisa_trie = ds.get_wikititle_case_insensitive_marisa_trie()
    logger.info('Creating dictionary')
    wikititle_by_id = {}
    for k in wikititle_marisa_trie.keys():
        wid = wikititle_marisa_trie.get(k)[0][0]
        wikititle_by_id[wid] = k

    logger.info('complete')

    output_filename = FileLocations.get_dropbox_wikipedia_path() + 'wikititle_by_id.pickle'
    logger.info('About to write %s', output_filename)
    with open(output_filename, 'wb') as handle:
        pickle.dump(wikititle_by_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info('file written = %s', output_filename)
