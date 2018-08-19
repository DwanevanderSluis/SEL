import logging

from sellibrary.wiki.wikipedia_datasets import WikipediaDataset

# set up logging
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    ds = WikipediaDataset()
    # this requires extract_curid_by_wikititle_trie to have been run first
    ds.extract_graph_from_compressed()
