import logging

from sellibrary.wiki.wikipedia_datasets import WikipediaDataset

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
    curid = 399877

    in_degree = ds.get_entity_in_degree(curid)
    out_degree = ds.get_entity_out_degree(curid)
    degree = ds.get_entity_degree(curid)
    logger.info('degree %d', degree)
    logger.info('in degree %d', in_degree)
    logger.info('out degree %d', out_degree)

    assert(degree >= 54)
    assert(in_degree >= 9)
    assert(out_degree >= 45)
