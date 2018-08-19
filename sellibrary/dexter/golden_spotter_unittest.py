from sellibrary.dexter.golden_spotter import GoldenSpotter
import logging


# set up logging
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)

doc_body = "Iranian representatives say negotiations with Europe on its nuclear program are in the final stages. Iran’s foreign minister, Kamal Kharazi, told state television Saturday Iranian negotiators have given their final response to a European Union proposal to suspend Iran’s uranium enrichment program. He said it is now up to the Europeans to decide whether or not to accept their decision. Iran and the European Union’s big three powers - Britain, Germany, and France - have been negotiating a deal under which Tehran would agree to freeze sensitive nuclear work to avoid possible U.N. Security Council sanctions. U.S. Secretary of State Colin Powell, says that Iran’s nuclear program is intended to make nuclear weapons. Iran authorities have insisted that their nuclear ambitions are limited to generating electricity from atomic energy plants, not making bombs. Critics of the position of the United States point to Israel’s nuclear program. Israel maintains a policy of nuclear ambiguity, but is widely believed to possess at least 82 nuclear weapons. The program has not been condemned by the United States. "
doc_title = "Iran close to decision on nuclear program"



full_json_doc = "{\"docId\": 1, \"title\": \"Iran close to decision on nuclear program\", \"saliency\": [{\"entityid\": 9282173, \"score\": 1.0}, {\"entityid\": 186801, \"score\": 1.67}, {\"entityid\": 721807, \"score\": 3.0}, {\"entityid\": 31717, \"score\": 1.0}, {\"entityid\": 5843419, \"score\": 1.0}, {\"entityid\": 21785, \"score\": 1.67}, {\"entityid\": 6984, \"score\": 1.0}, {\"entityid\": 57654, \"score\": 1.33}, {\"entityid\": 14653, \"score\": 3.0}, {\"entityid\": 31956, \"score\": 1.67}, {\"entityid\": 1166971, \"score\": 1.67}, {\"entityid\": 9239, \"score\": 2.33}, {\"entityid\": 11867, \"score\": 1.0}, {\"entityid\": 9317, \"score\": 2.33}, {\"entityid\": 32293, \"score\": 1.0}, {\"entityid\": 31743, \"score\": 2.0}, {\"entityid\": 3434750, \"score\": 1.0}], \"timestamp\": \"2013-08-21T16:07:41Z\", \"wikiTitle\": \"Iran_close_to_decision_on_nuclear_program\", \"wikinewsId\": 779, \"document\": [{\"name\": \"headline\", \"value\": \"Iran close to decision on nuclear program\"}, {\"name\": \"dateline\", \"value\": \"November 13, 2004\"}, {\"name\": \"body_par_000\", \"value\": \"Iranian representatives say negotiations with Europe on its nuclear program are in the final stages.\"}, {\"name\": \"body_par_001\", \"value\": \"Iran's foreign minister, Kamal Kharazi, told state television Saturday Iranian negotiators have given their final response to a European Union proposal to suspend Iran's uranium enrichment program. He said it is now up to the Europeans to decide whether or not to accept their decision.\"}, {\"name\": \"body_par_002\", \"value\": \"Iran and the European Union's big three powers &mdash; Britain, Germany, and France &mdash; have been negotiating a deal under which Tehran would agree to freeze sensitive nuclear work to avoid possible U.N. Security Council sanctions.\"}, {\"name\": \"body_par_003\", \"value\": \"U.S. Secretary of State Colin Powell, says that Iran's nuclear program is intended to make nuclear weapons. Iran authorities have insisted that their nuclear ambitions are limited to generating electricity from atomic energy plants, not making bombs.\"}, {\"name\": \"body_par_004\", \"value\": \"Critics of the position of the United States point to Israel's nuclear program. Israel maintains a policy of nuclear ambiguity, but is widely believed to possess at least 82 nuclear weapons. The program has not been condemned by the United States.\"}]}"
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset

wikipediaDataset = WikipediaDataset()

gs = GoldenSpotter([full_json_doc],wikipediaDataset)

def test_1():
    assert len(gs.get_spots_in_text('', 'israel', 3)) == 0
    assert len(gs.get_spots_in_text('Israel israeL', 'israel', 3)) == 2


def test_2():
    l = gs.get_entity_candidates(doc_body,1)
    logger.info('list of spots: %d',len(l))
    logger.info('list of spots: %s',str(l))

    assert len(l) == 74 # 57


def test_3():
    l = gs.get_entity_candidates(doc_body,1)
    logger.info('list of spots: %s',str(l))



if __name__ == "__main__":
    test_2()
