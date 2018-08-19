import logging

from sellibrary.spot import Spot
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset


class WikipediaSpotter:

    def __init__(self):
        # set up logging
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(handler)
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        # set up instance variables
        self.wikiDS = WikipediaDataset()

    @staticmethod
    def find_nth(haystack, needle, n):
        start = haystack.find(needle)
        while start >= 0 and n > 1:
            start = haystack.find(needle, start + len(needle))
            n -= 1
        if start == -1:  # if not found return all
            start = len(haystack)
        return start

    def get_entity_candidates_using_wikititle(self, text):

        candidates = []
        wikititle_trie = self.wikiDS.get_wikititle_case_insensitive_marisa_trie()

        # trie.items("fo")   give all this thing below this
        # trie["foo"] returns the key of this item
        # key in trie2 # returns true / false

        i = 0
        jump = 1
        while i < len(text):
            for num_words in range(1, 5):  # 4 word max
                nth = self.find_nth(text[i:], ' ', num_words)
                if num_words == 1:
                    jump = max(1, nth + 1)
                candidate = text[i:(i + nth)]
                candidate = candidate.lower()
                candidate = candidate.replace('.', ' ')
                candidate = candidate.strip()
                candidate = candidate.replace(' ', '_')
                if candidate in wikititle_trie:
                    t = wikititle_trie[candidate]
                    cid = t[0][0]
                    value_list = [i, (i + nth), candidate, cid]
                    candidates.append(value_list)
                    self.logger.info(value_list)
            i += jump

    def get_entity_candidates(self, text):

        candidates = []
        text_trie = self.wikiDS.get_anchor_text_case_insensitive_marisa_trie()

        # trie.items("fo")   give all this thing below this
        # trie["foo"] returns the key of this item
        # key in trie2 # returns true / false

        i = 0
        jump = 1
        while i < len(text):
            for num_words in range(1, 5):  # 4 word max
                nth = self.find_nth(text[i:], ' ', num_words)
                if num_words == 1:
                    jump = max(1, nth + 1)
                candidate = text[i:(i + nth)]
                candidate = candidate.lower()
                candidate = candidate.strip()
                if candidate in text_trie:
                    t = text_trie[candidate]
                    cid = t[0][0]
                    s = Spot(cid, i, (i+nth), text[i:(i+nth)])
                    candidates.append(s)

                if len(candidate) > 1 and candidate[-1] == '.': # special case remove trailing full stop
                    candidate = candidate[0:-1]
                    if candidate in text_trie:
                        t = text_trie[candidate]
                        cid = t[0][0]
                        s = Spot(cid, i, (i+nth-1), text[i:(i+nth-1)])
                        candidates.append(s)

                if i+jump >= len(text):
                    break
            i += jump

        return candidates


if __name__ == "__main__":
    ws = WikipediaSpotter()
    example = "Iranian representatives say negotiations with Europe on its nuclear program are in the final stages. Iran's foreign minister, Kamal Kharazi, told state television Saturday Iranian negotiators have given their final response to a European Union proposal to suspend Iran's uranium enrichment program. He said it is now up to the Europeans to decide whether or not to accept their decision. Iran and the European Union's big three powers; Britain, Germany, and France; have been negotiating a deal under which Tehran would agree to freeze sensitive nuclear work to avoid possible U.N. Security Council sanctions. U.S. Secretary of State Colin Powell, says that Iran's nuclear program is intended to make nuclear weapons. Iran authorities have insisted that their nuclear ambitions are limited to generating electricity from atomic energy plants, not making bombs. Critics of the position of the United States point to Israel's nuclear program. Israel maintains a policy of nuclear ambiguity, but is widely believed to possess at least 82 nuclear weapons. The program has not been condemned by the United States. Nuclear program of Iran. junk "
    #example = "United States. Nuclear program of Iran. junk "

    ws.get_anchor_text_spots(example)
