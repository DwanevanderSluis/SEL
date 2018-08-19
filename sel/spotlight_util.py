import logging
import pprint

import requests

from sellibrary.spot import Spot
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset


class SpotlightUtil:
    def __init__(self):
        # Set up logging
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(handler)
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)

        # instance variables
        self.wiki_ds = WikipediaDataset()

    def hit_spotlight_web_retun_text(self,doc_text, confidence):

        if doc_text.find(">") > -1 or doc_text.find(">") > -1:
            self.logger.info('contains html characters')
            pprint.pprint(doc_text)
            doc_text = doc_text.replace(">", " ")
            doc_text = doc_text.replace("<", " ")

        # for documentation see https://www.dbpedia-spotlight.org/api
        # confidence
        # http://api.dbpedia-spotlight.org/en/annotate?confidence=0.5&text= happy Hippo
        link = "http://api.dbpedia-spotlight.org/en/annotate?confidence="+str(confidence)+"&text="+doc_text

        for i in range(3): # max 3 attempts
            try:
                #headers = {'Accept': 'application/json'} # json would be easier to parse
                headers = {'Accept': 'text/html'}
                f = requests.get(link, headers=headers)
                if f.ok:
                    pprint.pprint(f.text)
                    i_start = f.text.find("<body>")
                    i_end = f.text.find("</body>")
                    if i_end > -1 and i_start > -1:
                        html = f.text[(i_start + 6):(i_end)]
                    else:
                        html = ""
                    self.logger.info("Spotlight returned "+str(len(html))+" bytes")
                    return html
                else:
                    self.logger.info('obtained error code %d %s',f.status_code, f.reason)
            except:
                self.logger.info('error', exc_info=True)
        raise EnvironmentError('Could not complete web request')



    def my_pp(self,text):
        text = text.replace("\n", " ")
        text = text.replace("<a","\na<")
        text = text.replace("</a>","\n</a>\n")
        return text


    def get_wid_from_link_text(self, link):
        wid = -1
        t = "href=\"http://dbpedia.org/resource/"
        if link.find(t) >= 0:
            link_key_start = link.find(t) + len(t)
            link_key_end = link.find("\"", link_key_start)
            link_key = link[link_key_start:link_key_end]
            # print('link key:' + link_key)
            t = self.wiki_ds.get_wikititle_case_insensitive_marisa_trie()
            if link_key.lower() in t:
                values = t[link_key.lower()]
                wid = values[0][0]
                # print('wid:' + str(wid))
                # print('loc:',start_char,'-',end_char)
        return wid



    def post_process_html(self, html):
        #
        # There is a chance that this routine changes the text as it extracts the links from it
        # therefore the text is returned as well.
        #
        text = html.strip()

        if text.startswith("<div>"):
            text = text[5:]
        if text.endswith("</div>"):
            text = text[:-6]
        spots = []

        while text.find("<a",0) >= 0:

            start_char = text.find("<a")

            end_char = text.find("</a>")
            if end_char == -1:
                break
            next_link_start = text.find("<a")
            link = '-'
            while next_link_start < end_char and next_link_start != -1 and end_char != -1 and link != '':
                # we have interleaved links
                end_of_interleaved_link = text.find(">") + 1
                link = text[next_link_start:end_of_interleaved_link]

                #find piece to remove
                end_of_piece = end_of_interleaved_link
                if text.find("<a",end_of_interleaved_link) != -1 and \
                    text.find("<a",end_of_interleaved_link) < text.find("</a>",end_of_interleaved_link):
                    end_of_piece = text.find("<a",end_of_interleaved_link)

                text = text[0:next_link_start] + text[end_of_piece:]
                full_link = text[next_link_start:end_of_piece]

                if full_link.find("</a>") > -1 and full_link.find("</a>") < full_link.find(">"):
                    # this is a 'normal' link
                    anchor_text = full_link[:full_link.find("</a>")]
                else:
                    anchor_text = full_link[full_link.find(">")+1:]
                # extract link key
                wid = self.get_wid_from_link_text(link)
                s = Spot(wid, start_char, start_char + len(anchor_text), anchor_text)
                spots.append(s)
                print(wid, start_char, start_char + len(anchor_text), anchor_text)
                next_link_start = text.find("<a")
                end_char = text.find("</a>")

            # Remove the </a>
            end_char = text.find("</a>")
            if end_char > -1:
                text = text[0:end_char] + text[end_char+4:]
        return text, spots

    def hit_spotlight_return_spot_list(self, text, confidence):
        body_html = self.hit_spotlight_web_retun_text(text, confidence)
        processed_text, spot_list = self.post_process_html(body_html)
        return processed_text, spot_list







