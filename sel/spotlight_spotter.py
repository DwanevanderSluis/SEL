import hashlib
import logging
import os
import pickle
import time
from shutil import move

from sel.file_locations import FileLocations
from sel.spotlight_util import SpotlightUtil


class SpotlightCachingSpotter:
    # set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def __init__(self, run_multi_process_compatable = True):

        # set up instance variables
        self.spotlight_util = SpotlightUtil()
        self.cache = {}
        self.run_multi_process_compatable = run_multi_process_compatable
        self.last_web_request = 0
        self.number_requests = 0

    def get_entity_candidates(self, text, confidence):

        if self.run_multi_process_compatable or self.number_requests < 1:
            self.load_cache()  # always load the cache in case another process has updated it.

        self.number_requests += 1
        key = str(confidence) + text
        hash_key = hashlib.sha512(key.encode())
        hash_as_hex = hash_key.hexdigest()
        self.logger.debug(hash_as_hex)

        if hash_as_hex in self.cache:
            return self.cache[hash_as_hex]

        self.logger.info('Not in cache, hitting spotlight web service with confidence = %f \nkey=[%s]', confidence, key)
        time_since_last_request = time.time() - self.last_web_request
        if time_since_last_request < 1.0:
            self.logger.info('last requect was less than a second ago. sleeping for %f seconds',
                             1 - time_since_last_request)
            time.sleep(1 - time_since_last_request)

        self.last_web_request = time.time()
        processed_text, candidates_spot_list = self.spotlight_util.hit_spotlight_return_spot_list(text, confidence)
        self.cache[hash_as_hex] = candidates_spot_list
        self.save_cache()
        return candidates_spot_list

    def save_cache(self):
        output_filename = FileLocations.get_dropbox_intermediate_path() + 'spotlight_docs_list.pickle'
        temp_filename = FileLocations.get_dropbox_intermediate_path() + "spotlight_docs_list.tmp.2.pickle"
        lock_acquired = False
        while not lock_acquired:
            try:
                if not os.path.isfile(output_filename) and not os.path.isfile(temp_filename):
                    pass
                else:
                    move(output_filename, temp_filename)
            except:
                time.sleep(3)
            else:
                lock_acquired = True
                self.logger.info('lock acquired')
        # do your writing
        self.logger.info('About to write %s', temp_filename)
        with open(temp_filename, 'wb') as handle:
            pickle.dump(self.cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info('file written = %s', output_filename)

        # release file by moving it back
        move(temp_filename, output_filename)
        # lock_acquired = False
        self.logger.info('lock released')

    def load_cache(self):
        input_filename = FileLocations.get_dropbox_intermediate_path() + 'spotlight_docs_list.pickle'

        count = 0
        if count < 3 and not os.path.isfile(input_filename):
            # may be being written to by another process, wait
            time.sleep(1)

        if os.path.isfile(input_filename):
            self.logger.info('loading spotlight cache from %s', input_filename)
            with open(input_filename, 'rb') as handle:
                self.cache = pickle.load(handle)
            self.logger.info('%d items loaded', len(self.cache.values()))
        else:
            self.logger.info('file does not exists %s', input_filename)
            self.cache = {}
