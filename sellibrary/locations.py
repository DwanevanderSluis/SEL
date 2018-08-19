import sys
import os

class FileLocations:

    @staticmethod
    def get_dropbox_wikipedia_path():
        if 'PWD' in os.environ and os.environ['PWD'] == '/root/experiments':
            path = '/root/experiments/Dropbox/Datasets/wikipedia/' # docker image
        else:
            if sys.platform == 'win32':
                path = 'C:\\Users\\dwane\\Dropbox\\Datasets\\wikipedia\\'
            else:
                if sys.platform == 'linux':  # Sherlock ML
                    path = '/project/Dropbox/Datasets/wikipedia/'
                else:
                    path = '/Users/dsluis/Dropbox/Datasets/wikipedia/'
        return path


    @staticmethod
    def get_dropbox_dexter_path():
        # Data down loaded from
        # https://github.com/dexter/dexter-datasets/tree/master/entity-saliency
        if 'PWD' in os.environ and os.environ['PWD'] == '/root/experiments':
            path = '/root/experiments/Dropbox/Datasets/dexter/' # docker image
        else:
            if sys.platform == 'win32':
                path = 'C:\\Users\\dwane\\Dropbox\\Datasets\\dexter\\'
            else:
                if sys.platform == 'linux': # Sherlock ML
                    path = '/project/Dropbox/Datasets/dexter/'
                else:
                    path = '/Users/dsluis/Dropbox/Datasets/dexter/'
        return path


    @staticmethod
    def get_temp_path():
        if 'PWD' in os.environ and os.environ['PWD'] == '/root/experiments':
            path = '/root/experiments/tmp/' # docker image
        else:
            if sys.platform == 'win32':
                path = 'C:\\temp\\'
            else:
                if sys.platform == 'linux':  # Sherlock ML
                    path = '/project/tmp/'
                else:
                    path = '/Users/dsluis/Data/tmp/' # mac
        return path


    @staticmethod
    def get_dropbox_intermediate_path():
        if 'PWD' in os.environ and os.environ['PWD'] == '/root/experiments':
            dropbox_intermediate_path = '/root/experiments/Dropbox/Datasets/intermediate/' # docker image
        else:
            if sys.platform == 'win32':
                dropbox_intermediate_path = 'C:\\Users\\dwane\\Dropbox\\Datasets\\intermediate\\'
            else:
                if sys.platform == 'linux':  # Sherlock ML
                    dropbox_intermediate_path = '/project/Dropbox/Datasets/intermediate/'
                else:
                    dropbox_intermediate_path = '/Users/dsluis/Dropbox/Datasets/intermediate/'  # mac
        return dropbox_intermediate_path


    @staticmethod
    def get_trec_eval_executable_location():
        if sys.platform == 'win32':
            path = 'C:\\programs\\trec_eval-master\\trec_eval.exe'
        else:
            if sys.platform == 'linux':  # Sherlock ML
                path = '/project/trec_eval-master/trec_eval'
            else:
                path = '/Users/dsluis/code/trec_eval.9.0.4/trec_eval'
        return path


    @staticmethod
    def get_dropbox_datasets_path():
        if 'PWD' in os.environ and os.environ['PWD'] == '/root/experiments':
            dropbox_intermediate_path = '/root/experiments/Dropbox/Datasets/' # docker image
        else:
            if sys.platform == 'win32':
                dropbox_intermediate_path = 'C:\\Users\\dwane\\Dropbox\\Datasets\\'
            else:
                if sys.platform == 'linux':  # Sherlock ML
                    dropbox_intermediate_path = '/project/Dropbox/Datasets/'
                else:
                    dropbox_intermediate_path = '/Users/dsluis/Dropbox/Datasets/'  #mac
        return dropbox_intermediate_path



