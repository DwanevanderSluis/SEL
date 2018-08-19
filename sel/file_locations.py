# import sys
#
#
# class FileLocations:
#
#     @staticmethod
#     def get_dropbox_wikipedia_path():
#         if sys.platform == 'win32':
#             path = 'C:\\Users\\dwane\\Dropbox\\Datasets\\wikipedia\\'
#         else:
#             if sys.platform == 'linux':  # Sherlock ML
#                 path = '/project/Dropbox/Datasets/wikipedia/'
#             else:
#                 path = '/Users/dsluis/Dropbox/Datasets/wikipedia/'
#         return path
#
#
#     @staticmethod
#     def get_dropbox_dexter_path():
#         # Data down loaded from
#         # https://github.com/dexter/dexter-datasets/tree/master/entity-saliency
#         if sys.platform == 'win32':
#             path = 'C:\\Users\\dwane\\Dropbox\\Datasets\\dexter\\'
#         else:
#             if sys.platform == 'linux': # Sherlock ML
#                 path = '/project/Dropbox/Datasets/dexter/'
#             else:
#                 path = '/Users/dsluis/Dropbox/Datasets/dexter/'
#         return path
#
#
#     @staticmethod
#     def get_temp_path():
#         if sys.platform == 'win32':
#             path = 'C:\\temp\\'
#         else:
#             if sys.platform == 'linux':  # Sherlock ML
#                 path = '/project/tmp/'
#             else:
#                 path = '/Users/dsluis/Data/tmp/'
#         return path
#
#
#     @staticmethod
#     def get_dropbox_intermediate_path():
#         if sys.platform == 'win32':
#             dropbox_intermediate_path = 'C:\\Users\\dwane\\Dropbox\\Datasets\\intermediate\\'
#         else:
#             if sys.platform == 'linux':  # Sherlock ML
#                 dropbox_intermediate_path = '/project/Dropbox/Datasets/intermediate/'
#             else:
#                 dropbox_intermediate_path = '/Users/dsluis/Dropbox/Datasets/intermediate/'
#         return dropbox_intermediate_path
#
#
#     @staticmethod
#     def get_trec_eval_executable_location():
#         if sys.platform == 'win32':
#             path = 'C:\\programs\\trec_eval-master\\trec_eval.exe'
#         else:
#             if sys.platform == 'linux':  # Sherlock ML
#                 path = '/project/trec_eval-master/trec_eval'
#             else:
#                 path = '/Users/dsluis/code/trec_eval.8.1/trec_eval'
#         return path
