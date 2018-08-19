import logging

from sel.file_locations import FileLocations

#
#  _________________
#
# Use sellibrary.sel DexterDataset rather than this one.
# ________________

# class DatasetDexter:
#     # Set up logging
#     handler = logging.StreamHandler()
#     handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
#     logger = logging.getLogger(__name__)
#     logger.addHandler(handler)
#     logger.propagate = False
#     logger.setLevel(logging.INFO)
#
#     def __init__(self):
#         pass
#
#     @staticmethod
#     def get_dexter_dataset(path=None, filename='saliency-dataset.json'):
#         if path is None:
#             path = FileLocations.get_dropbox_dexter_path()
#         with open(path + filename) as f:
#             content = f.readlines()
#         return content
