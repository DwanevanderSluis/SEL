from sellibrary.trec.trec_util import TrecReferenceCreator

if __name__ == "__main__":
    df = TrecReferenceCreator()
    df.create_reference_file(True)
