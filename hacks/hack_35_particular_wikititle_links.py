

from sellibrary.wiki.wikipedia_datasets import WikipediaDataset



def do_stuff(word, wds):
    id = wds.get_id_from_wiki_title(word)
    print(id)
    in_degree = wds.get_entity_in_degree(id)
    print('in_degree',in_degree)
    out_degree = wds.get_entity_out_degree(id)
    print('out_degree',out_degree)



if __name__ == "__main__":

    wds = WikipediaDataset()

    madrid = 41188263
    barcelona = 4443
    apple_inc = 8841385
    steve_jobs = 1563047
    steve_jobs = 7412236

    word = 'zorb'
    do_stuff(word,wds)

    word = 'united_states'
    do_stuff(word,wds)


