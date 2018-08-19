from sellibrary.text_file_loader import join_feature_matrix

from sellibrary.text_file_loader import load_feature_matrix
from sellibrary.locations import FileLocations
from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset
from sellibrary.filter_only_golden import FilterGolden
from sellibrary.util.const import Const

if __name__ == "__main__":

    const = Const()
    dropbox_intermediate_path = FileLocations.get_dropbox_intermediate_path()


    input_filename = dropbox_intermediate_path + 'joined_sel_sent_and_tf.txt'
    input_feature_names = const.get_joined_sel_sent_and_tf_feature_names()

    output_filename = dropbox_intermediate_path + 'efficient_2_features.txt'
    output_features_names = const.efficient_2_feature_names




    X1, y1, docid_array1, entity_id_array1 = load_feature_matrix(feature_filename=input_filename,
                                                                 feature_names=input_feature_names,
                                                                 entity_id_index=1,
                                                                 y_feature_index=2,
                                                                 first_feature_index=4,
                                                                 number_features_per_line=len(input_feature_names) + 4,
                                                                 tmp_filename='/tmp/temp_conversion_file.txt'
                                                                 )

    fg = FilterGolden()
    dexter_dataset = DatasetDexter()
    wikipedia_dataset = WikipediaDataset()
    X1, y1, docid_array1, entity_id_array1 = fg.get_only_golden_rows(X1, y1, docid_array1, entity_id_array1, dexter_dataset,
                                                        wikipedia_dataset)
    document_list = dexter_dataset.get_dexter_dataset(path=FileLocations.get_dropbox_dexter_path())
    golden_saliency_by_entid_by_docid = dexter_dataset.get_golden_saliency_by_entid_by_docid(document_list, wikipedia_dataset)



    print(y1.shape)





    indexes_to_keep = []
    for f in output_features_names:
        i = input_feature_names.index(f)
        if i == -1:
            print('Can not find feature',f,'in loaded file')
            raise Exception('can not find feature in loaded file')
        print('added',f)
        indexes_to_keep.append(i)



    file = open(output_filename, "w")

    for i in range(len(y1)):
        docid = docid_array1[i]
        entity_id = entity_id_array1[i]

        golden = 0
        if docid in golden_saliency_by_entid_by_docid:
            if entity_id in golden_saliency_by_entid_by_docid[docid]:
                golden = golden_saliency_by_entid_by_docid[docid][entity_id]
        features = X1[i, indexes_to_keep]
        line = str(docid) + ',' + str(entity_id) + ',' + str(golden) + ',0,' + str(list(features))

        if file is not None:
            file.write(line)
            file.write('\n')

    if file is not None:
        file.close()


    print('results written to : '+output_filename)