from sellibrary.text_file_loader import join_feature_matrix

from sellibrary.text_file_loader import load_feature_matrix
from sellibrary.locations import FileLocations
from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset
from sellibrary.filter_only_golden import FilterGolden
from sellibrary.util.const import Const



def join_files(file_A_feature_names, filename_A, file_B_feature_names, filename_B, output_filename):


    # Load File A
    X1, y1, docid_array1, entity_id_array1 = load_feature_matrix(feature_filename=filename_A,
                                                                 feature_names=file_A_feature_names,
                                                                 entity_id_index=1,
                                                                 y_feature_index=2,
                                                                 first_feature_index=4,
                                                                 number_features_per_line=len(file_A_feature_names) + 4,
                                                                 tmp_filename='/tmp/temp_conversion_file_x.txt'
                                                                 )

    print(y1.shape)
    dexter_dataset = DatasetDexter()
    wikipedia_dataset = WikipediaDataset()
    # fg = FilterGolden()
    # X1, y1, docid_array1, entity_id_array1 = fg.get_only_golden_rows(X1, y1, docid_array1, entity_id_array1, dexter_dataset,
    #                                                     wikipedia_dataset)



    print(y1.shape)

    # Load File B
    X2, y2, docid_array2, entity_id_array2 = load_feature_matrix(feature_filename=filename_B,
                                                                 feature_names=file_B_feature_names,
                                                                 entity_id_index=1,
                                                                 y_feature_index=2,
                                                                 first_feature_index=4,
                                                                 number_features_per_line=len(file_B_feature_names) + 4,
                                                                 tmp_filename='/tmp/temp_conversion_file.txt'
                                                                 )

    result_x, result_y, result_docid_list, result_entityid_list = join_feature_matrix(X1, y1, docid_array1, entity_id_array1, X2, y2, docid_array2, entity_id_array2)

    print('Result length:'+str(len(result_y)))


    file = open(output_filename, "w")

    for i in range(len(result_y)):
        docid = result_docid_list[i]
        entity_id = result_entityid_list[i]
        golden = 0
        features = result_x[i,:]
        line = str(docid) + ',' + str(entity_id) + ',' + str(golden) + ',0,' + str(list(features))

        if file is not None:
            file.write(line)
            file.write('\n')

    if file is not None:
        file.close()


    print('results written to : '+output_filename)


if __name__ == "__main__":

    # Command used to grep wp file to make system faster
    # grep -v '\[0,.0' wp.txt > wp_minus_0.txt

    const = Const()
    dropbox_intermediate_path = FileLocations.get_dropbox_intermediate_path()

    file_A_feature_names = const.get_sel_feature_names()
    filename_A = dropbox_intermediate_path + 'wp/wp_minus_0.txt'
    file_B_feature_names = const.sent_feature_names
    filename_B = dropbox_intermediate_path + 'wp_sentiment_simple.txt'

    output_filename = dropbox_intermediate_path + 'wp_joined_sel_sent.txt'
    join_files(file_A_feature_names, filename_A, file_B_feature_names, filename_B, output_filename)

    file_C_feature_names = const.get_joined_feature_names()
    filename_C = output_filename
    file_D_feature_names = const.tf_feature_names
    filename_D = dropbox_intermediate_path + 'wp_base_tf_simple_v2.txt'
    output_filename = dropbox_intermediate_path + 'wp_joined_sel_sent_and_tf.txt'

    join_files(file_C_feature_names, filename_C, file_D_feature_names, filename_D, output_filename)


