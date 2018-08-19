import json
import logging
import operator
from subprocess import check_output
import numpy as np

from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.locations import FileLocations

class TrecReferenceCreator:
    # set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def __init__(self):
        pass

    # prints key json fields
    def process_line(self, text):
        self.logger.info('____________________')
        data = json.loads(text)
        self.logger.info(data)
        print('____________________')
        for e in data['saliency']:
            entityid = e['entityid']
            score = e['score']
            self.logger.info('%s %d', entityid, score)
        print('____________________')

    @staticmethod
    def extract_saliency_by_ent_id_golden(data):
        saliency_by_ent_id_golden = {}
        for e in data['saliency']:
            entityid = e['entityid']
            score = e['score']
            saliency_by_ent_id_golden[entityid] = score
        return saliency_by_ent_id_golden

    @staticmethod
    def copy_dic_replacing_nones(x_by_y):
        dict2 = {}
        for k in x_by_y.keys():
            if x_by_y[k] is None:
                dict2[k] = 0.0
            else:
                dict2[k] = x_by_y[k]
        return dict2

    def get_ordered_list_from_dictionary(self, salience_by_entity_id):
        sorted_x = sorted(self.copy_dic_replacing_nones(salience_by_entity_id).items(), key=operator.itemgetter(1),
                          reverse=True)
        return sorted_x

    def create_reference_file(self, zero_less_than_2 ):

        # load the data
        dd = DatasetDexter()
        document_list = dd.get_dexter_dataset(path=FileLocations.get_dropbox_dexter_path())

        results = ''
        # process the data
        result_count = 0
        doc_count = 0

        for document in document_list:
            data = json.loads(document)
            saliency_by_ent_id_golden = self.extract_saliency_by_ent_id_golden(data)
            docid = data['docId']

            sorted_list = self.get_ordered_list_from_dictionary(saliency_by_ent_id_golden)

            for item in sorted_list:
                entity_id = item[0]
                salience = item[1]
                if zero_less_than_2:
                    if salience < 2.0:
                        salience = 0.0
                results = results + str(docid) + ' 0 ' + str(entity_id) + ' ' + str(salience) + '\n'
                result_count += 1

            self.logger.info('Documents Processed %d Entities Processed %d ', doc_count, result_count)
            doc_count += 1

        fn = FileLocations.get_dropbox_intermediate_path() + "trec_ground_truth.txt"
        self.logger.info('writing to %s ', fn)
        file = open(fn, "w")
        file.write(results)
        file.close()

    def create_results_file(self, salience_by_entity_by_doc_id, prefix):
        results = ''
        lines_written = 0
        for docid in salience_by_entity_by_doc_id.keys():
            if docid in salience_by_entity_by_doc_id:
                salience_by_entity_id = salience_by_entity_by_doc_id[docid]
                ordered_list = self.get_ordered_list_from_dictionary(salience_by_entity_id)
                for item in ordered_list:
                    entity_id = item[0]
                    salience = item[1]
                    results = results + str(docid) + ' 0 ' + str(entity_id) + ' 0 ' + str(salience) + ' STANDARD\n'
                    lines_written += 1

        fn = FileLocations.get_temp_path() + prefix + ".trec_results.txt"
        self.logger.info('writing to %s ', fn)
        file = open(fn, "w")
        file.write(results)
        file.close()
        return lines_written

    def isfloat(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False


    def extract_single_measure(self, cmd, param1, param2, trec_val_by_name, measure_name):
        options1 = "-m"
        options2 = measure_name
        self.logger.info('%s %s %s %s %s  ', cmd, options1, options2, param1, param2)
        output2 = check_output([cmd, options1, options2, param1, param2])
        s = output2.decode("utf-8").split('\t')
        v = float(s[2])
        trec_val_by_name[measure_name] = v

        return v, output2


    def get_report(self, golden_source_filename, prefix):
        cmd = FileLocations.get_trec_eval_executable_location()
        param1 = golden_source_filename
        param2 = FileLocations.get_temp_path() + prefix + ".trec_results.txt"
        self.logger.info('%s %s %s  ', cmd, param1, param2)
        output = check_output([cmd, param1, param2])
        s = output.decode("utf-8").split('\n')

        trec_val_by_name = {}
        for i in range(len(s)):
            s_list = s[i].split('\t')
            if len(s_list) >= 2:
                n = s_list[0].strip()
                v = s_list[2]
                if self.isfloat(v):
                    trec_val_by_name[n] = float(v)

        overall_ndcg, result1 = self.extract_single_measure(cmd, param1, param2, trec_val_by_name, "ndcg")
        p_1, result2 = self.extract_single_measure(cmd, param1, param2, trec_val_by_name, "P.1")
        p_2, result3 = self.extract_single_measure(cmd, param1, param2, trec_val_by_name, "P.2")
        p_3, result4 = self.extract_single_measure(cmd, param1, param2, trec_val_by_name, "P.3")
        p_4, result5 = self.extract_single_measure(cmd, param1, param2, trec_val_by_name, "P.4")

        result = output.decode("utf-8") + result1.decode("utf-8")+ result2.decode("utf-8")+ result3.decode("utf-8")+ result4.decode("utf-8")+ result5.decode("utf-8")
        self.logger.debug('%s', result)

        return result, overall_ndcg, trec_val_by_name







