from logging import getLogger
import pickle
from typing import List
import csv

import numpy
from tqdm import tqdm

import logging_config
from KB2E import KnowledgeGraph
from KB2E.models import TransE

logger = getLogger(__name__)

if __name__ == "__main__":
    #model_path = "models/freebase_mtr100_mte100_wo_NT_check.pickle"
    #test_data_path = "data/FB15k/freebase_mtr100_mte100-test.txt"

    model_path = "models/freebase_mtr100_mte100.pickle"
    test_data_path = "data/FB15k/freebase_mtr100_mte100-test.txt"

    Triple = List[int]
    trans_e = pickle.load(open(model_path, 'rb'))
    kb = trans_e._kb

    # TODO: NotImplemented, consider OOV.
    #test:List[Triple] = kb.transform("data/FB15k/freebase_mtr100_mte100-test.txt", format="TSV")
    test = list()
    test_data = open(test_data_path, 'r')
    test_data = csv.reader(test_data, delimiter='\t')
    for triple in test_data:
        [subj, pred, obj] = triple
        sid = kb._entity_dict.inverse[subj]
        oid = kb._entity_dict.inverse[obj]
        pid = kb._relation_dict.inverse[pred]
        test.append([sid, pid, oid])

    entire = 0
    accurate = 0
    for test_triple in tqdm(test):
        subj_vec = trans_e._entities[test_triple[0]]
        pred_vec = trans_e._relations[test_triple[1]]

        results = numpy.apply_along_axis(lambda row:numpy.linalg.norm(row, ord=2), 1, subj_vec+pred_vec-trans_e._entities)
        results = numpy.argsort(results)
        result:int = results[0]

        if result == test_triple[2]:
            accurate += 1
        else:
            pass
        entire += 1
    logger.debug("accuracy: {} = {}/{}".format(accurate/entire, accurate, entire))