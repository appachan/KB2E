from logging import getLogger
import pickle

import logging_config
from KB2E import KnowledgeGraph
from KB2E.models import TransE

logger = getLogger(__name__)

if __name__ == "__main__":
    kb = KnowledgeGraph()
    #model_path = 'models/freebase_mtr100_mte100.pickle'
    model_path = 'trans_e.pickle'
    #kb_file = open('data/test.tsv')
    kb_file = open('data/FB15k/freebase_mtr100_mte100-train.txt')
    logger.info("train data: {}".format(kb_file.name))
    kb.load(kb_file, format='TSV')
    #kb_file = open('data/freebase_mtr100_mte100-train.nt')
    #kb.load(kb_file, format='N-TRIPLE')
    trans_e = TransE(ganma=2.0, k=50, lmd=0.1)
    trans_e.fit(kb) # (2-3)

    with open(model_path, 'wb') as f:
        pickle.dump(trans_e, f)