from logging import getLogger
import pickle

import logging_config
from KB2E import KnowledgeGraph
from KB2E.models import TransE

logger = getLogger(__name__)

if __name__ == "__main__":
    #kb_file = open('data/test.tsv')
    #kb_file = open('data/FB15k/freebase_mtr100_mte100-train.txt')
    kb_file = open('data/freebase_mtr100_mte100-train.nt')
    kb = KnowledgeGraph()
    #kb.load(kb_file, format='TSV')
    kb.load(kb_file, format='N-TRIPLE')
    trans_e = TransE(ganma=2.0, k=50, lmd=0.1)
    trans_e.fit(kb) # (2-3)

    with open('models/trans_e.pickle', 'wb') as f:
        pickle.dump(trans_e, f)
    """
    with open('models/trans_e.pickle', 'rb') as f:
        trans_e = pickle.load(f)
    """