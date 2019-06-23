from logging import getLogger
import pickle

import logging_config
from KB2E import KnowledgeGraph
from KB2E.models import TransE

logger = getLogger(__name__)

if __name__ == "__main__":
    model_path = 'models/trans_e_20190619.pickle'
    #model_path = "models/freebase_mtr100_mte100.pickle"
    with open(model_path, 'rb') as f:
        trans_e = pickle.load(f)
        #subj="/m/027rn"
        #pred="/location/country/form_of_government"
        while True:
            #subj="<http://ja.dbpedia.org/resource/ドミニカ共和国>"
            #pred="<http://rdf.freebase.com/ns/location/country/form_of_government>"
            print("subj: ", end="")
            subj=input().strip()
            print("pred: ", end="")
            pred=input().strip()
            ranking = trans_e.predict_spv(subj=subj, pred=pred, limit=10) # (4)
            print("query: {} {} ?o .".format(subj, pred))
            for (obj, diff) in ranking:
                print("{}: {}".format(diff, obj))