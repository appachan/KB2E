from copy import deepcopy
from logging import getLogger
from typing import Any, List
import random

import numpy
from numpy import sqrt
from numpy import random as np_random
from tqdm import tqdm

from KB2E import KnowledgeGraph, Model

logger = getLogger(__name__)


class TransE(Model):
    def __init__(self, ganma:float=2, k:int=50, lmd:float=0.01) -> None:
        """TranE: translation-based model forcuses on a query (h, l, ?t).
        :param ganma: margin
        :param k: embeddings dim
        :param lmd: update parameter"""
        logger.debug("params: ganma={}, k={}".format(ganma, k))
        self._ganma: float = ganma
        self._k: int = k
        self._lambda: float = lmd
        self._kb = None
        super().__init__()

    def fit(self, kb:KnowledgeGraph) -> None:
        self._kb = kb
        ganma =  self._ganma
        k = self._k
        lmd = self._lambda

        def l2_normalize(row):
            l2_norm = numpy.linalg.norm(row, ord=2)
            return row/l2_norm

        # Initialize
        relations = numpy.array([np_random.uniform(low=-6/sqrt(k), high=6/sqrt(k), size=k) for _ in kb._relation_dict]) # U(-6/k^-2, 6/k^-2)
        relations = numpy.apply_along_axis(l2_normalize, axis=1, arr=relations) # L2-normalize
        entities = numpy.array([np_random.uniform(low=-6/sqrt(k), high=6/sqrt(k), size=k) for _ in kb._entity_dict]) # U(-6/k^-2, 6/k^-2)
        kb._id_triples = kb._id_triples[:int(len(kb._id_triples))]
        T = self._make_triple_pairs(kb._id_triples) # 事前にnegative samplesを生成

        # Training
        batch_size = int(len(kb._id_triples)/5) # < |FB-train-data| = 483,142
        #while True:
        updated = 100.0
        for _ in range(1000):
            logger.debug("start mini batch")
            entities = numpy.apply_along_axis(l2_normalize, axis=1, arr=entities) # L2-normalize
            S_batch = [i for i in range(len(kb._id_triples))]
            np_random.shuffle(S_batch)
            S_batch = [S_batch[i] for i in range(batch_size)]
            """
            S_batch = [kb._id_triples[i] for i in S_batch] # sample a minibatch of size batch_size
            # ミニバッチ毎にnegative samplesを生成
            T_batch = set() # initialize the set of pairs of triples
            for triple in S_batch:
                corrupted_triple = self._corrupt_triple(triple) # sample a corrupted triple
                T_batch.add((triple, corrupted_triple))
            T_batch = self._make_triple_pairs(S_batch)
            """
            T_batch = [T[i] for i in S_batch]
            logger.debug("start update")
            updates = list()
            for (triple, corrupted_triple) in tqdm(T_batch, desc="mini batch execution, size: {}".format(batch_size)):
            #for (triple, corrupted_triple) in T_batch:
                [h, ell, t] = triple
                [h_, _, t_] = corrupted_triple
                update = ganma + self._l2_distance(entities[h]+relations[ell], entities[t]) - self._l2_distance(entities[h_]+relations[ell], entities[t_])
                if update > 0:
                    true_triple_update = lmd*(entities[h]+relations[ell]-entities[t])/self._l2_distance(entities[h]+relations[ell], entities[t])
                    corrupted_triple_update = lmd*(entities[h_]+relations[ell]-entities[t_])/self._l2_distance(entities[h_]+relations[ell], entities[t_])

                    entities[h]    = entities[h]    - true_triple_update
                    entities[t]    = entities[t]    + true_triple_update
                    relations[ell] = relations[ell] - true_triple_update

                    entities[h_]   = entities[h_]   + corrupted_triple_update
                    entities[t_]   = entities[t_]   - corrupted_triple_update
                    relations[ell] = relations[ell] + corrupted_triple_update
                #logger.debug("update: {}".format(update))
                updates.append(max([0, update]))
            updates = numpy.array(updates)
            updates = numpy.sum(updates) / len(updates)
            logger.info(updates)
            if updated < updates:
                break
            updated = updates
        self._entities = entities
        self._relations = relations
    
    def predict_spv(self, subj:str, pred:str, limit:int =5) -> List:
        subj_id = self._kb._entity_dict.inverse[subj]
        pred_id = self._kb._relation_dict.inverse[pred]
        subj_vec = self._entities[subj_id]
        pred_vec = self._relations[pred_id]
        ranking = list()
        for id, e in enumerate(self._entities):
            diff = numpy.linalg.norm(subj_vec+pred_vec-e, ord=2)
            ranking.append((id, diff))
        ranking = sorted(ranking, key=lambda obj:obj[1])
        ranking = ranking[:limit]
        ranking = list(map(lambda obj:(self._kb._entity_dict[obj[0]], obj[1]), ranking))
        return ranking
    
    def _make_triple_pairs(self, true_triples):
        triple_pairs = list()
        all_triples = [i for i in range(len(true_triples))]
        np_random.shuffle(all_triples)
        pivot = int(len(true_triples)/2)
        subj_replaces = true_triples[0:pivot]
        obj_replaces = true_triples[pivot:len(true_triples)]
        entities = [i for i in range(len(self._kb._entity_dict))]
        logger.debug("negative sampling..., entity dict size: {}".format(len(entities)))
        for triple in tqdm(subj_replaces, desc="subject replacing"):
        #for triple in subj_replaces:
            entities.remove(triple[0])
            """
            candidate = random.sample(range(len(entities)), 1)[0]
            candidate = entities[candidate]
            """
            np_random.shuffle(entities)
            for i in entities:
                if i != triple[0] and [i, triple[1], triple[2]] not in self._kb._id_triples:
                    candidate = i
                    break
            triple_pairs.append((triple, [candidate, triple[1], triple[2]]))
            entities.append(triple[0])
        for triple in tqdm(obj_replaces, desc="object replacing"):
        #for triple in obj_replaces:
            entities.remove(triple[2])
            np_random.shuffle(entities)
            for i in entities:
                if i != triple[2] and [triple[0], triple[1], i] not in self._kb._id_triples:
                    candidate = i
                    break
            triple_pairs.append((triple, [triple[0], triple[1], candidate]))
            entities.append(triple[2])
        #logger.debug(self._kb.decode_triple(triple_pairs[0][0]), " -> ", self._kb.decode_triple(triple_pairs[0][1]))
        return triple_pairs
    
    def _l2_distance(self, target, source):
        return numpy.linalg.norm(target-source, ord=2)
