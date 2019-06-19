import csv
from logging import getLogger
from typing import Dict, List

from bidict import bidict

logger = getLogger(__name__)
Triple = List[int]


class KnowledgeGraph():
    def __init__(self) -> None:
        self._subject_dict = bidict()
        self._predicate_dict = bidict()
        self._object_dict = bidict()
        self._entity_dict = bidict()
        self._relation_dict = self._predicate_dict
        self._id_triples: List[Triple] = list()

    def load(self, kb_file, *, format="N-TRIPLE", encode="utf-8") -> None:
        """
        :param kb_file: file-like object of knowledge base
        :param format: serialize format
        :param encode: ..."""
        if format == "TSV":
            kb = csv.reader(kb_file, delimiter='\t')
            for triple in kb:
                [subj, pred, obj] = triple

                def get_resource_id(resource: str, bidict) -> int:
                    if resource not in bidict.inverse:
                        id = len(bidict)
                        bidict[id] = resource
                        return id
                    else:
                        return bidict.inverse[resource]
                #subj_id: int = get_resource_id(subj, self._subject_dict)
                #obj_id: int = get_resource_id(obj, self._object_dict)
                subj_id: int = get_resource_id(subj, self._entity_dict)
                obj_id: int = get_resource_id(obj, self._entity_dict)
                pred_id: int = get_resource_id(pred, self._predicate_dict)
                self._id_triples.append([
                    subj_id,
                    pred_id,
                    obj_id
                ])
        elif format == "N-TRIPLE":
            # TODO: rdflib
            kb = csv.reader(kb_file, delimiter=' ')
            for triple in kb:
                [subj, pred, obj, _] = triple

                def get_resource_id(resource: str, bidict) -> int:
                    if resource not in bidict.inverse:
                        id = len(bidict)
                        bidict[id] = resource
                        return id
                    else:
                        return bidict.inverse[resource]
                #subj_id: int = get_resource_id(subj, self._subject_dict)
                #obj_id: int = get_resource_id(obj, self._object_dict)
                subj_id: int = get_resource_id(subj, self._entity_dict)
                obj_id: int = get_resource_id(obj, self._entity_dict)
                pred_id: int = get_resource_id(pred, self._predicate_dict)
                self._id_triples.append([
                    subj_id,
                    pred_id,
                    obj_id
                ])
        else:
            raise NotImplementedError("only *.tsv can be accepted.")
        logger.debug("size of entity_dict: {}, s_dict: {}, p_dict: {}, o_dict: {}".format(
            len(self._entity_dict),
            len(self._subject_dict),
            len(self._predicate_dict),
            len(self._object_dict)
        ))
        logger.debug("size of triples: {}".format(len(self._id_triples)))

    def decode_triple(self, id_triple):
        return (
            self._entity_dict[id_triple[0]],
            self._relation_dict[id_triple[1]],
            self._entity_dict[id_triple[2]],
            )
