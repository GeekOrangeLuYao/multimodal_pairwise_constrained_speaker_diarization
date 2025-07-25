from typing import List

import codecs
import json


def write_embedding2cluster_file_in_list(filename, embedding2cluster: List):
    """
        embedding2cluster: list(
            embedding_id: str
            utt_id: str
            label: int / str
        )
    """
    with open(filename, "w") as fw:
        for emb_id, utt_id, label in embedding2cluster:
            fw.write(f"{emb_id} {utt_id} {label}\n")
        fw.flush()


def read_constraint_propagation_json(constraint_json_file):
    """
    {
      "content": {
        "must_link": [
          // must link list
          {
            "pre_emb_id": "embedding_101",
            "nxt_emb_id": "embedding_102",
            "constraint": 1
          }
        ],
        "cannot_link": [
          // cannot link list
          {
            "pre_emb_id": "embedding_001",
            "nxt_emb_id": "embedding_002",
            "constraint": -1
          }
        ]
      }
    }
    """
    with codecs.open(constraint_json_file, "r", encoding="utf-8") as fr:
        data_dict = json.load(fr)
        return data_dict


def write_constraint_propagation_json(constraint_json_file, cp_dict):
    """
    {
      "content": {
        "must_link": [
          // must link list
          {
            "pre_emb_id": "embedding_101",
            "nxt_emb_id": "embedding_102",
            "constraint": 1
          }
        ],
        "cannot_link": [
          // cannot link list
          {
            "pre_emb_id": "embedding_001",
            "nxt_emb_id": "embedding_002",
            "constraint": -1
          }
        ]
      }
    }
    """
    with codecs.open(constraint_json_file, "w", encoding="utf-8") as fw:
        json.dump(cp_dict, fw, indent=2, ensure_ascii=False)


def read_pairwise_constraints_embedding2cluster_file(embedding2cluster_file):
    """
        Format:
            utt2embedding2cluster: dict(
                utt_id: str -> dict(
                    embedding_id: str -> Tuple(
                        cluster_label: int,
                        embedding_start_time: float(int seconds),
                        embedding_end_time: float(in seconds),
                    )
                )
            )
    """
    utt2embedding2cluster = dict()
    with open(embedding2cluster_file, "r") as fr:
        lines = fr.readlines()
        for line in lines:
            ps = line.strip().split()
            assert len(ps) == 3, f"line: {line}"
            embedding_id, utt_id, cluster_label = ps[0], ps[1], ps[2]
            if utt_id not in utt2embedding2cluster:
                utt2embedding2cluster[utt_id] = dict()
            st_ed = embedding_id.strip().split("_")
            st, ed = float(st_ed[-2]), float(st_ed[-1])
            utt2embedding2cluster[utt_id][embedding_id] = (cluster_label, st, ed)
    return utt2embedding2cluster
