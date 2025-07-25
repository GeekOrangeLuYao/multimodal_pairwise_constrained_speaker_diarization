from typing import Callable
import argparse
import tqdm

import numpy as np

from utils.config import YamlConfigLoader
from utils.builder import deep_build
from utils.logger import get_logger
from utils.file import read_wav_scp

from utils.file import write_embedding2cluster_file_in_list, read_constraint_propagation_json

logger = get_logger()


def get_args():
    parser = argparse.ArgumentParser(
        description="Run pairwise constraints cluster algorithm"
    )
    parser.add_argument(
        "--embedding_scp_file", required=True, help="The input embeddings.scp file"
    )
    parser.add_argument(
        "--constraint_scp_file", required=True, help="The constraints.scp file"
    )
    parser.add_argument(
        "--cluster_label_file", required=True, help="The output cluster label file"
    )
    parser.add_argument(
        "--cluster_config_file", required=True, help="The cluster config file"
    )
    return parser.parse_args()


def build_cluster_algorithm(cluster_config_file) -> Callable:
    config_loader = YamlConfigLoader(cluster_config_file)
    config = config_loader.instance()

    logger.info(f"{config}")
    pcc_function = deep_build(config.cluster_function)
    return pcc_function


def main():
    args = get_args()
    logger.info(f"{args}")

    embedding_scp_file = args.embedding_scp_file
    constraint_scp_file = args.constraint_scp_file
    cluster_label_file = args.cluster_label_file
    cluster_config_file = args.cluster_config_file

    cluster_function = build_cluster_algorithm(cluster_config_file)

    embedding_scp = read_wav_scp(embedding_scp_file)
    constraint_scp = read_wav_scp(constraint_scp_file)

    utterances_num = len(embedding_scp)
    utterance_list = list(embedding_scp.keys())

    embedding2cluster_list = []
    for i in tqdm.tqdm(range(utterances_num)):
        utt_id = utterance_list[i]
        logger.info(f"utt_id = {utt_id}")
        embedding_file = embedding_scp[utt_id]
        constraint_file = constraint_scp[utt_id]
        logger.info(f"embedding_file = {embedding_file}, constraint_file = {constraint_file}")
        embedding_dict = np.load(embedding_file, allow_pickle=True).item()
        constraint_dict = read_constraint_propagation_json(constraint_file)

        cluster_labels = cluster_function(embedding_dict, constraint_dict)

        for emb_id, emb_cluster_label in cluster_labels.items():
            embedding2cluster_list.append((
                emb_id, utt_id, emb_cluster_label
            ))
        logger.info(f"Cluster and label {len(cluster_labels)} cluster labels")

    logger.info(f"Collect results and write to {cluster_label_file}")
    write_embedding2cluster_file_in_list(cluster_label_file, embedding2cluster_list)


if __name__ == '__main__':
    main()
