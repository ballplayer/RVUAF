import networkx as nx
from typing import List, Tuple

import numpy as np


def generate_negative_samples(kg: nx.DiGraph, positive_answer: str, num_neg: int = 3) -> List[str]:
    """从知识图谱中生成无关负样本"""
    # 获取KG中所有实体名称
    all_entities = [node for node in kg.nodes()]

    # 排除正样本及其相关实体（简单过滤）
    related_entities = set()
    related_entities.add(positive_answer)
    # 添加正样本的邻接实体
    for u, v in kg.edges(positive_answer):
        related_entities.add(v)
    for u, v in kg.in_edges(positive_answer):
        related_entities.add(u)

    # 筛选负样本
    negative_candidates = [ent for ent in all_entities if ent not in related_entities]
    # 随机选择num_neg个负样本
    np.random.seed(42)
    negative_samples = np.random.choice(negative_candidates, min(num_neg, len(negative_candidates)), replace=False)

    return list(negative_samples)
