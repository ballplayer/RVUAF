import openai
import json
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import numpy as np

# 配置OpenAI API Key（替换为你的有效Key）
openai.api_key = "YOUR_OPENAI_API_KEY"

# 加载语义相似度模型
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')


# 定义历史知识图谱数据结构
class KnowledgeGraph:
    def __init__(self):
        self.entities = []  # 实体列表：[{"id": str, "name": str, "attributes": dict}]
        self.relations = []  # 关系列表：[{"head": str, "relation": str, "tail": str}]
        self.next_entity_id = 1  # 新实体ID递增计数器

    def get_entity_by_id(self, entity_id):
        """根据ID查找历史实体"""
        return next((ent for ent in self.entities if ent["id"] == entity_id), None)


# 工具函数：计算语义相似度
def compute_semantic_sim(text1: str, text2: str) -> float:
    emb1 = semantic_model.encode(text1, convert_to_tensor=True)
    emb2 = semantic_model.encode(text2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()


# 工具函数：计算属性重叠率（考虑键值对匹配）
def compute_attr_overlap(attr1: dict, attr2: dict) -> float:
    keys1, keys2 = set(attr1.keys()), set(attr2.keys())
    intersection = keys1 & keys2
    union = keys1 | keys2

    if len(union) == 0:
        return 0.0

    # 统计键值对完全匹配的数量
    match_count = sum(1 for k in intersection if attr1[k] == attr2[k])
    return match_count / len(union)
