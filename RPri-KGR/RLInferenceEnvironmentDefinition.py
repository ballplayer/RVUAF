### pip install networkx torch transformers numpy
import networkx as nx
import numpy as np
import torch
from gym import Env, spaces
from transformers import AutoTokenizer, AutoModel, pipeline


class KGReasoningEnv(Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, kg: nx.DiGraph, question: str, parser):
        super().__init__()
        self.kg = kg
        self.question = question
        self.parser = parser

        # 解析问题
        self.source_entity = self.parser.extract_source_entity(question)
        self.target_relation = self.parser.extract_target_relation(question)
        assert self.source_entity and self.target_relation, "问题解析失败"

        # 状态初始化
        self.current_entity = self.source_entity
        self.step_count = 0
        self.max_steps = 10

        # 动作空间：当前实体的所有出边索引
        self.action_space = spaces.Discrete(len(list(self.kg.out_edges(self.current_entity))))

        # 观测空间：实体嵌入+路径特征（用BERT生成768维嵌入）
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = AutoModel.from_pretrained("bert-base-uncased")
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(768,), dtype=np.float32)

        # 预计算正确答案（目标实体）
        self.answer_entity = self._get_correct_answer()

    def _get_correct_answer(self) -> str:
        """从KG中获取问题的正确答案实体"""
        for u, v, data in self.kg.out_edges(self.source_entity, data=True):
            if data["relation"] == self.target_relation:
                return v
        return None

    def _get_observation(self) -> np.ndarray:
        """生成观测：当前实体的BERT嵌入"""
        inputs = self.tokenizer(self.current_entity, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding

    def step(self, action: int) -> tuple:
        """执行动作并返回状态、奖励、终止标志"""
        self.step_count += 1

        # 获取当前实体的出边
        out_edges = list(self.kg.out_edges(self.current_entity, data=True))
        if action >= len(out_edges):
            return self._get_observation(), -1.0, True, {}  # 无效动作惩罚

        # 执行动作：选择边
        u, v, edge_data = out_edges[action]
        next_entity = v
        edge_weight = edge_data["weight"]

        # 计算奖励
        terminal_reward = 10.0 if next_entity == self.answer_entity else 0.0  # 终端奖励
        path_reward = self.kg.nodes[next_entity]["weight"] + edge_weight  # 路径优先级奖励
        efficiency_penalty = -0.1 * self.step_count  # 效率惩罚
        total_reward = terminal_reward + path_reward + efficiency_penalty

        # 更新状态
        self.current_entity = next_entity
        done = (terminal_reward > 0) or (self.step_count >= self.max_steps)

        return self._get_observation(), total_reward, done, {}

    def reset(self) -> np.ndarray:
        """重置环境状态"""
        self.current_entity = self.source_entity
        self.step_count = 0
        return self._get_observation()

    def render(self, mode="human") -> None:
        """打印当前状态"""
        print(f"当前实体：{self.current_entity}, 步骤：{self.step_count}")

    def close(self) -> None:
        """关闭环境"""
        pass


# 测试环境初始化
env = KGReasoningEnv(kg, question, parser) # 可注释掉
print("环境观测空间：", env.observation_space.shape)
print("环境动作空间：", env.action_space.n)
