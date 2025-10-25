import torch
import networkx as nx
import openai
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration  # 用于图像描述

# 导入之前模块的核心函数（需确保已实现）
from v_kggen_module import update_kg  # 来自模块一：KG更新函数
from rpri_kgr_module import KGReasoningEnv, QuestionParser, infer_answer  # 来自模块二
from rav_kg_module import FeatureExtractor, align_visual_semantic  # 来自模块三

# 初始化OpenAI API（替换为你的密钥）
openai.api_key = "your_openai_api_key"

# 图像描述模型（BLIP）：生成图像文本描述，用于V-KGGen的LLM输入
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


class ClosedLoopSystem:
    def __init__(self, initial_kg: nx.DiGraph, question: str, align_threshold: float = 0.8, max_iter: int = 5):
        """初始化闭环系统"""
        self.kg = initial_kg  # 初始知识图谱
        self.question = question  # 目标推理问题
        self.align_threshold = align_threshold  # 对齐分数阈值
        self.max_iter = max_iter  # 最大迭代次数

        # 初始化核心模块
        self.feature_extractor = FeatureExtractor()  # 视觉/语义特征提取器
        self.question_parser = QuestionParser()  # 问题解析器
        self.rpri_kgr_model = self._load_rpri_kgr_model()  # 加载RPri-KGR模型

        # 图像描述模型（BLIP）
        self.blip_processor = blip_processor
        self.blip_model = blip_model

    def _load_rpri_kgr_model(self):
        """加载预训练的RPri-KGR模型（PPO）"""
        from stable_baselines3 import PPO
        return PPO.load("rpri_kgr_ppo_model")  # 需提前训练并保存

    def _generate_image_desc(self, image_path: str) -> str:
        """用BLIP生成图像文本描述（替代直接调用多模态LLM）"""
        image = Image.open(image_path).convert("RGB")
        inputs = self.blip_processor(image, return_tensors="pt")
        out = self.blip_model.generate(**inputs)
        return self.blip_processor.decode(out[0], skip_special_tokens=True)

    def _llm_generate_kg_update(self, image_desc: str) -> dict:
        """调用LLM生成KG更新（基于图像描述+历史KG）"""
        prompt = f"""
        基于以下信息生成知识图谱更新（JSON格式）：
        1. 图像描述：{image_desc}
        2. 历史KG：{nx.readwrite.json_graph.node_link_data(self.kg)}

        输出要求：
        - entities: 新增/更新实体列表（id, name, attributes）
        - relations: 新增/更新关系列表（head_id, relation_type, tail_id）
        仅返回JSON，无额外文字！
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return eval(response.choices[0].message.content)  # 需确保LLM输出合法JSON

    def run_v_kggen(self, image_path: str) -> nx.DiGraph:
        """执行V-KGGen模块：生成KG更新并合并"""
        # Step1: 生成图像描述
        image_desc = self._generate_image_desc(image_path)
        # Step2: LLM生成KG更新
        kg_update = self._llm_generate_kg_update(image_desc)
        # Step3: 更新历史KG
        self.kg = update_kg(self.kg, kg_update)
        return self.kg

    def run_rpri_kgr(self) -> str:
        """执行RPri-KGR模块：推理问题答案"""
        env = KGReasoningEnv(self.kg, self.question, self.question_parser)
        _, answer = infer_answer(self.rpri_kgr_model, env)
        return answer

    def run_rav_kg(self, image_path: str, answer: str) -> float:
        """执行RAV-KG模块：优化视觉模型并返回对齐分数"""
        _, align_score = align_visual_semantic(
            feature_extractor=self.feature_extractor,
            kg=self.kg,
            image_path=image_path,
            positive_answer=answer,
            epochs=3  # 每轮迭代优化3次
        )
        return align_score

    def start_loop(self, image_paths: list) -> tuple:
        """启动闭环迭代"""
        align_score = 0.0
        iter_num = 0

        while align_score < self.align_threshold and iter_num < self.max_iter:
            print(f"\n=== Iteration {iter_num + 1}/{self.max_iter} ===")
            current_image = image_paths[iter_num % len(image_paths)]  # 循环使用图像序列

            # Step1: 更新KG（V-KGGen）
            print("Running V-KGGen...")
            self.run_v_kggen(current_image)

            # Step2: 推理答案（RPri-KGR）
            print("Running RPri-KGR...")
            answer = self.run_rpri_kgr()
            print(f"RPri-KGR Answer: {answer}")

            # Step3: 优化视觉模型（RAV-KG）
            print("Running RAV-KG...")
            align_score = self.run_rav_kg(current_image, answer)
            print(f"Current Align Score: {align_score:.4f}")

            iter_num += 1

        print("\n=== Loop Finished ===")
        return self.feature_extractor.vit, self.kg, align_score

if __name__ == "__main__":
    # 初始化参数
    initial_kg = nx.DiGraph()  # 空初始KG
    target_question = "What entity is the red cup on top of?"
    image_paths = ["frame_1.jpg", "frame_2.jpg", "frame_3.jpg"]  # 图像序列
    align_threshold = 0.8
    max_iter = 5

    # 启动闭环
    closed_loop = ClosedLoopSystem(initial_kg, target_question, align_threshold, max_iter)
    optimized_vit, final_kg, final_align_score = closed_loop.start_loop(image_paths)

    # 保存结果
    torch.save(optimized_vit.state_dict(), "optimized_vit_closed_loop.pth")
    nx.write_json(final_kg, "final_kg.json")
    print(f"\nFinal Align Score: {final_align_score:.4f}")
    print("Optimized ViT and Final KG saved!")


