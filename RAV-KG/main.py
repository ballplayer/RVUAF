### pip install torch torchvision transformers pillow networkx numpy
import torch
# import torch.nn.functional as F
# from torchvision import models, transforms
# from transformers import BertTokenizer, BertModel
# from PIL import Image
import networkx as nx
# import numpy as np
# from typing import List, Tuple
from InitializationandFeatureExtraction import FeatureExtractor
from VisualSemanticAlignmentTraining import align_visual_semantic

if __name__ == "__main__":
    # 1. 初始化工具
    feature_extractor = FeatureExtractor()

    # 2. 构建示例KG
    kg = nx.DiGraph()
    kg.add_nodes_from(["cup", "book", "table", "chair", "computer"])
    kg.add_edges_from([
        ("cup", "book", {"relation": "next_to"}),
        ("book", "table", {"relation": "on_top_of"}),
        ("chair", "table", {"relation": "next_to"})
    ])

    # 3. 输入数据
    image_path = "test_image.jpg"  # 替换为你的图像路径
    positive_answer = "blue"  # 问题答案（语义正样本）

    # 4. 执行对齐训练
    optimized_vit, final_align_score = align_visual_semantic(
        feature_extractor=feature_extractor,
        kg=kg,
        image_path=image_path,
        positive_answer=positive_answer,
        epochs=5
    )

    # 5. 保存优化后的视觉模型
    torch.save(optimized_vit.state_dict(), "optimized_vit.pth")
    print(f"Final Align Score: {final_align_score:.4f}")
