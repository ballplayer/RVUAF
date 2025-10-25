import torch
import torch.nn.functional as F
from torchvision import models, transforms
import networkx as nx
from typing import List, Tuple
from NegativeSampleGeneration import generate_negative_samples
from KACComparativeLossCalculation import kac_loss

def align_visual_semantic(
        feature_extractor,
        kg: nx.DiGraph,
        image_path: str,
        positive_answer: str,
        num_neg: int = 3,
        learning_rate: float = 1e-5,
        epochs: int = 5
) -> Tuple[models.ViT_B_16_Weights, float]:
    """执行视觉-语义反馈对齐训练"""
    # 1. 提取特征
    visual_feat = feature_extractor.extract_visual_feature(image_path)
    pos_semantic_feat = feature_extractor.extract_semantic_feature(positive_answer)

    # 2. 生成负样本
    negative_samples = generate_negative_samples(kg, positive_answer, num_neg)
    neg_semantic_feats = [feature_extractor.extract_semantic_feature(neg) for neg in negative_samples]

    # 3. 初始化优化器（仅优化ViT）
    optimizer = torch.optim.Adam(feature_extractor.vit.parameters(), lr=learning_rate)

    # 4. 训练对齐
    feature_extractor.vit.train()  # 切换到训练模式
    for epoch in range(epochs):
        optimizer.zero_grad()

        # 计算损失
        loss = kac_loss(visual_feat, pos_semantic_feat, neg_semantic_feats)

        # 反向传播与优化
        loss.backward()
        optimizer.step()

        # 计算对齐分数（余弦相似度）
        with torch.no_grad():
            align_score = F.cosine_similarity(visual_feat, pos_semantic_feat, dim=0).item()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f} | Align Score: {align_score:.4f}")

    # 恢复ViT为评估模式
    feature_extractor.vit.eval()

    return feature_extractor.vit, align_score
