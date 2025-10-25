import torch
from typing import List, Tuple
import torch.nn.functional as F

def kac_loss(
        visual_feat: torch.Tensor,
        pos_semantic_feat: torch.Tensor,
        neg_semantic_feats: List[torch.Tensor],
        temperature: float = 0.07
) -> torch.Tensor:
    """计算视觉-语义对比损失（InfoNCE变种）"""
    # 计算正样本相似度
    pos_sim = F.cosine_similarity(visual_feat, pos_semantic_feat, dim=0) / temperature

    # 计算负样本相似度
    neg_sims = []
    for neg_feat in neg_semantic_feats:
        neg_sim = F.cosine_similarity(visual_feat, neg_feat, dim=0) / temperature
        neg_sims.append(neg_sim)

    # 合并正/负样本相似度
    logits = torch.cat([pos_sim.unsqueeze(0), torch.stack(neg_sims)])  # shape: (1+num_neg,)
    labels = torch.tensor([0], dtype=torch.long).to(visual_feat.device)  # 正样本索引为0

    # 交叉熵损失
    loss = F.cross_entropy(logits.unsqueeze(0), labels)
    return loss
