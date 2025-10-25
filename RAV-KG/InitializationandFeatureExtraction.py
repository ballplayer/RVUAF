import torch
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from PIL import Image

class FeatureExtractor:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

        # 视觉模型（ViT-b-16）
        self.vit = models.vit_b_16(pretrained=True).to(device)
        self.vit.eval()  # 初始为评估模式，训练时切换为train()

        # 语义模型（BERT-base）
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
        self.bert_model.eval()  # 冻结BERT，仅优化ViT

        # 图像预处理（符合ViT输入要求）
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_visual_feature(self, image_path: str) -> torch.Tensor:
        """从图像中提取ViT特征（768维）"""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            visual_feat = self.vit(image_tensor).squeeze()  # shape: (768,)
        return visual_feat

    def extract_semantic_feature(self, text: str) -> torch.Tensor:
        """从文本中提取BERT特征（768维）"""
        inputs = self.bert_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            semantic_feat = outputs.last_hidden_state.mean(dim=1).squeeze()  # shape: (768,)
        return semantic_feat
