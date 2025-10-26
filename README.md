
# Robot Vision Enhancement Framework (RVF)  
**论文复现：机器人视觉理解增强框架——基于大语言模型生成的强化图像知识图谱**  

本项目实现了论文中的三大核心模块（V-KGGen、RPri-KGR、RAV-KG）及闭环系统，支持机器人视觉场景的知识图谱生成、推理和视觉-语义对齐。


## 🚀 快速开始  
### 1. 环境配置  
#### 1.1 硬件要求  
- **GPU**：推荐NVIDIA GPU（显存≥8GB，支持CUDA 11.7+）  
- **CPU**：Intel i7或等价处理器（备用选项）  

#### 1.2 软件依赖  
- Python 3.8~3.11  
- PyTorch 2.0+  
- 其他依赖：见`requirements.txt`  

#### 1.3 安装步骤  
```bash
# 克隆仓库（假设已创建）
git clone https://github.com/ballplayer/Robot-Vision-Understanding-Augment-Framework.git && cd rvf

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 配置OpenAI API密钥（必填，用于V-KGGen模块）
export OPENAI_API_KEY="your-api-key"  # Linux/macOS
set OPENAI_API_KEY="your-api-key"     # Windows
```


### 2. 数据集准备  
#### 2.1 数据集结构  
```
dataset/
├── images/                # 机器人视角图像序列（JPG/PNG）
├── annotations.jsonl      # 图像+KG+QA标注（JSON Lines格式）
└── preprocessed/          # 预提取的特征（可选）
```  

#### 2.2 生成数据集  
使用脚本自动生成标注：  
```bash
# 生成KG（基于图像+LLM）
python scripts/generate_kg.py --image_dir dataset/images --output dataset/annotations.jsonl

# 生成QA对（基于KG）
python scripts/generate_qa.py --kg_file dataset/annotations.jsonl --output dataset/qa_pairs.jsonl
```  

#### 2.3 公开数据集推荐  
- **图像**：COCO、Visual Genome、RobotScene  
- **KG**：Freebase、DBpedia（需适配格式）  


### 3. 模块运行示例  
#### 3.1 V-KGGen模块（生成/更新KG）  
```python
from v_kggen_module import ClosedLoopSystem

# 初始化系统
system = ClosedLoopSystem(initial_kg=nx.DiGraph(), question="What is the cup on?")

# 运行V-KGGen
updated_kg = system.run_v_kggen(image_path="dataset/images/frame_001.jpg")
print("Updated KG:", nx.readwrite.json_graph.node_link_data(updated_kg))
```  

#### 3.2 RPri-KGR模块（推理答案）  
```python
answer = system.run_rpri_kgr()
print("RPri-KGR Answer:", answer)
```  

#### 3.3 RAV-KG模块（视觉-语义对齐）  
```python
align_score = system.run_rav_kg(image_path="dataset/images/frame_001.jpg", answer=answer)
print("Align Score:", align_score)
```  


### 4. 闭环系统启动  
```bash
python main.py --image_dir dataset/images --question "What is the red cup on?" --max_iter 5
```  

**预期输出**：  
```
=== Iteration 1/5 ===
Running V-KGGen...
Running RPri-KGR...
RPri-KGR Answer: table
Running RAV-KG...
Current Align Score: 0.6523
...
=== Loop Finished ===
Final Align Score: 0.8215
Optimized ViT saved to: optimized_vit_closed_loop.pth
```  


## 📚 模块说明  
| 模块          | 功能                          | 输入输出                                                                 |  
|---------------|-------------------------------|--------------------------------------------------------------------------|  
| **V-KGGen**   | 图像→知识图谱生成/更新         | 输入：图像路径；输出：更新后的KG（NetworkX格式）                          |  
| **RPri-KGR**  | KG→问题推理                   | 输入：KG+问题；输出：答案实体                                             |  
| **RAV-KG**    | 视觉-语义对齐优化             | 输入：图像+答案；输出：对齐分数+优化后的ViT模型                          |  
| **闭环系统**  | 迭代优化（V→K→R→A）           | 输入：图像序列+问题；输出：优化模型+最终KG+对齐分数                      |  


### 5. 注意事项  
- **API密钥安全**：不要硬编码密钥，使用环境变量或`.env`文件  
- **硬件加速**：确保CUDA可用（PyTorch需安装对应版本）  
- **模型预训练**：RPri-KGR模块需要预训练的PPO模型，运行`scripts/train_rpri_kgr.py`生成  
- **常见问题**：  
  - 依赖冲突：使用`pip install --upgrade [package]`解决  
  - LLM调用失败：检查API密钥和网络连接  


### 6. 贡献指南  
- 欢迎提交PR（修复bug、优化代码、添加新功能）  
- 报告问题：提交Issue（附错误日志和复现步骤）  


### 7. 许可证  
本项目采用MIT许可证，详见`LICENSE`文件。  

---  
**联系作者**：lingfengfeng@ruc.edu.cn  
**项目地址**：https://github.com/ballplayer/RVUAF
**论文引用**：under-review  
