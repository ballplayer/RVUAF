### pip install openai sentence-transformers pillow json

from Initialization import KnowledgeGraph
from MLLMcallGPT4VandLaVA import call_multimodal_llm
from Entitymatchinglogic import match_entities
from Updatehistoricalknowledgegraph import update_kg
import json

def main(image_path: str):
    """模块一主函数：从图像生成并更新KG"""
    # 初始化历史KG
    history_kg = KnowledgeGraph()

    # 1. 调用LLM获取当前帧KG
    current_kg = call_multimodal_llm(image_path)
    if not current_kg:
        return history_kg

    # 2. 实体匹配
    entity_matches = match_entities(current_kg["entities"], history_kg)

    # 3. 更新历史KG
    updated_kg = update_kg(history_kg, current_kg, entity_matches)

    # 输出结果
    print("更新后的KG实体：", json.dumps(updated_kg.entities, indent=2))
    print("更新后的KG关系：", json.dumps(updated_kg.relations, indent=2))

    return updated_kg


# 测试运行（替换为你的图像路径）
if __name__ == "__main__":
    main("test_image.jpg")
