from modelscope import pipeline


class QuestionParser:
    def __init__(self):
        # 实体提取：用NER模型
        self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        # 关系提取：用文本分类模型（自定义训练或规则匹配）
        self.relation_keywords = {"on_top_of": ["on top of", "above"], "next_to": ["next to", "beside"]}

    def extract_source_entity(self, question: str) -> str:
        """提取问题中的源实体"""
        ner_results = self.ner_pipeline(question)
        if not ner_results:
            return None
        # 合并连续实体（如"story book"）
        entities = []
        current_entity = ""
        for res in ner_results:
            if res["entity"].startswith("B-"):
                if current_entity:
                    entities.append(current_entity.strip())
                current_entity = res["word"]
            elif res["entity"].startswith("I-"):
                current_entity += " " + res["word"]
        if current_entity:
            entities.append(current_entity.strip())
        return entities[0] if entities else None

    def extract_target_relation(self, question: str) -> str:
        """提取问题中的目标关系"""
        question_lower = question.lower()
        for rel, keywords in self.relation_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return rel
        return None


# 测试问题解析
parser = QuestionParser()
question = "What entity is the cup on top of?"
source_entity = parser.extract_source_entity(question)
target_relation = parser.extract_target_relation(question)
print(f"源实体：{source_entity}, 目标关系：{target_relation}")
