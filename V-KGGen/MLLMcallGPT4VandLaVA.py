import openai

def call_multimodal_llm(image_path: str) -> dict:
    """调用GPT-4V获取当前帧的KG三元组"""
    v_kggen_prompt = """
请分析图像并返回严格JSON格式的结果：
1. entities: 实体列表，每个实体含id(e1/e2...)、name、attributes（颜色/大小/位置等）；
2. relations: 关系列表，每个关系含head(实体id)、relation(如on_top_of)、tail(实体id)；
注意：无额外文字，仅返回JSON！
"""

    # 调用GPT-4V
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": v_kggen_prompt},
                    {"type": "image_url", "image_url": {"url": f"file://{image_path}"}}
                ]
            }
        ],
        max_tokens=1024
    )

    # 解析返回结果（提取纯JSON）
    content = response.choices[0].message.content
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        print("LLM返回格式错误，请检查prompt或图像质量")
        return None
