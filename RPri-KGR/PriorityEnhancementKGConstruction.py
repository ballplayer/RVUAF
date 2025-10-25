import networkx as nx


def build_priority_kg():
    """构建优先级增强的知识图谱（示例）"""
    G = nx.DiGraph()

    # 添加实体节点（含优先级权重）
    entities = [
        ("cup", {"weight": 0.8, "attributes": {"color": "red", "size": "small"}}),
        ("book", {"weight": 0.9, "attributes": {"color": "blue", "type": "storybook"}}),
        ("table", {"weight": 0.7, "attributes": {"material": "wood", "shape": "rectangular"}}),
        ("chair", {"weight": 0.6, "attributes": {"color": "brown", "position": "next_to_table"}})
    ]
    G.add_nodes_from(entities)

    # 添加关系边（含关系类型、优先级权重）
    relations = [
        ("cup", "book", {"relation": "next_to", "weight": 0.7}),
        ("book", "table", {"relation": "on_top_of", "weight": 0.8}),
        ("cup", "table", {"relation": "on_top_of", "weight": 0.6}),
        ("chair", "table", {"relation": "next_to", "weight": 0.5})
    ]
    G.add_edges_from(relations)

    return G


# 测试KG构建
kg = build_priority_kg()
print("KG节点：", kg.nodes(data=True))
print("KG边：", kg.edges(data=True))
