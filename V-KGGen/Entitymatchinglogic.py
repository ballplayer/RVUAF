from Initialization import compute_semantic_sim, compute_attr_overlap


def match_entities(current_entities: list, history_kg) -> list:
    """对比当前实体与历史KG，返回匹配结果"""
    matches = []
    current_entity_map = {ent["id"]: ent for ent in current_entities}

    for current_id, current_ent in current_entity_map.items():
        best_match_id = None
        best_score = 0.0

        # 遍历历史实体找最优匹配
        for history_ent in history_kg.entities:
            sim_score = compute_semantic_sim(current_ent["name"], history_ent["name"])
            overlap_score = compute_attr_overlap(current_ent["attributes"], history_ent["attributes"])
            total_score = 0.5 * sim_score + 0.5 * overlap_score  # 加权综合得分

            if total_score > best_score and total_score >= 0.7:  # 阈值可调整
                best_score = total_score
                best_match_id = history_ent["id"]

        # 生成匹配结果
        if best_match_id:
            matches.append({"current_id": current_id, "history_id": best_match_id, "is_new": False})
        else:
            new_id = f"e{history_kg.next_entity_id}"
            history_kg.next_entity_id += 1
            matches.append({"current_id": current_id, "history_id": new_id, "is_new": True})

    return matches
