def update_kg(history_kg, current_kg: dict, entity_matches: list):
    """合并当前KG与历史KG，返回更新后的KG"""
    # 1. 更新实体
    current_entity_map = {ent["id"]: ent for ent in current_kg["entities"]}
    for match in entity_matches:
        current_ent = current_entity_map[match["current_id"]]
        if match["is_new"]:
            # 添加新实体
            history_kg.entities.append({
                "id": match["history_id"],
                "name": current_ent["name"],
                "attributes": current_ent["attributes"]
            })
        else:
            # 更新旧实体属性
            history_ent = history_kg.get_entity_by_id(match["history_id"])
            if history_ent:
                history_ent["attributes"] = current_ent["attributes"]

    # 2. 更新关系（去重）
    current_relation_map = {(rel["head"], rel["relation"], rel["tail"]): rel for rel in current_kg["relations"]}
    history_relation_set = set((r["head"], r["relation"], r["tail"]) for r in history_kg.relations)

    for rel_tuple, rel in current_relation_map.items():
        if rel_tuple not in history_relation_set:
            history_kg.relations.append(rel)

    return history_kg
