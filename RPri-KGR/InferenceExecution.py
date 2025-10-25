def infer_answer(model, env) -> tuple:
    """用训练好的模型推理答案"""
    obs = env.reset()
    done = False
    path = [env.current_entity]

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        path.append(env.current_entity)
        env.render()

    return path, env.current_entity


# 测试推理
path, answer = infer_answer(model, env) # 可注释掉
print(f"推理路径：{path}")
print(f"问题答案：{answer}")
