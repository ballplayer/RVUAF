from stable_baselines3 import PPO

def train_ppo_model(env, total_timesteps: int = 10000) -> PPO:
    """训练PPO强化学习模型"""
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01
    )
    model.learn(total_timesteps=total_timesteps)
    model.save("rpri_kgr_ppo_model")
    return model

# 训练模型
model = train_ppo_model(env)
