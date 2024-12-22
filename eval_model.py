import supersuit as ss
from stable_baselines3 import PPO
from social_dilemmas.envs.map_env import MapEnv
from stable_baselines3.common import vec_env
import pygame
from sb3_contrib import RecurrentPPO
from social_dilemmas.envs.cleanup import CleanupEnv

import time
GRID_SIZE = 30

def draw(rgb_array, screen, height, width):
    for y in range(height):
        for x in range(width):
            color = rgb_array[y, x]
            rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(screen, color, rect)

def evaluate_model(model_path, env_name="cleanup", num_agents=5, num_episodes=1):
    env = CleanupEnv(num_agents=num_agents)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
    env = vec_env.VecMonitor(env)
    env = vec_env.VecTransposeImage(env, True)

    model = PPO.load(model_path)
    pygame.init()

    # org_env = MapEnv.ENVS[-1]
    org_env = env
    rgb_arr = org_env.render(mode="array")
    height, width, _ = rgb_arr.shape
    screen = pygame.display.set_mode((width * GRID_SIZE, height * GRID_SIZE))

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            action, _states = model.predict(obs, deterministic=True)  
            obs, rewards, dones, infos = env.step(action) 
            
            draw(rgb_arr, screen, height, width)
            pygame.display.update()
            rgb_arr = org_env.render(mode="array")

            total_reward += sum(rewards)
            done = all(dones) 
            print(action)
            time.sleep(0.2)
        print(f"ep {episode + 1}: total Reward: {total_reward}")
    
    env.close()
    pygame.quit()

if __name__ == "__main__":
    evaluate_model("results/sb3/cleanup_ppo_paramsharing/best_model")