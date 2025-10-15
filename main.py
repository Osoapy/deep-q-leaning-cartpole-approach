import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import os

# --- Configuração de dispositivo ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Configurações básicas ---
env = gym.make("CartPole-v1", render_mode="none")
state_size = env.observation_space.shape[0]  # 4
action_size = env.action_space.n             # 2

# Hiperparâmetros (ajusta se quiser)
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
lr = 1e-3
batch_size = 64
memory = deque(maxlen=20000)
target_update_freq = 2       # atualizar target_net a cada N episódios
max_episodes = 10000         # limite alto por segurança
solved_score = 495.0         # média móvel de 100 episódios considerada "perfeita"
save_path = "dqn_cartpole_perfeito.pth"

# --- 2. Rede Neural ---
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=lr)

# --- Funções auxiliares ---
def act(state, epsilon):
    """ε-greedy action selection"""
    if random.random() <= epsilon:
        return random.randrange(action_size)
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = policy_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

def replay():
    """Treinamento com amostra do replay buffer"""
    if len(memory) < batch_size:
        return None
    minibatch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)

    states_t = torch.FloatTensor(states).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    rewards_t = torch.FloatTensor(rewards).to(device)
    next_states_t = torch.FloatTensor(next_states).to(device)
    dones_t = torch.FloatTensor(dones).to(device)

    # Q(s,a) atual
    q_values = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

    # Q_target = r + gamma * max_a' Q_target(s', a') * (1 - done)
    with torch.no_grad():
        next_q_values = target_net(next_states_t).max(1)[0]
    targets = rewards_t + (gamma * next_q_values * (1 - dones_t))

    loss = nn.MSELoss()(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    # clipping de gradiente pra estabilidade
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
    optimizer.step()

    return loss.item()

# --- Loop de treinamento até "perfeição" ---
episode_rewards = deque(maxlen=100)  # pra média móvel de 100 episódios

episode = 0
best_avg = -np.inf

while episode < max_episodes:
    episode += 1
    state, _ = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action = act(state, epsilon)
        next_state, reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated

        memory.append((state, action, reward, next_state, float(done)))
        state = next_state
        total_reward += reward

        replay()

    # pós-episódio
    episode_rewards.append(total_reward)
    avg_reward = np.mean(episode_rewards)

    # decai epsilon
    if epsilon > epsilon_min:
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # atualiza target_net periodicamente
    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # salvar modelo quando melhora média
    if avg_reward > best_avg:
        best_avg = avg_reward
        torch.save(policy_net.state_dict(), save_path)

    # logs úteis
    print(f"Eps {episode:4d} | Rew {total_reward:6.1f} | Avg100 {avg_reward:6.2f} | EpsGreedy {epsilon:.3f}")

    # Condição de parada: média móvel de 100 episódios >= solved_score
    if len(episode_rewards) == 100 and avg_reward >= solved_score:
        print(f"\n🟢 PARABÉNS — atingido avg100 >= {solved_score:.1f} em {episode} episódios.")
        print(f"Modelo salvo em: {os.path.abspath(save_path)}")
        break

# caso não consiga atingir nos max_episodes
if episode >= max_episodes:
    print(f"\n🔴 Atingido limite de episódios ({max_episodes}). Melhor avg100 registrada: {best_avg:.2f}")
    print(f"Modelo do melhor checkpoint salvo em: {os.path.abspath(save_path)}")

env.close()
