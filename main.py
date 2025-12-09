import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# 1. MODELO DQN
# =====================================================
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# =====================================================
# 2. FUNÇÃO DE TREINO P/ GRID SEARCH
# =====================================================
def train_dqn(config):
    env = gym.make("CartPole-v1", render_mode="none")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01

    policy_net = DQN(state_size, action_size, config["hidden"]).to(device)
    target_net = DQN(state_size, action_size, config["hidden"]).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=config["lr"])
    memory = deque(maxlen=100000)

    def act(state, epsilon):
        if random.random() < epsilon:
            return random.randrange(action_size)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            return int(policy_net(state_t).argmax(1).item())

    def replay(batch_size):
        if len(memory) < batch_size:
            return None
        minibatch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states_t = torch.FloatTensor(np.array(states)).to(device)
        actions_t = torch.LongTensor(np.array(actions)).to(device)
        rewards_t = torch.FloatTensor(np.array(rewards)).to(device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(device)
        dones_t = torch.FloatTensor(np.array(dones)).to(device)

        q_values = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = policy_net(next_states_t).argmax(1)
            next_q_values = target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        targets = rewards_t + gamma * next_q_values * (1 - dones_t)

        loss = nn.MSELoss()(q_values, targets)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
        optimizer.step()

    episodes = config["episodes"]
    episode_rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = act(state, epsilon)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            memory.append((state, action, reward, next_state, float(done)))
            state = next_state
            total_reward += reward
            replay(config["batch"])

        epsilon = max(epsilon_min, epsilon * config["eps_decay"])

        if ep % config["target_update"] == 0:
            target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(total_reward)

    env.close()
    return np.mean(episode_rewards[-20:])  # score final


# =====================================================
# 3. GRID SEARCH
# =====================================================
param_grid = {
    "lr": [1e-3, 5e-4],
    "batch": [32, 64],
    "hidden": [64, 128],
    "eps_decay": [0.995, 0.990],
    "target_update": [2, 5],
    "episodes": [300]
}

keys = list(param_grid.keys())
best_score = -9999
best_config = None
results = []

print("\n====== INICIANDO GRID SEARCH ======\n")

for combo in itertools.product(*param_grid.values()):
    config = dict(zip(keys, combo))
    print(f"Testando config: {config}")

    score = train_dqn(config)
    results.append((config, score))

    print(f"→ Score médio final: {score:.2f}\n")

    if score > best_score:
        best_score = score
        best_config = config

print("\n🔝 MELHOR CONFIGURAÇÃO ENCONTRADA:")
print(best_config)
print(f"Score: {best_score:.2f}\n")


# =====================================================
# 4. TREINO FINAL COM MELHOR CONFIG
# =====================================================
print("Treinando modelo final com a melhor configuração...\n")

env = gym.make("CartPole-v1", render_mode="none")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = best_config["eps_decay"]
lr = best_config["lr"]
batch_size = best_config["batch"]
hidden = best_config["hidden"]
target_update_freq = best_config["target_update"]
max_episodes = 10000
print_freq = 25
save_path = "dqn_cartpole_perfeito.pth"

policy_net = DQN(state_size, action_size, hidden).to(device)
target_net = DQN(state_size, action_size, hidden).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
memory = deque(maxlen=100000)

def act(state, epsilon):
    if random.random() <= epsilon:
        return random.randrange(action_size)
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = policy_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

def replay():
    if len(memory) < batch_size:
        return None
    minibatch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)

    states_t = torch.FloatTensor(np.array(states)).to(device)
    actions_t = torch.LongTensor(np.array(actions)).to(device)
    rewards_t = torch.FloatTensor(np.array(rewards)).to(device)
    next_states_t = torch.FloatTensor(np.array(next_states)).to(device)
    dones_t = torch.FloatTensor(np.array(dones)).to(device)

    q_values = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_actions = policy_net(next_states_t).argmax(1)
        next_q_values = target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)

    targets = rewards_t + (gamma * next_q_values * (1 - dones_t))

    loss = nn.MSELoss()(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
    optimizer.step()

    return loss.item()


# LOOP FINAL DE TREINO
episode_rewards = []
avg_rewards = [0.0]
losses = []
perfect_episodes = 0
episode = 0

while avg_rewards[-1] < 450 and episode < max_episodes:
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

        loss = replay()
        if loss is not None:
            losses.append(loss)

    episode_rewards.append(total_reward)
    avg_rewards.append(np.mean(episode_rewards[-100:]))

    if epsilon > epsilon_min:
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if total_reward >= 500:
        perfect_episodes += 1

    if episode % print_freq == 0:
        print(f"📊 Episódio {episode} | Rew: {total_reward:.2f} | Média(100): {avg_rewards[-1]:.2f} | Eps: {epsilon:.3f}")

print(f"\n✅ Treinamento concluído em {episode} episódios. Perfeitos: {perfect_episodes}")

torch.save(policy_net.state_dict(), save_path)
env.close()


# =====================================================
# 5. TESTE FINAL COM RENDER
# =====================================================
env = gym.make("CartPole-v1", render_mode="human")
state, _ = env.reset()
done = False
total_reward = 0

print("\n🎬 Mostrando agente final...")
input("Pressione ENTER para iniciar...")

while not done:
    time.sleep(0.05)
    action = act(state, epsilon=0.0)
    state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"\nPontuação final do agente treinado: {total_reward}")
env.close()


# =====================================================
# 6. GRÁFICOS
# =====================================================
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.plot(episode_rewards, label="Recompensa por episódio", alpha=0.7)
plt.plot(avg_rewards, label="Média móvel (100)", color='red')
plt.title("Evolução da Recompensa")
plt.xlabel("Episódio")
plt.ylabel("Recompensa")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(losses, label="Loss", color="purple", alpha=0.7)
plt.title("Evolução da Perda (Loss)")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 3, 3)
weights = policy_net.fc1.weight.detach().cpu().numpy()
sns.heatmap(weights, cmap="coolwarm", cbar=True)
plt.title("Heatmap dos Pesos da Primeira Camada")

plt.tight_layout()
plt.show()
