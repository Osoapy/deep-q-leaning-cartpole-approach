import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# --- Configuração de dispositivo ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Configurações básicas ---
env = gym.make("CartPole-v1", render_mode="none")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hiperparâmetros
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
lr = 1e-3
batch_size = 64
memory = deque(maxlen=100000)
target_update_freq = 2
print_freq = 25
max_episodes = 10000
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
        next_actions = policy_net(next_states_t).argmax(1)  # escolhe ação com policy
        next_q_values = target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)  # avalia no target

    targets = rewards_t + (gamma * next_q_values * (1 - dones_t))

    loss = nn.MSELoss()(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
    optimizer.step()

    return loss.item()

# --- Loop de treinamento ---
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

    # pós-episódio
    episode_rewards.append(total_reward)
    avg_rewards.append(np.mean(episode_rewards[-100:]))

    # decai epsilon
    if epsilon > epsilon_min:
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # atualiza target_net periodicamente
    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # checa se foi um episódio perfeito
    if total_reward >= 500:
        perfect_episodes += 1

    # imprime progresso
    if episode % print_freq == 0:
        print(f"📊 Eps {episode} | Rew: {total_reward:.2f} | Avg(100): {avg_rewards[-1]:.2f} | Epsilon: {epsilon:.3f}")

# --- Após 25 episódios perfeitos ---
print(f"\n✅ Treinamento concluído após {episode} episódios ({perfect_episodes:.0f}/25 perfeitos).")
torch.save(policy_net.state_dict(), save_path)
env.close()

# --- Teste gráfico ---
env = gym.make("CartPole-v1", render_mode="human")
state, _ = env.reset()
done = False
total_reward = 0

print("\n🎬 Mostrando agente final aprendendo em tempo real...")
input("Pressione Enter para iniciar...")
while not done:
    time.sleep(0.05)  # controla a velocidade da simulação
    action = act(state, epsilon=0.0)  # sempre greedy agora
    state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"\nPontuação final do agente treinado: {total_reward}")
env.close()

# --- Gráficos de desempenho ---
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
