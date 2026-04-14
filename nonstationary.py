import numpy as np

class NonstationaryBandit:
    def __init__(self, k):
        self.k = k
        # Giá trị thật ban đầu của mỗi arm
        self.q_true = np.random.normal(0, 1, k)

    def step(self, action):
        # Nonstationary: q_true thay đổi theo thời gian (random walk)
        self.q_true += np.random.normal(0, 0.01, self.k)

        # Reward lấy từ phân phối chuẩn quanh q_true[action]
        return np.random.normal(self.q_true[action], 1)


def epsilon_greedy(Q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(Q))  # explore
    else:
        return np.argmax(Q)              # exploit

k = 10
steps = 1000
epsilon = 0.1
alpha = 0.1   # 🔥 quan trọng: learning rate cố định

bandit = NonstationaryBandit(k)

Q = np.zeros(k)   # ước lượng ban đầu
rewards = []

for t in range(steps):
    action = epsilon_greedy(Q, epsilon)     # chọn hành động
    R = bandit.step(action)                 # nhận reward
    rewards.append(R)
    Q[action] += alpha * (R - Q[action])

# Kết quả
print("Estimated Q:", Q)
print("True q*:", bandit.q_true)
print("Average reward:", np.mean(rewards))