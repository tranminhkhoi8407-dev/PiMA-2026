import numpy as np  # thư viện để xử lý số và random

# =========================
# Tạo môi trường bandit
# =========================
class Bandit:
    def __init__(self, k):
        self.k = k  # số lượng arm (hành động)
        #np.random.normal(loc, scale, size); loc: mean, scale: std, size: số lượng phần tử
        self.q_true = np.random.normal(0, 1, k)  
        # giá trị thật q*(a), mỗi arm có mean khác nhau (không biết trước)
    def pull(self, action):
        # khi chọn action, trả về reward ngẫu nhiên
        return np.random.normal(self.q_true[action], 1)
        # reward ~ N(q*(a), 1)


# =========================
# Thuật toán ε-greedy
# =========================
def epsilon_greedy(k=10, steps=1000, epsilon=0.1):
    
    bandit = Bandit(k)  # tạo môi trường
    
    Q = np.zeros(k)     # Q(a) ← 0 (ước lượng ban đầu)
    N = np.zeros(k)     # N(a) ← 0 (số lần chọn mỗi action)

    rewards = []        # lưu reward để theo dõi

    for t in range(steps):  # loop forever (ở đây chạy steps lần)

        if np.random.rand() < epsilon:
            action = np.random.randint(k)  
            # với xác suất ε → chọn random (exploration)
        else:
            action = np.argmax(Q)          
            # với xác suất 1-ε → chọn greedy (exploitation)
    
        # Nhận reward
        R = bandit.pull(action)  
        # R ← bandit(A)
        # Cập nhật số lần chọn
        N[action] += 1           
        # N(A) ← N(A) + 1

        # =========================
        # Cập nhật Q (incremental update)
        # =========================
        Q[action] += (1 / N[action]) * (R - Q[action])  
        # Q(A) ← Q(A) + 1/N(A) * [R - Q(A)]

        rewards.append(R)        # lưu lại reward

    return Q, rewards, bandit.q_true
    # trả về: estimate, reward, và giá trị thật để so sánh


# =========================
# Chạy thử
# =========================
Q, rewards, q_true = epsilon_greedy(k=10, steps=1000, epsilon=0.1)

print("Estimated Q:", Q)     # giá trị ước lượng
print("True q*:", q_true)   # giá trị thật (để kiểm tra)
print("Average reward:", np.mean(rewards))  # reward trung bình