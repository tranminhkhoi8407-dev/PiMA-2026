# 🎰 K-Armed Bandit — Reinforcement Learning

Implement thuật toán **ε-greedy** cho bài toán **k-armed bandit** — một bài toán kinh điển trong Reinforcement Learning về sự đánh đổi giữa **Exploration vs Exploitation**.

## 📌 Giới thiệu

Bài toán k-armed bandit mô phỏng tình huống: bạn có k cần gạt (arms), mỗi lần kéo cho phần thưởng ngẫu nhiên với phân phối khác nhau. Mục tiêu là tối đa hóa tổng phần thưởng theo thời gian.

Dự án này được thực hiện trong khuôn khổ chương trình **PiMA 2026**.

## 📁 Files

| File | Mô tả |
|---|---|
| `bandit.py` | Stationary bandit + ε-greedy |
| `nonstationary.py` | Non-stationary bandit (phần thưởng thay đổi theo thời gian) |
| `k-armed bandit lecture.pdf` | Slide bài giảng lý thuyết |

## ⚙️ Thuật toán

### ε-Greedy
- Với xác suất **ε** → chọn ngẫu nhiên (exploration)
- Với xác suất **1-ε** → chọn arm có Q(a) cao nhất (exploitation)
- Cập nhật: `Q(a) ← Q(a) + 1/N(a) × [R - Q(a)]`

## 🚀 Chạy thử

```bash
pip install numpy
python bandit.py
python nonstationary.py
```

## 📦 Requirements
numpy
## 📚 References

- Sutton & Barto — *Reinforcement Learning: An Introduction* (Chapter 2)
- PiMA 2026 Lecture Notes
