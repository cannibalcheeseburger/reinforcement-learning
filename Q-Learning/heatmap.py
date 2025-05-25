import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load Q-table from pickle file
pickle_path = './Q_table_8x8.pkl'
with open(pickle_path, 'rb') as f:
    Q = pickle.load(f)

# Aggregate max Q-value per state and reshape to 8x8 grid
q_values_grid = np.max(Q, axis=1).reshape((8, 8))

plt.figure(figsize=(6, 6))
plt.imshow(q_values_grid, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Q-value')
plt.title('Learned Q-values for each state')
plt.grid(True)

for i in range(8):
    for j in range(8):
        plt.text(j, i, f'{q_values_grid[i, j]:.2f}', ha='center', va='center', color='black')

plt.show()
