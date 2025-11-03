import numpy as np


class LifeGameScrambling:
    def __init__(self, block_dim=(8, 8)):
        self.block_dim = block_dim

    def stp_multiply(self, matrix_a, matrix_b):
        m, n = matrix_a.shape
        p, q = matrix_b.shape
        if n % p == 0:
            factor = n // p
            output = np.zeros((m, factor * q))
            for row in range(m):
                for col in range(factor * q):
                    total = 0
                    for idx in range(p):
                        total += matrix_a[row, idx * factor + col // q] * matrix_b[idx, col % q]
                    output[row, col] = total
            return output
        else:
            return np.kron(matrix_a, matrix_b)

    def process_block(self, data_block, selection_ratio=0.5):
        rows, cols = data_block.shape
        m_dim, n_dim = self.block_dim

        data_blocks = []
        positions = []

        for i in range(0, rows, m_dim):
            for j in range(0, cols, n_dim):
                block = data_block[i:i + m_dim, j:j + n_dim]
                if block.shape == (m_dim, n_dim):
                    data_blocks.append(block)
                    positions.append((i, j))

        block_count = len(data_blocks)
        transform_matrix = np.array([[1, 0, 0, 0], [0, 1, 1, 1]])

        selected_count = int(block_count * selection_ratio)
        chosen_indices = np.random.choice(block_count, selected_count, replace=False)

        for idx in chosen_indices:
            current_block = data_blocks[idx].copy()
            block_rows, block_cols = current_block.shape
            flat_data = current_block.flatten()
            new_flat = np.zeros_like(flat_data)

            for pos in range(len(flat_data)):
                current_val = flat_data[pos]
                vector_form = np.array([current_val, 1 - current_val])

                adjacent = []
                neighbor_positions = [pos - 1, pos + 1, pos - block_cols, pos + block_cols,
                                      pos - block_cols - 1, pos - block_cols + 1,
                                      pos + block_cols - 1, pos + block_cols + 1]

                for neighbor in neighbor_positions:
                    if 0 <= neighbor < len(flat_data):
                        adjacent.append(flat_data[neighbor])

                neighbor_sum = sum(adjacent)

                if neighbor_sum == 3:
                    logic_vec = np.array([1, 0])
                elif neighbor_sum == 2:
                    logic_vec = np.array([0, 1])
                else:
                    logic_vec = np.array([0, 1])

                new_val = self.stp_multiply(logic_vec.reshape(1, -1), vector_form.reshape(-1, 1))
                new_flat[pos] = 1 if new_val[0, 0] > 0.5 else 0

            updated_block = new_flat.reshape((block_rows, block_cols))
            data_blocks[idx] = updated_block

        result = np.zeros_like(data_block)
        for (i, j), block in zip(positions, data_blocks):
            result[i:i + m_dim, j:j + n_dim] = block

        return result