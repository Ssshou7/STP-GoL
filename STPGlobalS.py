import numpy as np


class GlobalScrambling:
    def __init__(self):
        pass

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

    def apply_scrambling(self, data_block, sequence):
        rows, cols = data_block.shape
        total_elements = rows * cols

        if len(sequence) < total_elements:
            extended = []
            while len(extended) < total_elements:
                extended.extend(sequence)
            sequence = np.array(extended[:total_elements])

        sorted_indices = np.argsort(sequence[:total_elements])

        permutation_matrix = np.zeros((total_elements, total_elements), dtype=int)
        for new_pos, old_pos in enumerate(sorted_indices):
            permutation_matrix[new_pos, old_pos] = 1

        flat_data = data_block.flatten()
        scrambled_flat = self.stp_multiply(permutation_matrix, flat_data.reshape(-1, 1))
        scrambled_block = scrambled_flat.reshape((rows, cols))

        return scrambled_block