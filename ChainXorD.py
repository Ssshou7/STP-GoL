import numpy as np


class Diffusion:
    def __init__(self):
        pass

    def apply_diffusion(self, data_blocks, sequence):
        block_count = len(data_blocks)
        rows, cols = data_blocks[0].shape
        total_elements = rows * cols

        vector_list = []
        for i in range(block_count):
            vector = data_blocks[i].flatten()
            vector_list.append(vector)

        vector_matrix = np.array(vector_list)

        if len(sequence) < total_elements * block_count:
            extended = []
            while len(extended) < total_elements * block_count:
                extended.extend(sequence)
            sequence = np.array(extended[:total_elements * block_count])

        integer_keys = (np.abs(sequence[:total_elements * block_count] * 256) % 256).astype(int)
        binary_keys = []
        for num in integer_keys:
            binary_rep = format(num, '08b')
            for bit in binary_rep:
                binary_keys.append(int(bit))

        key_matrix = np.array(binary_keys[:total_elements * block_count])
        key_matrix = key_matrix.reshape((block_count, total_elements))

        forward_result = np.zeros((block_count, total_elements), dtype=int)
        forward_result[:, 0] = vector_matrix[:, 0] ^ key_matrix[:, 0]

        for pos in range(1, total_elements):
            forward_result[:, pos] = vector_matrix[:, pos] ^ key_matrix[:, pos] ^ forward_result[:, pos - 1]

        final_result = np.zeros((block_count, total_elements), dtype=int)
        final_result[:, -1] = forward_result[:, -1] ^ key_matrix[:, -1]

        for pos in range(total_elements - 2, -1, -1):
            final_result[:, pos] = forward_result[:, pos] ^ key_matrix[:, pos] ^ final_result[:, pos + 1]

        diffused_blocks = []
        for i in range(block_count):
            diffused_block = final_result[i, :].reshape((rows, cols))
            diffused_blocks.append(diffused_block)

        return diffused_blocks