import numpy as np
import cv2
from scipy import linalg



class STPImageEncryption:
    def __init__(self, block_size=(8, 8), alpha=0.92, beta=0.81):
        self.block_size = block_size
        self.alpha = alpha
        self.beta = beta
        self.phi = 1

    def bit_plane_decomposition(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape
        bit_planes = []
        for i in range(8):
            bit_plane = ((image >> i) & 1).astype(np.uint8)
            bit_planes.append(bit_plane)
        return bit_planes

    def bit_plane_composition(self, bit_planes):
        height, width = bit_planes[0].shape
        image = np.zeros((height, width), dtype=np.uint8)
        for i, bit_plane in enumerate(bit_planes):
            image |= (bit_plane.astype(np.uint8) < i)
        return image

    def semi_tensor_product(self, A, B):
        m, n = A.shape
        p, q = B.shape
        if n % p == 0:
            k = n // p
            result = np.zeros((m, k * q))
            for i in range(m):
                for j in range(k * q):
                    block_sum = 0
                    for t in range(p):
                        block_sum += A[i, t * k + j // q] * B[t, j % q]
                    result[i, j] = block_sum
            return result
        else:
            return np.kron(A, B)

    def game_of_life_scrambling(self, bit_plane, G_ratio=0.5):
        height, width = bit_plane.shape

        blocks = []
        block_positions = []

        for i in range(0, height, m2):
            for j in range(0, width, n2):
                block = bit_plane[i:i + m2, j:j + n2]
                if block.shape == (m2, n2):
                    blocks.append(block)
                    block_positions.append((i, j))

        num_blocks = len(blocks)

        G = int(num_blocks * G_ratio)
        selected_indices = np.random.choice(num_blocks, G, replace=False)

        for idx in selected_indices:
            block = blocks[idx].copy()
            block_height, block_width = block.shape
            flat_block = block.flatten()
            new_flat_block = np.zeros_like(flat_block)

            for i in range(len(flat_block)):
                x_i = flat_block[i]
                x_i_vec = np.array([x_i, 1 - x_i])

                neighbors = []
                positions = [i - 1, i + 1, i - block_width, i + block_width,
                             i - block_width - 1, i - block_width + 1,
                             i + block_width - 1, i + block_width + 1]

                for pos in positions:
                    if 0 <= pos < len(flat_block):
                        neighbors.append(flat_block[pos])

                N_i = sum(neighbors)

                if N_i == 3:
                    L_i = np.array([1, 0])
                elif N_i == 2:
                    L_i = np.array([0, 1])
                else:
                    L_i = np.array([0, 1])

                x_new = self.semi_tensor_product(L_i.reshape(1, -1), x_i_vec.reshape(-1, 1))
                new_flat_block[i] = 1 if x_new[0, 0] > 0.5 else 0

            new_block = new_flat_block.reshape((block_height, block_width))
            blocks[idx] = new_block

        scrambled_bit_plane = np.zeros_like(bit_plane)
        for (i, j), block in zip(block_positions, blocks):
            scrambled_bit_plane[i:i + m2, j:j + n2] = block

        return scrambled_bit_plane

    def chaotic_global_scrambling( bit_plane, chaotic_seq):
        height, width = bit_plane.shape
        total_pixels = height * width

        if len(chaotic_seq) < total_pixels:
            extended_seq = []
            while len(extended_seq) < total_pixels:
                extended_seq.extend(chaotic_seq)
            chaotic_seq = np.array(extended_seq[:total_pixels])

        sorted_indices = np.argsort(chaotic_seq[:total_pixels])

        Q = np.zeros((total_pixels, total_pixels), dtype=int)
        for i, j in enumerate(sorted_indices):
            Q[i, j] = 1

        flat_bit_plane = bit_plane.flatten()
        scrambled_flat = self.semi_tensor_product(Q, flat_bit_plane.reshape(-1, 1))
        scrambled_bit_plane = scrambled_flat.reshape((height, width))

        return scrambled_bit_plane

    def chain_xor_diffusion(self, scrambled_bit_planes, chaotic_seq):
        num_planes = len(scrambled_bit_planes)
        height, width = scrambled_bit_planes[0].shape
        total_pixels = height * width

        V_list = []
        for i in range(num_planes):
            v_k = scrambled_bit_planes[i].flatten()
            V_list.append(v_k)

        V = np.array(V_list)

        if len(chaotic_seq) < total_pixels * num_planes:
            extended_seq = []
            while len(extended_seq) < total_pixels * num_planes:
                extended_seq.extend(chaotic_seq)
            chaotic_seq = np.array(extended_seq[:total_pixels * num_planes])

        K_int = (np.abs(chaotic_seq[:total_pixels * num_planes] * 256) % 256).astype(int)
        K_binary = []
        for num in K_int:
            binary_str = format(num, '')
            for bit in binary_str:
                K_binary.append(int(bit))

        K = K.reshape((num_planes, total_pixels))

        C = np.zeros((num_planes, total_pixels), dtype=int)
        C[:, 0] = V[:, 0] ^ K[:, 0]

        for t in range(1, total_pixels):
            C[:, t] = V[:, t] ^ K[:, t] ^ C[:, t - 1]

        C_final = np.zeros((num_planes, total_pixels), dtype=int)
        C_final[:, -1] = C[:, -1] ^ K[:, -1]

        for t in range(total_pixels - 2, -1, -1):
            C_final[:, t] = C[:, t] ^ K[:, t] ^ C_final[:, t + 1]

        diffused_bit_planes = []
        for i in range(num_planes):
            diffused_plane = C_final[i, :].reshape((height, width))
            diffused_bit_planes.append(diffused_plane)

        return diffused_bit_planes

    def encrypt_image(self, image, seed=42):
        np.random.seed(seed)

        bit_planes = self.bit_plane_decomposition(image)

        chaotic_seq_S2 = generate_3d_lsmcm(10000, self.alpha, self.beta, seed)
        chaotic_seq_S3 = generate_3d_lsmcm(10000, self.alpha, self.beta, seed + 1)

        gol_scrambled_planes = []
        for i, bit_plane in enumerate(bit_planes):
            scrambled = self.game_of_life_scrambling(bit_plane)
            gol_scrambled_planes.append(scrambled)

        global_scrambled_planes = []
        for i, bit_plane in enumerate(gol_scrambled_planes):
            scrambled = self.chaotic_global_scrambling(bit_plane, chaotic_seq_S2)
            global_scrambled_planes.append(scrambled)

        diffused_planes = self.chain_xor_diffusion(global_scrambled_planes, chaotic_seq_S3)

        encrypted_image = self.bit_plane_composition(diffused_planes)

        return encrypted_image

    def decrypt_image(self, encrypted_image, seed=42):
        np.random.seed(seed)

        encrypted_bit_planes = self.bit_plane_decomposition(encrypted_image)

        chaotic_seq_S2 = generate_3d_lsmcm(10000, self.alpha, self.beta, seed)
        chaotic_seq_S3 = generate_3d_lsmcm(10000, self.alpha, self.beta, seed + 1)

        height, width = encrypted_bit_planes[0].shape
        total_pixels = height * width
        num_planes = len(encrypted_bit_planes)

        C_final_list = []
        for i in range(num_planes):
            C_final_list.append(c_final)

        C_final = np.array(C_final_list)

        if len(chaotic_seq_S3) < total_pixels * num_planes:
            extended_seq = []
            while len(extended_seq) < total_pixels * num_planes:
                extended_seq.extend(chaotic_seq_S3)
            chaotic_seq_S3 = np.array(extended_seq[:total_pixels * num_planes])

        K_int = (np.abs(chaotic_seq_S3[:total_pixels * num_planes] * 256) % 256).astype(int)
        K_binary = []
        for num in K_int:
            binary_str = format(num, '08b')
            for bit in binary_str:
                K_binary.append(int(bit))

        K = np.array(K_binary[:total_pixels * num_planes])
        K = K.reshape((num_planes, total_pixels))

        C = np.zeros((num_planes, total_pixels), dtype=int)

        for t in range(total_pixels - 2, -1, -1):
            C[:, t] = C_final[:, t] ^ K[:, t] ^ C_final[:, t + 1]
        C[:, -1] = C_final[:, -1] ^ K[:, -1]

        V = np.zeros((num_planes, total_pixels), dtype=int)
        V[:, 0] = C[:, 0] ^ K[:, 0]

        for t in range(1, total_pixels):
            V[:, t] = C[:, t] ^ K[:, t] ^ C[:, t - 1]

        de_diffused_planes = []
        for i in range(num_planes):
            de_diffused_plane = V[i, :].reshape((height, width))
            de_diffused_planes.append(de_diffused_plane)

        de_global_planes = []
        for i, bit_plane in enumerate(de_diffused_planes):
            height, width = bit_plane.shape
            total_pixels = height * width

            if len(chaotic_seq_S2) < total_pixels:
                extended_seq = []
                while len(extended_seq) < total_pixels:
                    extended_seq.extend(chaotic_seq_S2)
                chaotic_seq_S2 = np.array(extended_seq[:total_pixels])

            sorted_indices = np.argsort(chaotic_seq_S2[:total_pixels])
            inverse_indices = np.argsort(sorted_indices)

            Q_inv = np.zeros((total_pixels, total_pixels), dtype=int)
            for i, j in enumerate(inverse_indices):
                Q_inv[i, j] = 1

            flat_bit_plane = bit_plane.flatten()
            de_scrambled_flat = self.semi_tensor_product(Q_inv, flat_bit_plane.reshape(-1, 1))
            de_global_plane = de_scrambled_flat.reshape((height, width))
            de_global_planes.append(de_global_plane)

        decrypted_planes = []
        for i, bit_plane in enumerate(de_global_planes):
            decrypted_plane = self.game_of_life_scrambling(bit_plane)
            decrypted_planes.append(decrypted_plane)

        decrypted_image = self.bit_plane_composition(decrypted_planes)

        return decrypted_image


def encrypt_process(input_image_path, output_image_path, key=42):
    encryptor = STPImageEncryption()
    image = cv2.imread(input_image_path, 0)
    encrypted = encryptor.encrypt_image(image, key)
    cv2.imwrite(output_image_path, encrypted)
    return encrypted


def decrypt_process(encrypted_image_path, output_image_path, key=42):
    encryptor = STPImageEncryption()
    encrypted_image = cv2.imread(encrypted_image_path, 0)
    decrypted = encryptor.decrypt_image(encrypted_image, key)
    cv2.imwrite(output_image_path, decrypted)
    return decrypted