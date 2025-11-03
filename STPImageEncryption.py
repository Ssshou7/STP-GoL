import numpy as np
import cv2
from STPLifeGameScramblingLifeGameScrambling import LifeGameScrambling
from STPGlobalSGlobalScrambling import GlobalScrambling
from ChainXorDDiffusion import Diffusion
from"3D-LSMCM" import generate_3d_lsmcm


class STPImageEncryption:
    def __init__(self, block_dim=(8, 8)):
        self.block_dim = block_dim
        self.life_game = LifeGameScrambling(block_dim)
        self.global_scrambling = GlobalScrambling()
        self.diffusion = Diffusion()

    def split_bit_planes(self, image_data):
        if len(image_data.shape) == 3:
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        rows, cols = image_data.shape
        bit_planes = []
        for plane in range(8):
            plane_data = ((image_data >> plane) & 1).astype(np.uint8)
            bit_planes.append(plane_data)
        return bit_planes

    def combine_bit_planes(self, bit_planes):
        rows, cols = bit_planes[0].shape
        combined = np.zeros((rows, cols), dtype=np.uint8)
        for idx, plane in enumerate(bit_planes):
            combined |= (plane.astype(np.uint8) << idx)
        return combined

    def encrypt_image(self, image_data, key_value=42):
        np.random.seed(key_value)

        bit_planes = self.split_bit_planes(image_data)

        sequence_1 = generate_3d_lsmcm(10000, 0.92, 0.81, key_value)
        sequence_2 = generate_3d_lsmcm(10000, 0.92, 0.81, key_value + 1)

        life_game_result = []
        for plane in bit_planes:
            processed = self.life_game.process_block(plane)
            life_game_result.append(processed)

        global_scramble_result = []
        for plane in life_game_result:
            processed = self.global_scrambling.apply_scrambling(plane, sequence_1)
            global_scramble_result.append(processed)

        diffusion_result = self.diffusion.apply_diffusion(global_scramble_result, sequence_2)

        encrypted_image = self.combine_bit_planes(diffusion_result)

        return encrypted_image

    def decrypt_image(self, encrypted_data, key_value=42):
        np.random.seed(key_value)

        encrypted_planes = self.split_bit_planes(encrypted_data)

        sequence_1 = generate_3d_lsmcm(10000, 0.92, 0.81, key_value)
        sequence_2 = generate_3d_lsmcm(10000, 0.92, 0.81, key_value + 1)

        rows, cols = encrypted_planes[0].shape
        total_pixels = rows * cols
        plane_count = len(encrypted_planes)

        final_vectors = []
        for i in range(plane_count):
            vector = encrypted_planes[i].flatten()
            final_vectors.append(vector)

        final_matrix = np.array(final_vectors)

        if len(sequence_2) < total_pixels * plane_count:
            extended = []
            while len(extended) < total_pixels * plane_count:
                extended.extend(sequence_2)
            sequence_2 = np.array(extended[:total_pixels * plane_count])

        integer_keys = (np.abs(sequence_2[:total_pixels * plane_count] * 256) % 256).astype(int)
        binary_keys = []
        for num in integer_keys:
            binary_rep = format(num, '08b')
            for bit in binary_rep:
                binary_keys.append(int(bit))

        key_matrix = np.array(binary_keys[:total_pixels * plane_count])
        key_matrix = key_matrix.reshape((plane_count, total_pixels))

        intermediate = np.zeros((plane_count, total_pixels), dtype=int)

        for pos in range(total_pixels - 2, -1, -1):
            intermediate[:, pos] = final_matrix[:, pos] ^ key_matrix[:, pos] ^ final_matrix[:, pos + 1]
        intermediate[:, -1] = final_matrix[:, -1] ^ key_matrix[:, -1]

        original_vectors = np.zeros((plane_count, total_pixels), dtype=int)
        original_vectors[:, 0] = intermediate[:, 0] ^ key_matrix[:, 0]

        for pos in range(1, total_pixels):
            original_vectors[:, pos] = intermediate[:, pos] ^ key_matrix[:, pos] ^ intermediate[:, pos - 1]

        de_diffused = []
        for i in range(plane_count):
            plane_data = original_vectors[i, :].reshape((rows, cols))
            de_diffused.append(plane_data)

        de_global = []
        for plane in de_diffused:
            rows, cols = plane.shape
            total_pixels = rows * cols

            if len(sequence_1) < total_pixels:
                extended = []
                while len(extended) < total_pixels:
                    extended.extend(sequence_1)
                sequence_1 = np.array(extended[:total_pixels])

            sorted_indices = np.argsort(sequence_1[:total_pixels])
            inverse_indices = np.argsort(sorted_indices)

            inverse_matrix = np.zeros((total_pixels, total_pixels), dtype=int)
            for new_pos, old_pos in enumerate(inverse_indices):
                inverse_matrix[new_pos, old_pos] = 1

            flat_plane = plane.flatten()
            de_scrambled_flat = self.global_scrambling.stp_multiply(inverse_matrix, flat_plane.reshape(-1, 1))
            de_global_plane = de_scrambled_flat.reshape((rows, cols))
            de_global.append(de_global_plane)

        decrypted_planes = []
        for plane in de_global:
            decrypted_plane = self.life_game.process_block(plane)
            decrypted_planes.append(decrypted_plane)

        decrypted_image = self.combine_bit_planes(decrypted_planes)

        return decrypted_image