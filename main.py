import numpy as np
import cv2
import pickle
import os
import time
import argparse
from STPICS1 import STPICSLosslessCompressor, calculate_metrics
from STPImageEncryption import STPImageEncryption


class CompleteSystem:
    def __init__(self, crypto_key=42, compress_block=8):
        self.crypto_key = crypto_key
        self.compress_block = compress_block
        self.encryptor = STPImageEncryption()
        self.compressor = STPICSLosslessCompressor(block_size=compress_block)

    def compress_and_encrypt(self, input_path, output_crypto_path, output_param_path):
        print("=== Starting Compression + Encryption ===")

        original = cv2.imread(input_path, 0)
        if original is None:
            raise ValueError(f"Cannot load image from {input_path}")

        print(f"Original image shape: {original.shape}")

        compress_start = time.time()
        compressed_data, compress_params, cr_value, bpp_value = self.compressor.compress(original)
        compress_time = time.time() - compress_start

        print(f"Compression completed: CR={cr_value:.4f}, BPP={bpp_value:.3f}")

        compressed_array = np.frombuffer(compressed_data, dtype=np.uint8)
        compressed_rows = int(np.sqrt(len(compressed_array))) + 1
        compressed_cols = (len(compressed_array) + compressed_rows - 1) // compressed_rows

        compressed_image = np.zeros(compressed_rows * compressed_cols, dtype=np.uint8)
        compressed_image[:len(compressed_array)] = compressed_array
        compressed_image = compressed_image.reshape(compressed_rows, compressed_cols)

        crypto_start = time.time()
        encrypted_image = self.encryptor.encrypt_image(compressed_image, self.crypto_key)
        crypto_time = time.time() - crypto_start

        cv2.imwrite(output_crypto_path, encrypted_image)

        params_data = {
            'compress_params': compress_params,
            'compressed_shape': (compressed_rows, compressed_cols),
            'original_length': len(compressed_array),
            'cr': cr_value,
            'bpp': bpp_value,
            'compress_time': compress_time,
            'crypto_time': crypto_time
        }

        with open(output_param_path, 'wb') as f:
            pickle.dump(params_data, f)

        total_time = compress_time + crypto_time
        print(f"Total time: {total_time:.2f}s")
        print(f"Encrypted image saved: {output_crypto_path}")
        print(f"Parameters saved: {output_param_path}")

        return encrypted_image, params_data

    def decrypt_and_decompress(self, crypto_path, param_path, output_path):
        print("=== Starting Decryption + Decompression ===")

        encrypted = cv2.imread(crypto_path, 0)
        if encrypted is None:
            raise ValueError(f"Cannot load encrypted image from {crypto_path}")

        with open(param_path, 'rb') as f:
            params = pickle.load(f)

        decrypt_start = time.time()
        decrypted_image = self.encryptor.decrypt_image(encrypted, self.crypto_key)
        decrypt_time = time.time() - decrypt_start

        compressed_rows, compressed_cols = params['compressed_shape']
        original_length = params['original_length']

        compressed_array = decrypted_image.reshape(-1)[:original_length]
        compressed_data = compressed_array.tobytes()

        decompress_start = time.time()
        reconstructed = self.compressor.decompress(compressed_data, params['compress_params'])
        decompress_time = time.time() - decompress_start

        cv2.imwrite(output_path, reconstructed)

        total_time = decrypt_time + decompress_time
        print(f"Total time: {total_time:.2f}s")
        print(f"Decompressed image saved: {output_path}")

        return reconstructed, params

    def evaluate_results(self, original, reconstructed, params):
        mse_val, psnr_val, ssim_val, lossless = calculate_metrics(original, reconstructed)

        print("\n=== Performance Evaluation ===")
        print(f"Compression Ratio: {params['cr']:.4f}")
        print(f"Bits Per Pixel: {params['bpp']:.3f}")
        print(f"MSE: {mse_val:.6f}")
        print(f"PSNR: {psnr_val:.2f} dB")
        print(f"SSIM: {ssim_val:.6f}")
        print(f"Lossless: {lossless}")
        print(f"Compression Time: {params['compress_time']:.2f}s")
        print(f"Encryption Time: {params['crypto_time']:.2f}s")

        return {
            'cr': params['cr'],
            'bpp': params['bpp'],
            'mse': mse_val,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'lossless': lossless
        }


def main():
    parser = argparse.ArgumentParser(description='Complete System')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--mode', type=str, choices=['encrypt', 'decrypt', 'both'], default='both',
                        help='Operation mode')
    parser.add_argument('--key', type=int, default=42, help='Encryption key')
    parser.add_argument('--block_size', type=int, default=8, help='Compression block size')

    args = parser.parse_args()

    system = CompleteSystem(
        crypto_key=args.key,
        compress_block=args.block_size
    )

    base_name = os.path.splitext(os.path.basename(args.input))[0]
    crypto_path = f"{base_name}_encrypted.png"
    param_path = f"{base_name}_params.pkl"
    output_path = f"{base_name}_decrypted.png"

    if args.mode in ['encrypt', 'both']:
        print(f"Processing: {args.input}")
        encrypted_img, params = system.compress_and_encrypt(
            args.input, crypto_path, param_path
        )

        if args.mode == 'both':
            original_image = cv2.imread(args.input, 0)
            reconstructed_img, _ = system.decrypt_and_decompress(
                crypto_path, param_path, output_path
            )

            system.evaluate_results(original_image, reconstructed_img, params)

    elif args.mode == 'decrypt':
        if not os.path.exists(crypto_path) or not os.path.exists(param_path):
            print("Error: Required files not found")
            return

        reconstructed_img, params = system.decrypt_and_decompress(
            crypto_path, param_path, output_path
        )
        print("Decryption completed")


if __name__ == "__main__":
    main()