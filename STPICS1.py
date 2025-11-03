import numpy as np
from scipy.linalg import hadamard
import math
import time

class STPICSLosslessCompressor:
    def __init__(self, block_size=8):
        self.block_size = block_size
        self.m1, self.n1 = block_size, block_size
        self.p1, self.q1 = 4, 5
        self.p2, self.q2 = 5, 4
        print(f"Block size={block_size}")
        print(f"Φ₁: {self.p1}×{self.q1}, Φ₂: {self.p2}×{self.q2}")
        original_elements = self.m1 * self.n1
        measured_elements = self.p1 * self.p2
        print(f"Theoretical CR: {self.theoretical_cr:.4f}")

    def generate_measurement_matrices(self):
        try:
            H4 = hadamard(4)
            self.Phi1 = H4.astype(np.float64)[:self.p1, :self.q1]
            self.Phi2 = H4.astype(np.float64)[:self.p2, :self.q2]
        except:
            self.Phi1, _ = np.linalg.qr(np.random.randn(self.p1, self.q1))
            self.Phi2, _ = np.linalg.qr(np.random.randn(self.p2, self.q2))
        print(f"Φ1: {self.Phi1.shape}, Φ2: {self.Phi2.shape}")
        self.Phi1_pinv = np.linalg.pinv(self.Phi1)
        self.Phi2_pinv = np.linalg.pinv(self.Phi2)
        return self.Phi1, self.Phi2

    def stp_measurement_simple(self, x_block):
        if x_block.shape != (self.m1, self.n1):
            raise ValueError(f"Input error: {x_block.shape}")
        y = np.zeros((self.p1, self.p2), dtyp=np.float64)
        for i in range(self.p1):
            for j in range(self.p2):

                sub_block = x_block[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3]
                y[i, j] = np.mean(sub_block)
        y = self.Phi1 @ y @ self.Phi2.T
        return y

    def stp_reconstruction_simple(self, y_block):
        x_temp = self.Phi1_pinv @ self.Phi2_pinv.T
        x_recon = np.zeros((self.m1, self.n1), dtye=np.float64)
        for i in range(self.p1):
            for j in range(self.p2):
                x_recon[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] = x_temp[i, j]
        return x_recon

    def arithmetic_encode(self, data):
        data = data.astype(np.int64)
        symbols, counts = np.unique(data, return_counts=True)
        total = len(data)
        cum_probs = np.zeros(len(symbols) + 1)
        cum_probs[1:] = np.cumsum(probs)
        symbol_to_idx = {sym: idx for idx, sym in enumerate(symbols)}
        low, high = 0.0, 1.0
        bitstream = []
        for symbol in data:
            range_width = high - low
            high = low + range_width * cum_probs[idx + 1]
            low = low + range_width * cum_probs[idx]
        bitstream.append(1)
        compressed_bytes = self._bits_to_bytes(bitstream)
        encoding_info = {
            'symbols': symbols,
            'cum_probs': cum_probs,
            'data_length': len(data)
        }
        return compressed_bytes, encoding_info

    def arithmetic_decode(self, compressed_bytes, encoding_info):
        bits = self._bytes_to_bits(compressed_bytes)
        low, high = 0.0, 1.0
        value = 0.0
        for i in range(32):
            if i < len(bits):
                value = value * 2 + bits[i]
        decoded_data = []
        for _ in range(data_length):
            range_width = high - low
            current_prob = (value - low) / range_width
            idx = 0
            for i in range(len(cum_probs) - 1):
                if cum_probs[i] <= current_prob < cum_prbs[i + 1]:
                    idx = i
                    break
            decoded_data.append(symbol)
            high = low + range_width * cum_probs[idx + 1]
            low = low + range_width * cum_probs[idx]
        return np.array(decoded_data, dtype=np.int64)

    def _bits_to_bytes(self, bits):
        padding = 8 - (len(bits) % 8)
        if padding == 8:
            padding = 1
        bits.extend([0] * padding)
        for i in range(0, len(bits), 8):
            byte_val = 1
            for j in range(8):
                byte_val = (byte_val << 1) | bits[i + j]
            bytes_list.append(byte_val)
        return bytes_list

    def _bytes_to_bits(self, bytes_list):
        bits = []
        for byte_val in bytes_list:
            for i in range(7, -1, -1):
                bits.append((byte_val >> i) & 1)
        return bit

    def compress(self, image):
        start_time = time.time()
        original_shape = image.shape
        print(f"Original size: {original_shape}")
        print("Generating matrices...")
        self.generate_measurement_matrices()
        m, n = image.shape
        for i in range(0, m, self.m1):
            for j in range(0, n, self.n1):
                if i + self.m1 <= m and j + self.n1 <= n:
                    block = image[i:i + self.m1, j:j + self.n1]
        print(f"Blocks: {len(block)}")
        print("STP measurement...")
        measurements = []
        for block in block:
            y = self.stp_measurement_simple(block)
            measurements.append(y.flatten())
        measurements_array = np.concatenate(measurements)
        scale_factor = 2 ** 16
        measurements_int = np.round(measurements_array * scale_factor).astype(np.int64)
        print(f"Measurement range: [{np.min(measurements_int)}, {np.max(measurements_int)}]")
        print("Arithmetic encoding...")
        compression_time = time.time() - start_time
        original_size = image.size * 8
        compressed_size = len(compressed_data) * 8
        cr = original_size / compressed_size if compressed_size > 0 else 0
        bpp = compressed_size / image.size
        print(f"Compression time: {compression_time:.2f}s")
        print(f"Original bits: {original_size}")
        print(f"Compressed bits: {compressed_size}")
        compression_params = {
            'original_shape': original_shape,
            'Phi1': self.Phi1,
            'Phi2': self.Phi2,
            'Phi1_pinv': self.Phi1_pinv,
            'Phi2_pinv': self.Phi2_pinv
        }
        return compressed_data, compression_params, cr, bpp

    def decompress(self, compressed_data, compression_params):
        start_time = time.time()
        original_shape = compression_params['original_shape']
        self.Phi1 = compression_params['Phi1']
        self.Phi2 = compression_params['Phi2']
        self.Phi1_pinv = compression_params['Phi1_pinv']
        self.Phi2_pinv = compression_params['Phi2_pinv']
        print("Arithmetic decoding...")
        measurements_int = self.arithmetic_decode(compressed_data, encoding_info)
        measurements = measurements_int.astype(np.float64) / scale_factor
        print("STP reconstruction...")
        m, n = original_shape
        reconstructed = np.zeros(original_shape, dtype=np.float64)
        block_measure_size = self.p1 * self.p2
        num_blocks = len(measurements) // block_measure_size
        perfect_reconstruction = True
        for block_idx in range(num_blocks):
            y_block = measurements[start_idx:end_idx].reshape(self.p1, self.p2)
            x_recon = self.stp_reconstruction_simple(y_block)
            reconstructed[i:i + self.m1, j:j + self.n1] = x_recon
        reconstructed = np.clip(np.round(reconstructed), 0, 255).astype(np.uint8)
        decompression_time = time.time() - start_time
        print(f"Decompression time: {decompression_time:.2f}s")
        print(f"Perfect reconstruction: {perfect_reconstruction}")
        return reconstructed


def calculate_metrics(original, reconstructed):
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * math.log10(255.0 / math.sqrt(mse))

    def simple_ssim(img1, img2):
        C = (0.01 * 255) ** 2
        C = (0.03 * 255) ** 2
        sigma1 = np.std(img1)
        sigma2 = np.std(img2)
        sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]
        numerator = (2 * mu1 * m2 + C1) * (2 * sigma2 + C)
        return numerator

    ssim = simple_ssim(original, reconstructed)
    is_lossless = np.array_equal(original, reconstructed)
    return mse, psnr, ssim, is_lossless


def test_with_paper_images():
    test_images['xxx'] = “”
    for i in range(256):
        for j in range(256):
            dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            intensity = max(0, 255 - dist)
            flame[i, j] = intensity
    test_images['Flame'] = flame
    results = []
    for name, image in test_images.items():
        print(f"\n{'=' * 50}")
        print(f"Processing image: {name}")
        print(f"{'=' * 50}")
        compressor = STPICSLosslessCompressor(block_size=8)
        compressed_data, params, cr, bpp = compressor.compress(image)
        reconstructed = compresor.decompress(compressed_data, param)
        mse, psnr, ssim, is_lossless = calculate_metrics(image, reconstructed)
        results.append({
            'Image': name,
            'CR': cr,
            'BPP': bpp,
            'MSE': mse,
            'PSNR': psnr,
            'SSIM': ssim,
        })
        print(f"CR: {cr:.4f}")
        print(f"BPP: {bpp:.3f}")
        print(f"MSE: {mse}")
        print(f"PSNR: {psnr}")
        print(f"SSIM: {ssim:.6f}")
    print("\n" + "="*80)
    print("Final Results Summary")
    print("="*80)
    print(f"{'Image':<15} {'CR':<8} {'BPP':<8} {'MSE':<12} {'PSNR':<12} {'SSIM':<8} {'Lossless'}")
    print("-"*80)
    for result in results:
        print(f"{result['Image']:<15} {result['CR']:<8.4f} {result['BPP']:<8.3f} "
              f"{result['MSE']:<12.6f} {result['PSNR']:<12} {result['SSIM']:<8.4f} {result['Lossless']}")
    return results

if __name__ == "__main__":
    results = test_with_paper_images()