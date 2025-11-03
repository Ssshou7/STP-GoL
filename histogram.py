import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def plot_comparative_histogram(original_image, encrypted_image, save_path=None):

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')


    axes[0, 1].imshow(encrypted_image, cmap='gray')
    axes[0, 1].set_title('Encrypted Image')
    axes[0, 1].axis('off')

    axes[1, 0].hist(original_image.flatten(), bins=256, range=[0, 256],
                    alpha=0.7, color='blue', density=True)
    axes[1, 0].set_title('Original Image Histogram')
    axes[1, 0].set_xlabel('Pixel Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)

    orig_mean = np.mean(original_image)
    orig_std = np.std(original_image)
    orig_var = np.var(original_image)
    axes[1, 0].text(0.02, 0.98, f'Mean: {orig_mean:.2f}\nStd: {orig_std:.2f}\nVar: {orig_var:.2f}',
                    transform=axes[1, 0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[1, 1].hist(encrypted_image.flatten(), bins=256, range=[0, 256],
                    alpha=0.7, color='red', density=True)
    axes[1, 1].set_title('Encrypted Image Histogram')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)

    enc_mean = np.mean(encrypted_image)
    enc_std = np.std(encrypted_image)
    enc_var = np.var(encrypted_image)
    axes[1, 1].text(0.02, 0.98, f'Mean: {enc_mean:.2f}\nStd: {enc_std:.2f}\nVar: {enc_var:.2f}',
                    transform=axes[1, 1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    orig_hist, _ = np.histogram(original_image.flatten(), bins=256, range=[0, 256], density=True)
    enc_hist, _ = np.histogram(encrypted_image.flatten(), bins=256, range=[0, 256], density=True)

    orig_hist_var = np.var(orig_hist)
    enc_hist_var = np.var(enc_hist)

    fig.suptitle(
        f'Histogram Analysis: Original Variance = {orig_hist_var:.6f}, Encrypted Variance = {enc_hist_var:.6f}',
        fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比直方图已保存至: {save_path}")

    plt.show()

    return orig_hist_var, enc_hist_var


def calculate_chi_square_test(encrypted_image, bins=256):

    hist, _ = np.histogram(encrypted_image.flatten(), bins=bins, range=[0, 256])

    expected_frequency = len(encrypted_image.flatten()) / bins

    chi_square = np.sum((hist - expected_frequency) ** 2 / expected_frequency)

    return chi_square



if __name__ == "__main__":

    original_img = np.array(Image.open('House.jpg').convert('L'))


    encrypted_img = encryption_algorithm(original_img)


    orig_var, enc_var = plot_comparative_histogram(original_img, encrypted_img,
                                                   save_path='histogram_comparison.png')


    chi_square = calculate_chi_square_test(encrypted_img)
    print(f"Variance of the original image histogram: {orig_var:.6f}")
    print(f"Encrypted Image Histogram Variance: {enc_var:.6f}")
    print(f"Chi-Square Test Value for Encrypted Images: {chi_square:.2f}")
