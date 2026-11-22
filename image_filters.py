
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.ndimage import (
    minimum_filter,
    median_filter,
    maximum_filter,
    uniform_filter,
    convolve,
    gaussian_filter
)

from HSI_2_RGB import rgb_to_hsi, hsi_to_rgb


def apply_filters(hsi):

    results = {}

    minimum = np.empty(hsi.shape, hsi.dtype)

    for channel in range(3):
        minimum[:, :, channel] = minimum_filter(
            hsi[:, :, channel], size=3, mode="constant"
        )

    results['Minimum'] = minimum

    median = np.empty(hsi.shape, hsi.dtype)

    for channel in range(3):
        median[:, :, channel] = median_filter(
            hsi[:, :, channel], size=3, mode="constant"
        )

    results['Median'] = median

    maximum = np.empty_like(hsi)

    for channel in range(3):
        maximum[:, :, channel] = maximum_filter(
            hsi[:, :, channel], size=3, mode="constant"
        )

    results['Maximum'] = maximum

    average = np.empty_like(hsi)

    for channel in range(3):
        average[:, :, channel] = uniform_filter(
            hsi[:, :, channel], size=3, mode="constant"
        )

    results['Average'] = average

    weighted_kernel = np.array(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]],
        dtype=np.float32
    ) / 16

    weighted = np.empty_like(hsi)

    for channel in range(3):
        weighted[:, :, channel] = convolve(
            hsi[:, :, channel],
            weighted_kernel,
            mode='constant',
            cval=0
        )

    results['Weighted'] = weighted

    gaussian = np.empty_like(hsi)

    for channel in range(3):
        gaussian[:, :, channel] = gaussian_filter(
            hsi[:, :, channel],
            sigma=1,
            mode='constant',
            cval=0
        )

    results['Gaussian'] = gaussian

    intensity = hsi[:, :, 2]

    sobel_x = np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=np.float32
    )

    sobel_y = np.array(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],
        dtype=np.float32
    )

    gx = convolve(intensity, sobel_x, mode='constant')
    gy = convolve(intensity, sobel_y, mode='constant')

    G = np.sqrt(gx**2 + gy**2)

    sobel = G / (G.max() + 1e-8)

    results['Sobel'] = sobel

    laplacian_kernel = np.array(
        [[0, -1, 0],
         [-1, 4, -1],
         [0, -1, 0]],
        dtype=np.float32
    )

    laplacian = convolve(
        intensity,
        laplacian_kernel,
        mode="constant"
    )

    laplacian = np.abs(laplacian)

    laplacian = laplacian / (laplacian.max() + 1e-8)

    results['Laplacian'] = laplacian

    return results


def save_images(results):

    for name, image in results.items():

        if image.ndim == 3:
            image = hsi_to_rgb(image)

        image = np.clip(image * 255, 0, 255).astype(np.uint8)

        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        filename = "output/" + name + ".png"

        cv2.imwrite(filename, image)


if __name__ == "__main__":

    image = cv2.imread("input/image.png")

    if image is None:
        print("Image not found!")
        exit()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    hsi = rgb_to_hsi(image)

    results = apply_filters(hsi)

    save_images(results)

    print("All filtered images saved successfully!")

    plt.figure(figsize=(16, 13))

    plt.subplot(3, 3, 1)
    plt.imshow(image)
    plt.title("Original", pad=12)
    plt.axis("off")

    index = 2

    for name in results:

        filtered_image = results[name]

        plt.subplot(3, 3, index)

        if filtered_image.ndim == 3:
            filtered_image = hsi_to_rgb(filtered_image)
            plt.imshow(filtered_image)
        else:
            plt.imshow(filtered_image, cmap="gray")

        plt.title(name, pad=12)
        plt.axis("off")

        index += 1

    plt.subplots_adjust(
        hspace=0.35,
        wspace=0.12,
        top=0.95,
        bottom=0.04
    )

    plt.savefig("output/Comparison.png", dpi=300, bbox_inches="tight")
    plt.show()