import cv2
import numpy as np
import matplotlib.pyplot as plt


def rgb_to_hsi(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    intensity = (r + g + b) / 3

    minimum = np.minimum(r, np.minimum(g, b))
    saturation = 1 - (3 * minimum) / (r + g + b + 1e-8)

    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g)**2 + (r - b) * (g - b))

    theta = np.arccos(
        np.clip(numerator / (denominator + 1e-8), -1, 1)
    )

    hue = np.where(b <= g, theta, 2 * np.pi - theta)
    hue = hue / (2 * np.pi)
    hue = np.where(saturation == 0, 0, hue)

    hsi = np.stack((hue, saturation, intensity), axis=2)

    return hsi


def hsi_to_rgb(hsi):

    h = hsi[:, :, 0] * 2 * np.pi
    s = hsi[:, :, 1]
    i = hsi[:, :, 2]

    r = np.zeros(h.shape)
    g = np.zeros(h.shape)
    b = np.zeros(h.shape)

    mask = h < 2 * np.pi / 3

    hm = h[mask]

    b[mask] = i[mask] * (1 - s[mask])

    r[mask] = i[mask] * (
        1 + s[mask] * np.cos(hm) /
        (np.cos(np.pi / 3 - hm) + 1e-8)
    )

    g[mask] = 3 * i[mask] - r[mask] - b[mask]

    mask = (h >= 2 * np.pi / 3) & (h < 4 * np.pi / 3)

    hm = h[mask] - 2 * np.pi / 3

    r[mask] = i[mask] * (1 - s[mask])

    g[mask] = i[mask] * (
        1 + s[mask] * np.cos(hm) /
        (np.cos(np.pi / 3 - hm) + 1e-8)
    )

    b[mask] = 3 * i[mask] - r[mask] - g[mask]

    mask = h >= 4 * np.pi / 3

    hm = h[mask] - 4 * np.pi / 3

    g[mask] = i[mask] * (1 - s[mask])

    b[mask] = i[mask] * (
        1 + s[mask] * np.cos(hm) /
        (np.cos(np.pi / 3 - hm) + 1e-8)
    )

    r[mask] = 3 * i[mask] - g[mask] - b[mask]

    rgb = np.stack((r, g, b), axis=2)

    return np.clip(rgb, 0, 1)


if __name__ == "__main__":

    image = cv2.imread("input/image.png")

    if image is None:
        print("Image not found")
        exit()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    hsi = rgb_to_hsi(image)
    reconstructed = hsi_to_rgb(hsi)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original RGB")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed)
    plt.title("Reconstructed RGB")
    plt.axis("off")

    plt.show()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(hsi[:, :, 0], cmap="gray")
    plt.title("Hue")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(hsi[:, :, 1], cmap="gray")
    plt.title("Saturation")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(hsi[:, :, 2], cmap="gray")
    plt.title("Intensity")
    plt.axis("off")

    plt.show()

