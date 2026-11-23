
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from fractions import Fraction


def calculate_frequency(image):

    pixels = image.flatten()
    frequency = Counter(pixels)

    return pixels, frequency


def calculate_cumulative(frequency):

    total = sum(frequency.values())

    cumulative = {}

    low = Fraction(0, 1)

    for pixel in sorted(frequency):

        probability = Fraction(frequency[pixel], total)

        high = low + probability

        cumulative[pixel] = (low, high)

        low = high

    return cumulative


def arithmetic_encoding(pixels, cumulative):

    low = Fraction(0, 1)
    high = Fraction(1, 1)

    for pixel in pixels:

        range_value = high - low

        symbol_low, symbol_high = cumulative[pixel]

        high = low + range_value * symbol_high
        low = low + range_value * symbol_low

    encoded_value = (low + high) / 2

    return encoded_value


def arithmetic_decoding(encoded_value, cumulative, length):

    decoded = []

    value = encoded_value

    for i in range(length):

        found = False

        for pixel, (low, high) in cumulative.items():

            if low <= value < high:

                decoded.append(pixel)

                value = (value - low) / (high - low)

                found = True
                break

        if not found:
            print("Decoding failed at pixel:", i)
            break

    return decoded


def main():

    image = cv2.imread("input/image1.png", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_NEAREST)
    print("Image loaded:", image.shape, flush=True)

    pixels, freq = calculate_frequency(image)
    print("Frequency calculated:", freq, flush=True)

    cumulative = calculate_cumulative(freq)
    print("Cumulative intervals calculated", flush=True)

    encoded = arithmetic_encoding(pixels, cumulative)
    print("Encoding completed", flush=True)

    decoded = arithmetic_decoding(encoded, cumulative, len(pixels))
    print("Decoding completed:", len(decoded), flush=True)

    if image is None:
        print("Image not found!")
        return

    pixels, freq = calculate_frequency(image)

    cumulative = calculate_cumulative(freq)

    encoded = arithmetic_encoding(pixels, cumulative)

    print("Encoded value:", encoded)

    decoded = arithmetic_decoding(encoded, cumulative, len(pixels))

    print("Original pixels:", len(pixels))
    print("Decoded pixels:", len(decoded))

    if len(decoded) != len(pixels):
        print("Reconstruction failed!")
        return

    reconstructed = np.array(decoded, dtype=np.uint8)

    reconstructed = reconstructed.reshape(image.shape)
    enlarged = cv2.resize(
    reconstructed,
    (320, 320),
    interpolation=cv2.INTER_NEAREST
)


    cv2.imwrite("output/arithmetic_reconstructed.jpg",enlarged)

    if np.array_equal(image, reconstructed):
        print("Reconstruction successful!")
    else:
        print("Reconstruction does not match!")

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed, cmap="gray", vmin=0, vmax=255)
    plt.title("Reconstructed Image")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
