
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import heapq


class Node:

    def __init__(self, pixel, frequency):
        self.pixel = pixel
        self.frequency = frequency
        self.left = None
        self.right = None


def build_huffman_tree(frequency):

    heap = []
    counter = 0

    for pixel, count in frequency.items():

        node = Node(pixel, count)

        heapq.heappush(heap, (count, counter, node))
        counter += 1

    while len(heap) > 1:

        freq1, _, left = heapq.heappop(heap)
        freq2, _, right = heapq.heappop(heap)

        merged = Node(None, freq1 + freq2)

        merged.left = left
        merged.right = right

        heapq.heappush(heap, (merged.frequency, counter, merged))
        counter += 1

    return heap[0][2]


def generate_code(node, code, codes):

    if node.pixel is not None:

        codes[node.pixel] = code if code else "0"
        return

    generate_code(node.left, code + "0", codes)
    generate_code(node.right, code + "1", codes)


def huffman_encoding(image):

    pixels = image.flatten()

    frequency = Counter(pixels)

    root = build_huffman_tree(frequency)

    codes = {}
    generate_code(root, "", codes)

    encoded = ""

    for pixel in pixels:
        encoded += codes[pixel]

    return encoded, root, codes


def huffman_decoding(encoded, root):

    decoded = []
    node = root

    if root.pixel is not None:
        return np.full(len(encoded), root.pixel, dtype=np.uint8)

    for bit in encoded:

        if bit == "0":
            node = node.left
        else:
            node = node.right

        if node.pixel is not None:

            decoded.append(node.pixel)
            node = root

    return np.array(decoded, dtype=np.uint8)


def main():

    image = cv2.imread("input/image.png")

    if image is None:
        print("Error: Could not load image.")
        return

    encoded, root, codes = huffman_encoding(image)

    print("Number of unique pixel values:", len(codes))
    print("Original size in bits:", image.size * 8)
    print("Encoded size in bits:", len(encoded))


    decoded = huffman_decoding(encoded, root)


    reconstructed = decoded.reshape(image.shape)


    original_bits = image.size * 8
    encoded_bits = len(encoded)

    compression_ratio = original_bits / encoded_bits

    print("Compression ratio:", compression_ratio)

    cv2.imwrite("output/huffman_reconstructed.png", reconstructed)
    plt.imshow(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
