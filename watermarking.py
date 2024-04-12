import cv2
import numpy as np

def normalize_image(image, alpha, beta, delta, phi):
    """
    Normalize the input image using affine transformation.

    Parameters:
        image: numpy array, input image
        alpha: float, scaling factor along x-axis
        beta: float, shearing factor
        delta: float, scaling factor along y-axis
        phi: float, rotation angle in radians (0 to pi)

    Returns:
        normalized_image: numpy array, normalized image
    """
    rows, cols = image.shape[:2]

    # Define affine transformation matrix
    M = np.array([[alpha * np.cos(phi), np.sin(phi) + beta * np.cos(phi), 0],
                  [-np.sin(phi) + beta * np.cos(phi), alpha * np.cos(phi), 0]])

    # Apply affine transformation
    normalized_image = cv2.warpAffine(image, M, (cols, rows))

    return normalized_image


def quadtree_decomposition(image, threshold, min_block_size=4, start_x=0, start_y=0, block_size=None, position=0, homogeneous_blocks=[], non_homogeneous_blocks=[]):
    """
    Perform quadtree decomposition on the image.

    
    Parameters:
        image: numpy array, input grayscale image
        threshold: float, intensity difference threshold for splitting
        min_block_size: int, minimum block size
        start_x: int, starting x-coordinate of the block
        start_y: int, starting y-coordinate of the block
        block_size: int, size of the block
        position: int, position of the block
        homogeneous_blocks: list, stores information about homogeneous blocks
        non_homogeneous_blocks: list, stores information about non-homogeneous blocks

    Returns:
        homogeneous_blocks, non_homogeneous_blocks: lists containing information about homogeneous and non-homogeneous blocks
    """
    if block_size is None:
        block_size = min(image.shape[0], image.shape[1])

    if block_size <= min_block_size:
        if max(image[start_y:start_y + block_size, start_x:start_x + block_size].flatten()) - min(image[start_y:start_y + block_size, start_x:start_x + block_size].flatten()) <= threshold:
            homogeneous_blocks.append((start_x, start_y, block_size, position))
        else:
            non_homogeneous_blocks.append((start_x, start_y, block_size, position))
    else:
        sub_block_size = block_size // 2

        sub_images = [
            image[start_y:start_y + sub_block_size, start_x:start_x + sub_block_size],
            image[start_y:start_y + sub_block_size, start_x + sub_block_size:start_x + 2 * sub_block_size],
            image[start_y + sub_block_size:start_y + 2 * sub_block_size, start_x:start_x + sub_block_size],
            image[start_y + sub_block_size:start_y + 2 * sub_block_size, start_x + sub_block_size:start_x + 2 * sub_block_size]
        ]

        max_intensity_diff = max(sub_image.max() - sub_image.min() for sub_image in sub_images)

        if max_intensity_diff > threshold:
            quadtree_decomposition(image, threshold, min_block_size, start_x, start_y, sub_block_size, position * 4 + 1, homogeneous_blocks, non_homogeneous_blocks)
            quadtree_decomposition(image, threshold, min_block_size, start_x + sub_block_size, start_y, sub_block_size, position * 4 + 2, homogeneous_blocks, non_homogeneous_blocks)
            quadtree_decomposition(image, threshold, min_block_size, start_x, start_y + sub_block_size, sub_block_size, position * 4 + 3, homogeneous_blocks, non_homogeneous_blocks)
            quadtree_decomposition(image, threshold, min_block_size, start_x + sub_block_size, start_y + sub_block_size, sub_block_size, position * 4 + 4, homogeneous_blocks, non_homogeneous_blocks)
        else:
            homogeneous_blocks.append((start_x, start_y, block_size, position))

    return homogeneous_blocks, non_homogeneous_blocks

# Load the image
image = cv2.imread('1.jpeg', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded
if image is None:
    print('Could not open or find the image')
else:
    # Define parameters
    threshold = 10  # Intensity difference threshold
    min_block_size = 4  # Minimum block size

    # Perform quadtree decomposition
    homogeneous_blocks, non_homogeneous_blocks = quadtree_decomposition(image, threshold, min_block_size)

    # Create an image view
    image_view = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw homogeneous blocks in green and non-homogeneous blocks in red
    for block in homogeneous_blocks:
        x, y, size, _ = block
        cv2.rectangle(image_view, (x, y), (x + size, y + size), (0, 255, 0), 1)

    for block in non_homogeneous_blocks:
        x, y, size, _ = block
        cv2.rectangle(image_view, (x, y), (x + size, y + size), (0, 0, 255), 1)


    print(len(homogeneous_blocks))
    print(len(non_homogeneous_blocks))

    # Display the image view
    cv2.imshow('Image View', image_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()