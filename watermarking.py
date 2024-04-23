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

def categorize_blocks(image , threshold=10):
    """
    Categorize image blocks into 'smooth' or 'texture' categories.

    Parameters:
        image: numpy array, input grayscale image
        threshold: float, intensity difference threshold for splitting

    Returns:
        smooth_blocks: list, contains positions of smooth blocks
        texture_blocks: list, contains positions of texture blocks
    """
    smooth_blocks = []
    texture_blocks = []
    block_id = 0  # Initial block ID

    block_size = min(image.shape[0], image.shape[1])

    # Iterate through the image and categorize each 4x4 block
    for y in range(0, block_size, 4):
        for x in range(0, block_size, 4):
            # Check if the 4x4 block is homogeneous or not
            block_homogeneous = False
            if max(image[y:y + 4, x:x + 4].flatten()) - min(image[y:y + 4, x:x + 4].flatten()) <= threshold:
                block_homogeneous = True

            if block_homogeneous:
                smooth_blocks.append((x, y, block_id))
            else:
                texture_blocks.append((x, y, block_id))

            # Increment the block ID for the next block
            block_id += 1

    return smooth_blocks, texture_blocks



position = 0

def quadtree_decomposition(image, threshold, min_block_size=4, start_x=0, start_y=0, block_size=None, homogeneous_blocks=[], non_homogeneous_blocks=[]):
    global position
    """
    Perform quadtree decomposition on the image.

    
    Parameters:
        image: numpy array, input grayscale image
        threshold: float, intensity difference threshold for splitting
        min_block_size: int, minimum block size
        start_x: int, starting x-coordinate of the block
        start_y: int, starting y-coordinate of the block
        block_size: int, size of the block
        homogeneous_blocks: list, stores information about homogeneous blocks
        non_homogeneous_blocks: list, stores information about non-homogeneous blocks

    Returns:
        homogeneous_blocks, non_homogeneous_blocks: lists containing information about homogeneous and non-homogeneous blocks
    """
    if block_size is None:
        block_size = min(image.shape[0], image.shape[1])

    if block_size <= min_block_size:
        position+=1
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
            quadtree_decomposition(image, threshold, min_block_size, start_x, start_y, sub_block_size, homogeneous_blocks, non_homogeneous_blocks)
            quadtree_decomposition(image, threshold, min_block_size, start_x + sub_block_size, start_y, sub_block_size, homogeneous_blocks, non_homogeneous_blocks)
            quadtree_decomposition(image, threshold, min_block_size, start_x, start_y + sub_block_size, sub_block_size, homogeneous_blocks, non_homogeneous_blocks)
            quadtree_decomposition(image, threshold, min_block_size, start_x + sub_block_size, start_y + sub_block_size, sub_block_size, homogeneous_blocks, non_homogeneous_blocks)
        else:
            position+=1
            homogeneous_blocks.append((start_x, start_y, block_size, position))

    return homogeneous_blocks, non_homogeneous_blocks



# Load the image
image = cv2.imread('1.jpeg', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded
if image is None:
    print('Could not open or find the image')
else:
    # Define parameters
    threshold = 100  # Intensity difference threshold
    min_block_size = 4  # Minimum block size

    # Calculate the nearest dimensions that are divisible by 4 up to a maximum size
    max_size = 1024  # Maximum size for precalculation
    valid_sizes = [4 * i for i in range(1, max_size // 4 + 1)]

    # Get the dimensions of the image
    height, width = image.shape
    best_size = min(valid_sizes, key=lambda x: abs(x - height) + abs(x - width))

    print('Best size:', best_size)
    print('height:', height)
    print('width:', width)

    # Resize the image to the nearest calculated dimensions
    image = cv2.resize(image, (256, 256))

    # Perform quadtree decomposition
    homogeneous_blocks, non_homogeneous_blocks = quadtree_decomposition(image, threshold, min_block_size)

    # Categorize blocks
    smooth_blocks, texture_blocks = categorize_blocks(image, threshold)

    # Create an image view

    # for i in range(0,len(image)):
    #     for j in range(0,len(image[0])):
    #         image[i][j] = 0


    image_view = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


    # Create a new array to store block positions
    position_matrix = np.zeros_like(image, dtype=int)

    # Populate the position matrix with the position values of homogeneous and non-homogeneous blocks

    print(homogeneous_blocks[0])
    for block in homogeneous_blocks:
        x, y, size, position = block
        for i in range(x,x+size):
            for j in range(y,y+size):
                position_matrix[j][i] = position

    for block in non_homogeneous_blocks:
        x, y, size, position = block
        for i in range(x,x+size):
            for j in range(y,y+size):
                position_matrix[j][i] = position

    for i in range(1,50):
        for j in range(1,50):
            print(position_matrix[i][j] , end=" ")
        print(end="\n")

    # # Draw homogeneous blocks in green and non-homogeneous blocks in red
    # for block in homogeneous_blocks:
    #     x, y, size, _ = block
    #     cv2.rectangle(image_view, (x, y), (x + size, y + size), (0, 255, 0), 1)

    

    # for block in non_homogeneous_blocks:
    #     x, y, size, _ = block 
    #     cv2.rectangle(image_view, (x, y), (x + size, y + size), (0, 0, 255), 1)


    # # Color homogeneous blocks in green and non-homogeneous blocks in red
    # for block in homogeneous_blocks:
    #     x, y, size, _ = block
    #     image_view[y:y+size, x:x+size] = (0, 255, 0)  # Green

    # for block in non_homogeneous_blocks:
    #     x, y, size, _ = block
    #     image_view[y:y+size, x:x+size] = (0, 0, 255)  # Red


    # for block in smooth_blocks:
    #     x,y,_ = block
    #     size = 4
    #     cv2.rectangle(image_view, (x, y), (x + size, y + size), (0, 255, 0), 1)

    # for block in texture_blocks:
    #     x,y,_ = block
    #     size = 4
    #     cv2.rectangle(image_view, (x, y), (x + size, y + size), (0, 0, 255), 1)


    # for block in non_homogeneous_blocks:
    #     print(block)


    mapping_smooth = {}
    mapping_texture = {}

    for smooth_block in smooth_blocks:
        x, y, position = smooth_block
        mapping_smooth[position] = position_matrix[x][y]

    for texture_block in mapping_texture:
        x, y, position = texture_block
        mapping_texture[position] = position_matrix[x][y]

    
    print(mapping_smooth)

    

    


    print(len(homogeneous_blocks))
    print(len(non_homogeneous_blocks))

    print(len(smooth_blocks))
    print(len(texture_blocks))


    # # Display the image view
    # cv2.imshow('Image View', image_view)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()