import cv2

# Load the image
image = cv2.imread('1.jpeg', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded
if image is None:
    print('Could not open or find the image')
else:
    # Define parameters
    threshold = 10  # Intensity difference threshold
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
    image = cv2.resize(image, (best_size, best_size))

    # # Perform quadtree decomposition
    # homogeneous_blocks, non_homogeneous_blocks = quadtree_decomposition(image, threshold, min_block_size)

    # # Categorize blocks
    # smooth_blocks, texture_blocks = categorize_blocks(image, homogeneous_blocks)

    # Create an image view
    image_view = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Display the image view
    cv2.imshow('Image View', image_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
