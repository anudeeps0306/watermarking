import cv2
import numpy as np

position = 1

def quadtree_decomposition(image, threshold=10, min_block_size=4, start_x=0, start_y=0, block_size=None, homogeneous_blocks=[], non_homogeneous_blocks=[]):
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
    print("threshold", threshold)
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


def load_image(image_path):
    """
    Load an image from the given file path using OpenCV.

    Args:
        image_path (str): The file path to the image.

    Returns:
        numpy.ndarray: The loaded image as a NumPy array.
    """
    return cv2.imread(image_path)


class WatermarkEmbeddingAlgorithm:
    def __init__(self, image):
        image = cv2.resize(image, (256, 256))
        # rotation_angle = 0.10  # 45 degrees in radians
        # translation = (0, 0)  # Translation of (20, -10) pixels
        # scaling_factor = 1.1  # Scaling factor of 1.5
        # self.image = image
        # Call the normalize_image method with the example values
        # image = self.normalize_image(rotation_angle, translation, scaling_factor)
        self.image = image

    def normalize_image(self, rotation_angle, translation, scaling_factor):
        """
        Normalize an image based on rotation, translation, and scaling parameters.

        Args:
            image (numpy.ndarray): The input image as a NumPy array.
            rotation_angle (float): The rotation angle in radians (φ).
            translation (tuple): Tuple of translation values (a, b).
            scaling_factor (float): The scaling factor (δ).

        Returns:
            numpy.ndarray: The normalized image.
        """
        # Get image dimensions
        height, width = self.image.shape[:2]

        # Define the affine transformation matrix
        affine_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                                [np.sin(rotation_angle), np.cos(rotation_angle), 0]])

        # Apply shear transformation
        shear_matrix = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [translation[0], translation[1], 1]])

        # Apply scaling transformation
        scaling_matrix = np.array([[scaling_factor, 0, 0],
                                [0, scaling_factor, 0],
                                [0, 0, 1]])

        # Combine the transformation matrices
        combined_matrix = np.dot(affine_matrix, np.dot(scaling_matrix, shear_matrix))

        # Apply the transformation to the image
        normalized_image = cv2.warpAffine(self.image, combined_matrix[:2, :], (width, height))

        return normalized_image

    def decompose_image(self, image, threshold=10, min_block_size=4):
        """
        Decomposes an image into homogeneous and non-homogeneous blocks using quadtree decomposition.

        Args:
            image: A numpy array representing the grayscale image.
            threshold: Float, intensity difference threshold for splitting blocks.
            min_block_size: Integer, minimum block size for termination (default 4).

        Returns:
            homogeneous_blocks: List of tuples containing information about homogeneous blocks (start_x, start_y, block_size, position).
            non_homogeneous_blocks: List of tuples containing information about non-homogeneous blocks (start_x, start_y, block_size, position).
        """

        homogeneous_blocks, non_homogeneous_blocks = [], []
        quadtree_decomposition(image, threshold, min_block_size, 0, 0, image.shape[0], homogeneous_blocks, non_homogeneous_blocks)
        return homogeneous_blocks, non_homogeneous_blocks
        

    def generate_recovery_watermark(self, image, homogeneous_blocks, non_homogeneous_blocks):
        """
        Generates recovery watermark for homogeneous and non-homogeneous blocks with positions.

        Args:
            image: A numpy array representing the normalized grayscale image.
            homogeneous_blocks: List of tuples containing information about homogeneous blocks (start_x, start_y, block_size, position).
            non_homogeneous_blocks: List of tuples containing information about non-homogeneous blocks (start_x, start_y, block_size, position).

        Returns:
            homogeneous_watermarks: List of tuples containing (position, recovery_watermark) for homogeneous blocks.
            non_homogeneous_watermarks: List of tuples containing (position, recovery_watermark) for non-homogeneous blocks.
        """

        homogeneous_watermarks = []
        non_homogeneous_watermarks = []

        # Process homogeneous blocks
        for block_info in homogeneous_blocks:
            start_x, start_y, block_size, position = block_info
            block = image[start_y:start_y + block_size, start_x:start_x + block_size]
            average_value = np.mean(block)
            binary_string = format(int(average_value), '08b')  # Convert average to 8-bit binary string
            homogeneous_watermarks.append((position, binary_string))

        # Process non-homogeneous blocks
        for block_info in non_homogeneous_blocks:
            start_x, start_y, block_size, position = block_info
            sub_block_size = block_size // 2

            # Divide into non-overlapping sub-blocks
            sub_blocks = [
                image[start_y:start_y + sub_block_size, start_x:start_x + sub_block_size],
                image[start_y:start_y + sub_block_size, start_x + sub_block_size:start_x + block_size],
                image[start_y + sub_block_size:start_y + block_size, start_x:start_x + sub_block_size],
                image[start_y + sub_block_size:start_y + block_size, start_x + sub_block_size:start_x + block_size]
            ]

            sub_block_watermarks = []
            for sub_block in sub_blocks:
                # Implement sub-category encoding based on maximum pixel positions (refer to Fig 4 in the paper)
                # ... (fill in sub-category encoding logic here)
                # Calculate high-order average and difference as described in the paper
                high_avg_binary = format(int(np.mean(sub_block) >> 6), '06b')  # High 6 bits of average
                diff_binary = format((np.max(sub_block) + np.max(sub_block, axis=0))[0] - (np.min(sub_block) + np.min(sub_block, axis=0))[0] // 2, '02b')  # Difference between max and min sums (quantized)
                sub_block_watermarks.append(high_avg_binary + '*' + diff_binary)  # Concatenate with separator

            # Concatenate sub-block watermarks and convert to a single binary string (11 bits)
            sub_block_watermark_str = ''.join(sub_block_watermarks)
            non_homogeneous_watermarks.append((position, sub_block_watermark_str))

        # Encryption and scrambling (mentioned in the paper) are not implemented here
        # You can add them as separate steps based on your specific watermarking scheme

        return homogeneous_watermarks, non_homogeneous_watermarks

    def divide_into_sub_blocks(self):
        # Divide the normalized image into non-overlapping sub-blocks
        pass

    def calculate_entropy(self):
        # Calculate entropy for each sub-block
        pass

    def map_blocks(self):
        # Establish mapping function between original and watermarked sub-blocks
        pass

    def embed_watermark(self):
        # Embed watermark information using difference expansion and LSB algorithms
        pass

    def calculate_invariant_distance(self):
        # Calculate the invariant distance of the embedded recovery watermark image
        pass

    def embed_watermark_image(self):
        # Example values for rotation_angle, translation, and scaling_factor
        # Resize the image to the nearest calculated dimensions
        # image = cv2.resize(self.image, (256, 256))
        # rotation_angle = 0.10  # 45 degrees in radians
        # translation = (0, 0)  # Translation of (20, -10) pixels
        # scaling_factor = 1.1  # Scaling factor of 1.5

        # # Call the normalize_image method with the example values
        # normalized_image = self.normalize_image(rotation_angle, translation, scaling_factor)
        

        # Decompose the normalized image based on multiple scales
        homogeneous_blocks, non_homogeneous_blocks = self.decompose_image(self.image)

        # Print size of homogeneous and non-homogeneous blocks
        print("Number of homogeneous blocks:", len(homogeneous_blocks))
        print("Number of non-homogeneous blocks:", len(non_homogeneous_blocks))


        # # Generate recovery watermark for each block
        # recovery_watermark = self.generate_recovery_watermark(homogeneous_blocks, non_homogeneous_blocks)



        # # Draw homogeneous blocks in green and non-homogeneous blocks in red
        # for block in homogeneous_blocks:
        #     x, y, size, _ = block
        #     cv2.rectangle(self.image, (x, y), (x + size, y + size), (0, 255, 0), 1)

        

        # for block in non_homogeneous_blocks:
        #     x, y, size, _ = block 
        #     cv2.rectangle(self.image, (x, y), (x + size, y + size), (0, 0, 255), 1)
    


        # cv2.imshow('Image View', self.image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Print the output
        print("Normalized Image:", self.image)

        return homogeneous_blocks , non_homogeneous_blocks

    def tamper_detection(self , homogeneous_blocks, non_homogeneous_blocks ,  original_image):
        """
        Detects tampered regions in the image using homogeneous and non-homogeneous blocks.

        Args:
            image: A numpy array representing the potentially tampered image.
            homogeneous_blocks: List of tuples containing information about homogeneous blocks (start_x, start_y, block_size, position).
            non_homogeneous_blocks: List of tuples containing information about non-homogeneous blocks (start_x, start_y, block_size, position).
            original_image (optional): A numpy array representing the original image (for reference).

        Returns:
            tampered_image: A copy of the image with tampered regions marked in red.
        """
         
        THRESHOLD_HOMOGENEOUS = 10
         
        tampered_image = original_image.copy()

                


        # # Decompose the normalized image based on multiple scales
        # new_homogeneous_blocks, non_homogeneous_blocks = self.decompose_image(self.image)
    
        for block_info in homogeneous_blocks:
            start_x, start_y, block_size, _ = block_info
            block = original_image[start_y:start_y + block_size, start_x:start_x + block_size]
         
            maxi_block = max(block.flatten())  # Find maximum intensity in the block
            mini_block = min(block.flatten())  # Find minimum intensity in the block

            max_intensity_diff = maxi_block - mini_block

            if max_intensity_diff > THRESHOLD_HOMOGENEOUS:  # Define a threshold for homogeneous block variation
                cv2.rectangle(tampered_image, (start_x, start_y), (start_x + block_size, start_y + block_size), (0, 0, 255), -1)  # Fill red for tampered

    
        for block_info in non_homogeneous_blocks:
            start_x, start_y, block_size, _ = block_info
            block = original_image[start_y:start_y + block_size, start_x:start_x + block_size]
            
            maxi_block = max(block.flatten())  # Find maximum intensity in the block
            mini_block = min(block.flatten())  # Find minimum intensity in the block

            max_intensity_diff = maxi_block - mini_block

            if max_intensity_diff <= THRESHOLD_HOMOGENEOUS:  # Define a threshold for homogeneous block variation
                cv2.rectangle(tampered_image, (start_x, start_y), (start_x + block_size, start_y + block_size), (0, 0, 255), -1)  # Fill red for tampered    
        
       
        cv2.imshow('Tampered Image', tampered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
                


# Usage
image1 = load_image("edit.png")
image2 = load_image("lenna.jpeg")
watermark_algorithm = WatermarkEmbeddingAlgorithm(image2)
homogeneous_blocks , non_homogeneous_blocks = watermark_algorithm.embed_watermark_image()
image2 = cv2.resize(image2, (256, 256))
image1 = cv2.resize(image1, (256, 256))
watermark_algorithm.tamper_detection(homogeneous_blocks , non_homogeneous_blocks , image1)
