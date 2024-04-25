import cv2
import numpy as np
import galois
import random



position = 1

def psnr(image1, image2):
    # Convert images to numpy arrays
    img1 = np.array(image1)
    img2 = np.array(image2)
    
    # Calculate the mean squared error (MSE)
    mse = np.mean((img1 - img2) ** 2)
    
    # If MSE is zero, PSNR is infinite, return a very large value instead
    if mse == 0:
        return float('inf')
    
    # Calculate the maximum possible pixel value
    max_pixel = np.max(img1)
    
    # Calculate PSNR using the formula: PSNR = 20 * log10(MAX) - 10 * log10(MSE)
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr_value

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

        # Extract the sub-image corresponding to the entire block
        entire_block_image = image[start_y:start_y + block_size, start_x:start_x + block_size]

        # Calculate the maximum intensity difference within the entire block
        max_intensity_diff = entire_block_image.max() - entire_block_image.min()

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
    def __init__(self, image ,image1):
        image = cv2.resize(image, (256, 256))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image1 = cv2.resize(image1 , (256,256))
        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

        # rotation_angle = 0.10  # 45 degrees in radians
        # translation = (0, 0)  # Translation of (20, -10) pixels
        # scaling_factor = 1.1  # Scaling factor of 1.5
        # self.image = image
        # Call the normalize_image method with the example values
        # image = self.normalize_image(rotation_angle, translation, scaling_factor)
        self.image = gray_image
        self.image1 = gray_image1

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
    
    def check_homogeneity(self, block, threshold=10):
        """
        Check if a given block is homogeneous based on the intensity difference threshold.

        Args:
            block: A numpy array representing the image block.
            threshold: Intensity difference threshold for determining homogeneity.

        Returns:
            bool: True if the block is homogeneous, False otherwise.
        """
        max_intensity_diff = np.max(block) - np.min(block)
        return max_intensity_diff <= threshold
    


    def divide_and_categorize_blocks(self, image, threshold=10):
        """
        Divide the normalized image into 4x4 sub-blocks and categorize them as homogeneous or non-homogeneous.

        Args:
            image: A numpy array representing the normalized grayscale image.
            threshold: Intensity difference threshold for determining homogeneity.

        Returns:
            smooth_blocks: List of tuples containing information about smooth blocks (start_x, start_y, block_size).
            texture_blocks: List of tuples containing information about texture blocks (start_x, start_y, block_size).
        """
        smooth_blocks = []
        texture_blocks = []
        position = 0
        # Calculate number of 4x4 sub-blocks in each dimension
        num_blocks_x = image.shape[1] // 4
        num_blocks_y = image.shape[0] // 4

        # Iterate through each 4x4 sub-block
        for y in range(num_blocks_y):
            for x in range(num_blocks_x):
                start_x = x * 4
                start_y = y * 4
                sub_block = image[start_y:start_y+4, start_x:start_x+4]

                # Check homogeneity of the sub-block
                if self.calculate_entropy(sub_block)<=3:
                    smooth_blocks.append((start_x, start_y, 4, position))
                else:
                    texture_blocks.append((start_x, start_y, 4, position))
                position+=1

        return smooth_blocks, texture_blocks


    def calculate_entropy(self , sub_block):
        """
        Calculate the entropy of a given sub-block.

        Args:
            sub_block: A numpy array representing the sub-block.

        Returns:
            entropy: The entropy value of the sub-block.
        """
        # Compute the normalized histogram of the sub-block
        hist, _ = np.histogram(sub_block, bins=256, range=(0, 255), density=True)

        # Compute entropy using the histogram
        entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))

        return entropy
        
    def modify_embedded_image(self, watermarked_image, homogeneous_blocks, num_blocks_to_modify):
        """
        Modifies the embedded image by changing the values of a random number of homogeneous blocks.

        Args:
            watermarked_image: The watermarked image with the recovery watermark embedded.
            homogeneous_blocks: List of tuples containing information about homogeneous blocks (start_x, start_y, block_size, position).
            num_blocks_to_modify: Number of homogeneous blocks to modify.

        Returns:
            modified_image: The modified watermarked image.
            modified_indices: List of indices of the modified homogeneous blocks.
        """
        modified_image = watermarked_image.copy()
        modified_indices = []

        # Shuffle the homogeneous blocks list to randomly select blocks
        random.shuffle(homogeneous_blocks)

        # Select a random number of blocks to modify
        blocks_to_modify = random.sample(homogeneous_blocks, num_blocks_to_modify)

        for block_info in blocks_to_modify:
            start_x, start_y, block_size, _ = block_info

            # Generate a random value to replace the original block
            new_value = random.randint(180, 255)

            # Modify the block in the image
            modified_image[start_y:start_y + block_size, start_x:start_x + block_size] = new_value

            # Store the index of the modified block
            modified_indices.append(homogeneous_blocks.index(block_info))

        return modified_image, modified_indices


    def map_blocks(self):
        # Establish mapping function between original and watermarked sub-blocks
        pass

    def embed_watermark(self, homogeneous_blocks, smooth_blocks):
        """
        Embeds the recovery watermark from homogeneous blocks into smooth blocks and keeps track of the mapping.

        Args:
            homogeneous_blocks: List of tuples containing information about homogeneous blocks (start_x, start_y, block_size, position).
            smooth_blocks: List of tuples containing information about smooth blocks (start_x, start_y, block_size).

        Returns:
            watermarked_image: A copy of the original image with the recovery watermark embedded.
            mapping: Dictionary containing the mapping between homogeneous blocks and smooth blocks.
        """
        watermarked_image = self.image.copy()
        mapping = {}

        index = 0

        # Iterate over each homogeneous block
        for hom_block_info in homogeneous_blocks:
            hom_start_x, hom_start_y, hom_block_size, position = hom_block_info
            hom_block = self.image[hom_start_y:hom_start_y + hom_block_size, hom_start_x:hom_start_x + hom_block_size]

            # Calculate the average value of the homogeneous block
            average_value = int(np.mean(hom_block))


            # # Convert the average value to an 8-bit binary string
            binary_string = format(average_value & 0xFF, '08b')
           
            gf_array = np.array([int(bit) for bit in binary_string], dtype=np.uint8)


            mapping[hom_block_info] = smooth_blocks[index]

            # print(average_value)
            # print(binary_string)

            # Create a BCH code object.
            bch = galois.BCH(15, 11)
            encoded_integer = bch.encode(gf_array)

            # encoded_binary_string = format(encoded_integer, '015b')

            # gf = ffield.FField(15)

            # encoded_data = gf.EncodeBCH(average_value, 15 ,11)
            # print("code" , encoded_integer) 

            # encoded_integer = bch.decode(encoded_integer)

            # print("decode" , encoded_integer)

            # Define constants for LSB embedding
            LSB_MASK = 0xFE  # Mask to clear the LSB of a pixel value
            LSB_COUNT = 12   # Number of LSBs to embed

            # water embeding 
            start_x , start_y , block_size , s_position = smooth_blocks[index]
            encoded_bit_index = 0
           
            for y in range(start_y, start_y + block_size):
                for x in range(start_x, start_x + block_size):
                    # Get the pixel value from the watermarked image
                    pixel_value = watermarked_image[y, x]

                    # print("pixel_value" , pixel_value)

                    # Clear the LSB to make room for the encoded bit
                    modified_pixel_value = pixel_value & LSB_MASK

                    # print("modified_pixel_value",modified_pixel_value)

                    encoded_bit = int(encoded_integer[encoded_bit_index] & 1)

                    # print("encoded_bit",type(encoded_bit))

                    modified_pixel_value |= encoded_bit

                    # print("modified_pixel_value" , modified_pixel_value)



                    # # Get the LSB of the encoded integer
                    # encoded_bit = (encoded_integer >> encoded_bit_index) & 1

                    # print("encoded_bit" , encoded_bit)

                    # # Embed the LSB of the encoded integer into the pixel
                    # modified_pixel_value |= encoded_bit

                    # print("modified_pixel_value" , modified_pixel_value)

                    # Update the pixel value in the watermarked image
                    watermarked_image[y, x] = modified_pixel_value

                    # Increment the index to move to the next bit of the encoded integer
                    encoded_bit_index += 1

                    # If all LSBs have been embedded, break out of the loop
                    if encoded_bit_index >= LSB_COUNT:
                        break
                if encoded_bit_index >= LSB_COUNT:
                    break

            index+=1


        return watermarked_image, mapping



    

    def embed_watermark_non_homogeneous(self, non_homogeneous_blocks, texture_blocks, image):
        """
        Embeds the recovery watermark from non-homogeneous blocks into texture blocks and keeps track of the mapping.

        Args:
            non_homogeneous_blocks: List of tuples containing information about non-homogeneous blocks (start_x, start_y, block_size, position).
            texture_blocks: List of tuples containing information about texture blocks (start_x, start_y, block_size).

        Returns:
            watermarked_image: A copy of the original image with the recovery watermark embedded.
            mapping: Dictionary containing the mapping between non-homogeneous blocks and texture blocks.
        """
        watermarked_image = image.copy()
        mapping = {}
        index = 0

        # Iterate over each non-homogeneous block
        for non_hom_block_info in non_homogeneous_blocks:
            non_hom_start_x, non_hom_start_y, non_hom_block_size, position = non_hom_block_info
            non_hom_block = self.image[non_hom_start_y:non_hom_start_y + non_hom_block_size, non_hom_start_x:non_hom_start_x + non_hom_block_size]

            # Extract features from the non-homogeneous block (placeholder)
            feature_info = 'example_feature_info'

            # Find a suitable texture block to embed the watermark
            for texture_block_info in texture_blocks:
                texture_start_x, texture_start_y, texture_block_size,position1 = texture_block_info
                texture_block = watermarked_image[texture_start_y:texture_start_y + texture_block_size, texture_start_x:texture_start_x + texture_block_size]

                # Check if the texture block has enough space to embed the watermark
                if len(feature_info) <= texture_block.size:
                    # Embed the watermark into the texture block
                    # watermarked_block = self.embed_watermark_into_block(watermarked_image,texture_block, feature_info)

                    # Update the watermarked image with the embedded watermark
                    # watermarked_image[texture_start_y:texture_start_y + texture_block_size, texture_start_x:texture_start_x + texture_block_size] = watermarked_block

                    # Keep track of the mapping between non-homogeneous and texture blocks
                    mapping[position] = position1

                    index+=1

                    break  # Break the loop after embedding the watermark

        return watermarked_image, mapping



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
        smooth_blocks , texture_blocks = self.divide_and_categorize_blocks(self.image)

        
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



        # Sort the lists based on position
        homogeneous_blocks = sorted(homogeneous_blocks, key=lambda x: x[3])
        non_homogeneous_blocks = sorted(non_homogeneous_blocks, key=lambda x: x[3])
        smooth_blocks = sorted(smooth_blocks, key=lambda x: x[2])  # Sort by block size
        texture_blocks = sorted(texture_blocks, key=lambda x: x[2])  # Sort by block size

        # Print size of homogeneous and non-homogeneous blocks
        print("Number of homogeneous blocks:", len(homogeneous_blocks))
        print("Number of non-homogeneous blocks:", len(non_homogeneous_blocks))
        print("Smooth_blocks" , len(smooth_blocks))  
        print("Texture_blocks" , len(texture_blocks))

         # Embed watermark from homogeneous blocks
        watermarked_image_homogeneous, mapping_homogeneous = self.embed_watermark(homogeneous_blocks, smooth_blocks)

        psnr_value = psnr(watermarked_image_homogeneous, self.image)
        print("PSNR:", psnr_value)

        # self.tamper_detection(homogeneous_blocks, non_homogeneous_blocks , self.image1)

        
        # cv2.imwrite("watermarked_image_homogeneous.png", watermarked_image_homogeneous)

        print("Watermarked image saved successfully.")

        modified_image, modified_indices = self.modify_embedded_image(watermarked_image_homogeneous,homogeneous_blocks, 12)

        tampered_blocks =self.tamper_detection(homogeneous_blocks, non_homogeneous_blocks , modified_image)

        # cv2.imshow("watermarking" , modified_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Embed watermark from non-homogeneous blocks
        # watermarked_image_final, mapping_non_homogeneous = self.embed_watermark_non_homogeneous(non_homogeneous_blocks, texture_blocks , watermarked_image_homogeneous)
        # # Generate recovery watermark for each block
        # recovery_watermark = self.generate_recovery_watermark(homogeneous_blocks, non_homogeneous_blocks)

        #step 2

        # cv2.imshow('Watermark Image', watermarked_image_final)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Print the output
        # print("Normalized Image:", self.image)
        

        recovered_image=self.image_recovery(modified_image,mapping_homogeneous,tampered_blocks)
        psnr_value = psnr(recovered_image, self.image)
        print("PSNR:22222", psnr_value)
        cv2.imshow("watermarking" , recovered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    def image_recovery(self, watermarked_image, mapping, tampered_blocks):
        recovered_image = watermarked_image.copy()
        tampered_blocks_after_recovery = []

    # Iterate over each tampered block
        for tam_block_info in tampered_blocks:
            tam_start_x, tam_start_y, tam_block_size, position = tam_block_info
            tam_block = watermarked_image[tam_start_y:tam_start_y + tam_block_size, tam_start_x:tam_start_x + tam_block_size]

            # Retrieve the position of the corresponding smooth block
            smooth_block = mapping[tam_block_info]

            # Extract the recovery watermark bits from the smooth block
            recovered_bits = self.extract_recovered_bits(recovered_image, smooth_block)

            # Update the tampered block based on the extracted bits
            tampered_block_after_recovery = self.update_tampered_block(tam_block, recovered_bits)

            # Update the recovered image with the recovered tampered block
            recovered_image[tam_start_y:tam_start_y + tam_block_size, tam_start_x:tam_start_x + tam_block_size] = tampered_block_after_recovery

        return recovered_image

    def extract_recovered_bits(self, watermarked_image, smooth_position):
        """
        Extracts the recovery watermark bits from the smooth block in the watermarked image.

        Args:
            watermarked_image: Watermarked image with the recovery watermark embedded.
            smooth_position: Tuple containing information about the smooth block (start_x, start_y, block_size, position).

        Returns:
            recovered_bits: Array of recovered watermark bits extracted from the smooth block.
        """
        recovered_bits = []

        start_x, start_y, block_size, position = smooth_position

        for y in range(start_y, start_y + block_size):
            for x in range(start_x, start_x + block_size):
                pixel_value = watermarked_image[y, x]
                extracted_bit = pixel_value & 1
                recovered_bits.append(extracted_bit)

        return recovered_bits

    def update_tampered_block(self, tam_block, recovered_bits):
        """
        Updates the tampered block based on the recovered watermark bits.

        Args:
            tam_block: Tampered block from the watermarked image.
            recovered_bits: Array of recovered watermark bits extracted from the corresponding smooth block.

        Returns:
            tampered_block_after_recovery: Updated tampered block after applying the recovered bits.
        """
        tampered_block_after_recovery = tam_block.copy()

        for i in range(len(recovered_bits)):
            bit_value = recovered_bits[i]
            tampered_block_after_recovery.flat[i] &= 0xFE  # Clear the LSB
            tampered_block_after_recovery.flat[i] |= bit_value  # Update the LSB with the recovered bit

        return tampered_block_after_recovery

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
        tampered_blocks=[]
        
                


        # # Decompose the normalized image based on multiple scales
        # new_homogeneous_blocks, non_homogeneous_blocks = self.decompose_image(self.image)
    
        for block_info in homogeneous_blocks:
            start_x, start_y, block_size, _ = block_info
            block = original_image[start_y:start_y + block_size, start_x:start_x + block_size]
         
            maxi_block = max(block.flatten())  # Find maximum intensity in the block
            mini_block = min(block.flatten())  # Find minimum intensity in the block

            max_intensity_diff = maxi_block - mini_block

            if max_intensity_diff > THRESHOLD_HOMOGENEOUS+1:  # Define a threshold for homogeneous block variation
                cv2.rectangle(tampered_image, (start_x, start_y), (start_x + block_size, start_y + block_size), (0, 0, 255), -1)
                tampered_blocks.append(block_info)
                  # Fill red for tampered

        # for block_info in non_homogeneous_blocks:
        #     start_x, start_y, block_size, _ = block_info
        #     block = original_image[start_y:start_y + block_size, start_x:start_x + block_size]
            
        #     maxi_block = max(block.flatten())  # Find maximum intensity in the block
        #     mini_block = min(block.flatten())  # Find minimum intensity in the block

        #     max_intensity_diff = maxi_block - mini_block

        #     if max_intensity_diff <= THRESHOLD_HOMOGENEOUS:  # Define a threshold for homogeneous block variation
        #         cv2.rectangle(tampered_image, (start_x, start_y), (start_x + block_size, start_y + block_size), (255, 255, 255), -1)  # Fill red for tampered    
        
       
        # cv2.imshow('Tampered Image', tampered_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return tampered_blocks
    
        
                
# Usage
image1 = load_image("edit2.png")
image2 = load_image("lenna.jpeg")
watermark_algorithm = WatermarkEmbeddingAlgorithm(image2,image1)
homogeneous_blocks , non_homogeneous_blocks = watermark_algorithm.embed_watermark_image()
image2 = cv2.resize(image2, (256, 256))
image1 = cv2.resize(image1, (256, 256))
# watermark_algorithm.tamper_detection(homogeneous_blocks , non_homogeneous_blocks , image1)
