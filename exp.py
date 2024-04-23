def decompose_image(self, image):
  """
  Decomposes an image into multiple scales based on homogeneity criterion.

  Args:
      image: A numpy array representing the image.

  Returns:
      A list of lists containing decomposed image blocks.
  """
  # Check if image size is divisible by 4 for minimum block size requirement
  if image.shape[0] % 4 != 0 or image.shape[1] % 4 != 0:
    # Extend image with zeros to meet the requirement
    padded_image = np.pad(image, ((0, (4 - image.shape[0] % 4) // 2),
                                  ((0, (4 - image.shape[1] % 4) // 2))), mode='constant')
  else:
    padded_image = image

  # Recursive function to decompose image blocks
  def decompose_block(block, level):
    """
    Decomposes a single image block based on homogeneity criterion.

    Args:
        block: A numpy array representing the image block.
        level: The current decomposition level.

    Returns:
        A list containing decomposed blocks or the original block if homogeneous.
    """
    # Calculate average gray value
    average_gray = np.mean(block)

    # Define homogeneity criterion threshold
    threshold = (np.max(block) - 1) * 0.1  # Assuming gamma=0.1 here

    # Check if block meets homogeneity criterion
    if np.all(np.abs(block - average_gray) <= threshold):
      return [block]  # Block is homogeneous, return as-is

    # Split block into sub-blocks
    half_height, half_width = block.shape[0] // 2, block.shape[1] // 2
    sub_blocks = [
        decompose_block(block[:half_height, :half_width], level + 1),
        decompose_block(block[:half_height, half_width:], level + 1),
        decompose_block(block[half_height:, :half_width], level + 1),
        decompose_block(block[half_height:, half_width:], level + 1),
    ]

    # Flatten the list of sub-blocks (recursive case)
    return [item for sublist in sub_blocks for item in sublist]

  # Start decomposition from the entire image
  decomposed_blocks = decompose_block(padded_image, 1)

  # Return the decomposed image blocks
  return decomposed_blocks