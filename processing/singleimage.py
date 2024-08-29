from PIL import Image

def resize_and_pad(input_path, output_path, target_size=(1024, 1024)):
    """
    Resizes an image while maintaining aspect ratio and pads it to a target size.

    :param input_path: Path to the input image
    :param output_path: Path to save the resized and padded image
    :param target_size: Desired size as a tuple (width, height)
    """
    with Image.open(input_path) as img:
        # Resize image maintaining aspect ratio
        img.thumbnail(target_size, Image.LANCZOS)

        # Create a new image with the target size and a white background
        new_img = Image.new('RGB', target_size, (255, 255, 255))

        # Paste the resized image onto the center of the new image
        new_img.paste(
            img, ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2)
        )

        # Save the padded image
        new_img.save(output_path)
        print(f"Image saved to {output_path} with size {new_img.size}")

# Example usage
resize_and_pad(
    input_path='data\\folder1\\0.png',  # Input image path
    output_path='path_to_resized_image.jpg',  # Output image path
    target_size=(1024, 1024)  # Desired uniform size (width, height)
)

