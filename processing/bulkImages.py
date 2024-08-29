from PIL import Image
import os

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

def process_images_in_folder(input_folder, output_folder, target_size=(1024, 1024)):
    """
    Processes all images in the input folder: resizing and padding them to the target size.
    Saves the processed images to the output folder.

    :param input_folder: Path to the folder containing input images
    :param output_folder: Path to the folder to save processed images
    :param target_size: Desired size as a tuple (width, height)
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through each file in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        # Check if the file is an image by its extension
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            output_path = os.path.join(output_folder, filename)
            
            try:
                # Resize and pad the image
                resize_and_pad(input_path, output_path, target_size)
            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

# Example usage

if __name__ == "__main__":
    for i in range(1, 4):
        process_images_in_folder(
            input_folder=f'data\\folder{i}',  # Replace with your input folder path
            output_folder=f'data\\output{i}',  # Replace with your output folder path
            target_size=(1024, 1024)  # Desired uniform size (width, height)
       )