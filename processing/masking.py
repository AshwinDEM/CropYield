import os
import json
import numpy as np
from PIL import Image, ImageDraw

def create_mask_from_polygons(image_size, polygons):
    mask = Image.new('L', image_size, 0)  # 'L' mode for grayscale
    draw = ImageDraw.Draw(mask)
    
    for polygon in polygons:
        # Ensure polygon is closed
        if polygon[0] != polygon[-1]:
            polygon.append(polygon[0])
        draw.polygon(polygon, outline=255, fill=255)
    
    # Test whether the drawing is happening correctly
    # polygon = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
    
    return np.array(mask)

def process_annotation_file(annotation_file, image_folder, mask_folder):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)
    
    for img_name, img_data in data.items():
        image_path = os.path.join(image_folder, img_data['filename'])
        mask_path = os.path.join(mask_folder, img_data['filename'].replace('.png', '_mask.png'))
        
        # Load image to get dimensions
        with Image.open(image_path) as img:
            image_size = img.size

        print("Image Size:", image_size)

        
        polygons = []
        for region in img_data['regions']:
            shape_attributes = region['shape_attributes']
            points_x = shape_attributes['all_points_x']
            points_y = shape_attributes['all_points_y']
            print("Points X:", points_x)
            print("Points Y:", points_y)
            polygon = list(zip(points_x, points_y))
            polygons.append(polygon)
        
        # Create mask
        mask = create_mask_from_polygons(image_size, polygons)
        
        # Debug output
        print("Unique values in mask:", np.unique(mask))
        
        # Save mask
        Image.fromarray(mask).save(mask_path)
        print(f"Mask saved to {mask_path}")

# Example usage
annotation_file = 'test/annotations2.json'
image_folder = 'test/images'
mask_folder = 'test/masks'

process_annotation_file(annotation_file, image_folder, mask_folder)
