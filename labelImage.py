from PIL import Image, ImageDraw, ImageFont
import os

# Path to your images directory
images_directory = '/Users/harjotsingh/Desktop/repos/ComputerVisionProjectLLM/garbagedata/train/recycling'
output_directory = '/Users/harjotsingh/Desktop/repos/ComputerVisionProjectLLM/garbagedata/labled/recycling'

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Loop through all images in the directory
for filename in os.listdir(images_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Open the image
        image_path = os.path.join(images_directory, filename)
        img = Image.open(image_path)

        # Create a draw object
        draw = ImageDraw.Draw(img)

        # Get image dimensions
        width, height = img.size

        # Define rectangle properties (covering almost the whole image)
        margin = 10  # Set a small margin
        top_left = (0 + margin, 0 + margin)  # Top-left corner with margin
        bottom_right = (width - margin, height - margin)  # Bottom-right corner with margin

        # Draw the rectangle around the image
        draw.rectangle([top_left, bottom_right], outline="red", width=3)

        # Define the label
        label = "recycling"
        
        # Load a font
        font = ImageFont.load_default()  # You can load a specific font if needed

        # Calculate text size
        text_bbox = draw.textbbox((0, 0), label, font=font)  # Use textbbox to get the bounding box
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Calculate text position (top-left corner of the rectangle with some margin)
        text_x = top_left[0] + 5  # Adding a small margin from the left
        text_y = top_left[1] + 5  # Adding a small margin from the top

        # Draw a black rectangle behind the text
        background_bbox = (text_x, text_y, text_x + text_width + 10, text_y + text_height + 5)  # +10 and +5 for padding
        draw.rectangle(background_bbox, fill="black")

        # Add the label to the image
        draw.text((text_x + 5, text_y + 5), label, fill="white", font=font)  # Adjusting text position for padding

        # Save the modified image
        output_path = os.path.join(output_directory, filename)
        img.save(output_path)

print("Images processed and saved in the output directory.")

