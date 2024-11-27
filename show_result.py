import matplotlib.pyplot as plt
from PIL import Image
import os


def show_output(dirs, times, start_index=0, batch_size=8):
    """
    Display images from the specified directory with width adjustments for each type,
    showing each pair of Images_* and Sketch_* in a single row, resizing Sketch_* to match
    the height of Images_*, and automatically showing the next batch after the window is closed.

    Args:
        dirs (str): Directory containing the images.
        times (int): Ratio by which "Images_*" is wider than "Sketch_*".
        start_index (int, optional): Starting index for the images. Default is 0.
        batch_size (int, optional): Number of rows per batch. Default is 8 rows.
    """
    # Group images by prefix (Images_* or Sketch_*)
    images = [f for f in os.listdir(dirs) if f.startswith("Images_")]
    sketches = [f for f in os.listdir(dirs) if f.startswith("Sketch_")]

    # Sort files to maintain order
    images.sort()
    sketches.sort()

    # Combine lists based on index
    total_images = min(len(images), len(sketches))
    selected_images = images[start_index:total_images]
    selected_sketches = sketches[start_index:total_images]

    # Display in batches of 'batch_size'
    for batch_start in range(0, len(selected_images), batch_size):
        batch_end = batch_start + batch_size
        current_images = selected_images[batch_start:batch_end]
        current_sketches = selected_sketches[batch_start:batch_end]

        # Number of rows in the current batch
        num_rows = len(current_images)

        # Create the figure
        fig, axes = plt.subplots(num_rows, 2, figsize=(12, num_rows * 3))

        # Handle single row case for proper axis handling
        if num_rows == 1:
            axes = [axes]

        for idx, (img_name, sketch_name) in enumerate(zip(current_images, current_sketches)):
            img_path = os.path.join(dirs, img_name)
            sketch_path = os.path.join(dirs, sketch_name)

            # Open Images_* and Sketch_* files
            img = Image.open(img_path)
            sketch = Image.open(sketch_path)

            # Resize Sketch_* to match the height of Images_*
            img_width, img_height = img.size
            sketch_width, sketch_height = sketch.size
            aspect_ratio = sketch_width / sketch_height
            new_sketch_width = int(img_height * aspect_ratio)
            sketch = sketch.resize((new_sketch_width, img_height), Image.LANCZOS)

            # Display Images_*
            axes[idx][0].imshow(img)
            axes[idx][0].axis('off')
            axes[idx][0].set_title(img_name, pad=2)  # Reduced padding

            # Display resized Sketch_*
            axes[idx][1].imshow(sketch)
            axes[idx][1].axis('off')
            axes[idx][1].set_title(sketch_name, pad=2)  # Reduced padding

        plt.tight_layout(pad=1.0, h_pad=1.0)  # Reduce padding between rows and elements
        plt.show(block=True)  # Display and wait for the window to close


if __name__ == "__main__":
    show_output("output", 5, batch_size=3)

