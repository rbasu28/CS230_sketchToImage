import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def show_output_all(dirs, times, start_index=0, batch_size=8):
    # Group images by prefix (Images_* or Sketch_*)
    images = [f for f in os.listdir(dirs) if f.startswith("Images_")]
    sketches = [f for f in os.listdir(dirs) if f.startswith("Sketch_")]
    show_output_internal(dirs, times, images, sketches, start_index, batch_size)


def show_output_concise(dirs, times, start_index=0, batch_size=8):
    file_ids = [0, 13, 14, 18, 19, 28]
    images = [f'Images_{i}.png' for i in file_ids]
    sketches = [f'Sketch_{i}.png' for i in file_ids]
    show_output_internal(dirs, times, images, sketches, start_index, batch_size)


def show_output_internal(dirs, times, images, sketches, start_index, batch_size):
    """
    Display images from the specified directory with width adjustments for each type,
    showing Sketch_* on the left and Images_* on the right in a single row,
    resizing Sketch_* to match the height of Images_*, and automatically
    showing the next batch after the window is closed.

    Args:
        dirs (str): Directory containing the images.
        times (int): Ratio by which "Images_*" is wider than "Sketch_*".
        start_index (int, optional): Starting index for the images. Default is 0.
        batch_size (int, optional): Number of rows per batch. Default is 8 rows.
    """
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
        fig, axes = plt.subplots(num_rows, 2, width_ratios=[1, times], frameon=True)

        # Handle single row case for proper axis handling
        if num_rows == 1:
            axes = [axes]

        for idx, (img_name, sketch_name) in enumerate(zip(current_images, current_sketches)):
            img_path = os.path.join(dirs, img_name)
            sketch_path = os.path.join(dirs, sketch_name)

            # Open Images_* and Sketch_* files
            img = Image.open(img_path)
            sketch = Image.open(sketch_path)

            # Display Sketch_* on the left
            axes[idx][0].imshow(sketch)
            axes[idx][0].axis('off')

            # Add border around sketch subplot
            axes[idx][0].add_patch(Rectangle(
                (0, 0), 1, 1, transform=axes[idx][0].transAxes,
                linewidth=2, edgecolor='blue', facecolor='none'
            ))
            # Display Images_* on the right
            axes[idx][1].imshow(img)
            axes[idx][1].axis('off')
            # axes[idx][1].set_title(img_name, pad=0.1)  # Reduced padding

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout(pad=0.1, h_pad=0.1)  # Reduce padding between rows and elements
        plt.show(block=True)  # Display and wait for the window to close


if __name__ == "__main__":
    show_output_all("output", 5, batch_size=10)
    # show_output_concise("output", 5, batch_size=10)


