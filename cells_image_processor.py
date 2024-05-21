
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:
    @staticmethod
    def change_contrast(image_path, contrast_factor=2.0, brightness_factor=50):
        # Read the cells.jpg image and convert it to LAB color space.
        original_image = cv2.imread(image_path)
        lab_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2Lab)
        # Separate the L, a, and b channels.
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Apply contrast adjustment to the L channel using contrast factor of 2.0.
        adjusted_l_channel = (l_channel.astype(float) * contrast_factor).clip(0, 255).astype(np.uint8)

        # Decrease brightness in the adjusted L channel for better visibility.
        decreased_l_channel = np.clip(adjusted_l_channel.astype(int) - brightness_factor, 0, 255).astype(np.uint8)

        # Merge the adjusted channels back into the LAB image.
        adjusted_lab_image = cv2.merge([decreased_l_channel, a_channel, b_channel])
        # Convert the LAB image back into BGR color space.
        contrasted_image = cv2.cvtColor(adjusted_lab_image, cv2.COLOR_Lab2BGR)

        return contrasted_image

    @staticmethod
    def sharpen_image(contrasted_image, strength = 4.0, brightness_factor = 5, contrast_factor = 1.1):
        # Convert the contrasted image into LAB color space.
        lab_image = cv2.cvtColor(contrasted_image, cv2.COLOR_BGR2LAB)
        # Separate the L, a, and b channels.
        l_channel, _, _ = cv2.split(lab_image)

        # Apply sharpening to the L channel (lightness).
        kernel = np.array( [ [ -1, -1, -1 ],
                             [ -1, strength + 5.0, -1 ],
                             [ -1, -1, -1 ] ] )
        sharpened_l_channel = cv2.filter2D(l_channel, -1, kernel)

        # Increase brightness in the sharpened L channel.
        increased_l_channel = np.clip(sharpened_l_channel.astype(int) + brightness_factor, 0, 255).astype(np.uint8)

        # Apply contrast adjustment to the sharpened L channel.
        sharpened_image = (increased_l_channel.astype( float ) * contrast_factor).clip( 0, 255 ).astype( np.uint8 )

        return sharpened_image

    @staticmethod
    def edge_detection(sharpened_l_channel, threshold = 1, brightness = 127, contrast = 2.0):
        # Convert the sharpened image to grayscale for edge detection.
        grayscale_image = sharpened_l_channel

        # Apply contrast adjustment to the grayscale image.
        adjusted_grayscale_image = cv2.convertScaleAbs(grayscale_image, alpha = contrast, beta = brightness)

        # Apply Gaussian blur to the adjusted grayscale image.
        blurred_image = cv2.GaussianBlur(adjusted_grayscale_image, (5, 5), 0)

        # Perform edge detection using Canny.
        edges = cv2.Canny(blurred_image, 50, 150)

        return edges

# ---------
# Run:
# ---------

# Create an instance of the ImageProcessor class.
image_processor = ImageProcessor()

# Change contrast of the Cells.jpg image.
contrast_changed_image = image_processor.change_contrast('/content/Cells.jpg')

# Display the original and contrast-changed images using Matplotlib.
original_image = cv2.imread('/content/Cells.jpg')  # Read the original image using OpenCV
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Display original image.
axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')

# Display contrast-changed image.
axes[1].imshow(cv2.cvtColor(contrast_changed_image, cv2.COLOR_BGR2RGB))
axes[1].set_title('Contrast Changed Image')
axes[1].axis('off')

# Sharpen the contrast-changed image.
sharpened_l_channel = image_processor.sharpen_image(contrast_changed_image)

# Display sharpened image.
axes[2].imshow(cv2.cvtColor(sharpened_l_channel, cv2.COLOR_BGR2RGB))
axes[2].set_title('Sharpened Image')
axes[2].axis('off')

# Detect edges in the sharpened image and filter them based on the brightness threshold.
edge_detected_image = image_processor.edge_detection(sharpened_l_channel)

# Display edge-detected image.
axes[3].imshow(cv2.cvtColor(edge_detected_image, cv2.COLOR_BGR2RGB))
axes[3].set_title('Edge Detected Image')
axes[3].axis('off')

plt.tight_layout()
plt.show()