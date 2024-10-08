Cell Image Processor
Project Overview
This project, titled Cell Image Processor, focuses on the processing of microscopic cell images using a sequence of image transformations. The aim is to enhance image contrast, sharpen the details, and perform edge detection on cell images to reveal significant features for further analysis. This type of processing is especially useful in biomedical imaging, where image clarity and feature extraction are critical for identifying cellular structures and abnormalities.

Key Features
Contrast Adjustment:

Reads the input image and converts it from BGR to LAB color space.
Adjusts the contrast by amplifying the luminance (L channel) of the image, with additional fine-tuning through brightness adjustment.
Image Sharpening:

Sharpens the image by applying a kernel to the luminance (L) channel in the LAB color space.
Enhances brightness and adjusts the contrast of the sharpened image for better visualization of fine details.
Edge Detection:

Converts the sharpened image to grayscale, applies contrast and brightness adjustments, and performs Gaussian blurring.
Uses the Canny edge detection algorithm to highlight the edges in the image, which can reveal cellular structures or boundaries in the image.
Visualization:

Displays a series of images that show the transformation steps: the original image, the contrast-adjusted image, the sharpened image, and the edge-detected image.
Requirements
To run this project, the following Python libraries are required:

OpenCV: For image reading, manipulation, and processing.

bash
Copy code
pip install opencv-python
NumPy: For numerical operations, especially for matrix manipulations.

bash
Copy code
pip install numpy
Matplotlib: For displaying images and visualizations.

bash
Copy code
pip install matplotlib
Project Structure
Classes and Methods
ImageProcessor Class:
Methods:
change_contrast(image_path, contrast_factor=2.0, brightness_factor=50):

Adjusts the contrast and brightness of the image by manipulating the luminance channel in LAB color space.
sharpen_image(contrasted_image, strength=4.0, brightness_factor=5, contrast_factor=1.1):

Applies a sharpening filter to the luminance channel, adjusts brightness, and refines the contrast for clearer visualization.
edge_detection(sharpened_l_channel, threshold=1, brightness=127, contrast=2.0):

Performs edge detection by converting the sharpened image to grayscale, adjusting contrast, applying Gaussian blur, and detecting edges using the Canny algorithm.
Key Operations
Contrast Adjustment:

The image is first converted from BGR (standard image format) to LAB color space. The luminance (L) channel is adjusted to increase contrast and decrease brightness. This enhances visibility by making the bright regions stand out more clearly while darkening other parts.
Sharpening:

Sharpening is applied to enhance the edges and details of the cells. A kernel is used to highlight changes in intensity, which brings out the fine features of the image that may not be clearly visible in the original or contrast-adjusted images.
Edge Detection:

Canny edge detection is applied after converting the sharpened image to grayscale and applying Gaussian blur to remove noise. The resulting edges highlight the boundaries of cells or structures, aiding in further image analysis.
Visualizations
The project displays a set of images in a 4-panel layout using Matplotlib. The images include:
The Original Image.
The Contrast-Adjusted Image.
The Sharpened Image.
The Edge-Detected Image.
Usage
Steps to Run the Project:
Prepare the Image:

Ensure that the target image (e.g., Cells.jpg) is available in the working directory.
Run the Script:

The script will read the image, process it using the ImageProcessor class, and display the original, contrast-adjusted, sharpened, and edge-detected images in a visual layout.
python
Copy code
# Create an instance of the ImageProcessor class.
image_processor = ImageProcessor()

# Change contrast of the Cells.jpg image.
contrast_changed_image = image_processor.change_contrast('/content/Cells.jpg')

# Sharpen the contrast-changed image.
sharpened_l_channel = image_processor.sharpen_image(contrast_changed_image)

# Detect edges in the sharpened image.
edge_detected_image = image_processor.edge_detection(sharpened_l_channel)

# Display the results in a 4-panel layout.
original_image = cv2.imread('/content/Cells.jpg')
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(contrast_changed_image, cv2.COLOR_BGR2RGB))
axes[1].set_title('Contrast Changed Image')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(sharpened_l_channel, cv2.COLOR_BGR2RGB))
axes[2].set_title('Sharpened Image')
axes[2].axis('off')

axes[3].imshow(cv2.cvtColor(edge_detected_image, cv2.COLOR_BGR2RGB))
axes[3].set_title('Edge Detected Image')
axes[3].axis('off')

plt.tight_layout()
plt.show()
Example Output:
Original Image: Displays the original image as read from the disk.
Contrast-Adjusted Image: Shows the image with enhanced contrast, making the bright regions more visible.
Sharpened Image: Displays the sharpened image where edges and fine details are more prominent.
Edge-Detected Image: Displays the edges of the cells, highlighting boundaries and important structures within the image.
Conclusion
This project provides a comprehensive image processing pipeline for cell images. It enhances contrast, sharpens the image to highlight fine details, and uses edge detection to outline the boundaries of cellular structures. This pipeline is essential for improving the clarity of cell images in biomedical research and can be extended for other applications like object detection or texture analysis in images.

Future Enhancements
Real-time Processing: Implement the ability to process images in real-time, either from a video stream or a camera.
Advanced Filtering: Introduce more sophisticated filters such as bilateral filtering or non-local means filtering to reduce noise while preserving important edges.
Feature Extraction: Add functionality for extracting cell features such as size, shape, or intensity for further analysis.
Automated Segmentation: Implement automated cell segmentation for detecting and isolating individual cells in complex images.
