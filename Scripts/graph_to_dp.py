import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def process_graph_image(image_path, json_output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image file not found or unable to read.")
        return

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to highlight lines and points
    _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Detect contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Image dimensions
    height, width = image.shape[:2]

    data_points = []

    # Process each contour to extract data points
    for contour in contours:
        # Get the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Normalize coordinates to graph dimensions (0 to 1 range)
            x_normalized = cX / width
            y_normalized = (height - cY) / height  # Invert Y-axis

            data_points.append({"x": round(x_normalized, 4), "y": round(y_normalized, 4)})

    # Sort points by x-coordinate for better plotting later
    data_points.sort(key=lambda point: point["x"])

    # Write the points to JSON
    with open(json_output_path, 'w') as json_file:
        json.dump(data_points, json_file, indent=4)

    print(f"Data points successfully saved to '{json_output_path}'.")

    # Plotting for visualization
    x_vals = [point["x"] for point in data_points]
    y_vals = [point["y"] for point in data_points]
    plt.scatter(x_vals, y_vals, color='red')
    plt.title("Extracted Data Points")
    plt.xlabel("X-axis (Normalized)")
    plt.ylabel("Y-axis (Normalized)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    image_path = input("Enter the path to the graph image: ")
    json_output_path = input("Enter the path to save the JSON file: ")

    # Ensure output directory exists
    output_dir = os.path.dirname(json_output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_graph_image(image_path, json_output_path)
