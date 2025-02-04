import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import os

# Function to plot data points back into an image with continuous lines
def create_graph_image_from_json(json_path, output_image_path, image_size=(600, 400)):
    # Read the data points from the JSON file
    try:
        with open(json_path, 'r') as json_file:
            data_points = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading JSON file: {e}")
        return

    # Create a blank white image
    width, height = image_size
    graph_image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw grid lines for better visualization
    for x in range(0, width, width // 10):
        cv2.line(graph_image, (x, 0), (x, height), (220, 220, 220), 1)
    for y in range(0, height, height // 10):
        cv2.line(graph_image, (0, y), (width, y), (220, 220, 220), 1)

    # Sort points by x-coordinate to draw continuous lines
    data_points.sort(key=lambda point: point['x'])

    # Scale points to image coordinates and store for plotting
    points = [(int(point['x'] * width), int((1 - point['y']) * height)) for point in data_points]

    # Draw continuous lines connecting the points
    for i in range(1, len(points)):
        cv2.line(graph_image, points[i - 1], points[i], (0, 0, 255), 2)

    # Save the image
    cv2.imwrite(output_image_path, graph_image)
    print(f"Graph image successfully saved at '{output_image_path}'.")

    # Show the plot for verification
    plt.imshow(cv2.cvtColor(graph_image, cv2.COLOR_BGR2RGB))
    plt.title("Reconstructed Graph with Continuous Lines")
    plt.axis("on")
    plt.show()

if __name__ == "__main__":
    json_path = input("Enter the path to the JSON file containing data points: ")
    output_image_path = input("Enter the output image path to save the graph: ")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    create_graph_image_from_json(json_path, output_image_path)
