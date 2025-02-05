import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image_path = "Media\single_line_wide_ecg.jpg"  # Update this path
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image to binary
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Find contours (assuming continuous lines in the graph)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract data points for each signal
all_signals_points = []
for contour in contours:
    points = []
    for point in contour:
        x, y = point[0]
        points.append((x, y))
    points = sorted(points, key=lambda p: p[0])  # Sort by X-coordinate
    all_signals_points.append(points)

# Normalize and separate data points for plotting
plt.figure(figsize=(10, 5))
for i, signal_points in enumerate(all_signals_points):
    x_points = [p[0] for p in signal_points]
    y_points = [img.shape[0] - p[1] for p in signal_points]  # Flip Y-axis for correct plotting

    # Normalize the data points for better visualization
    x_points = np.array(x_points)
    y_points = np.array(y_points)

    # Plot original signals
    plt.plot(x_points, y_points, label=f"Signal {i + 1}")

plt.legend()
plt.title("Extracted Graph Data Points")
plt.show()

# Plot reconstructed signals
plt.figure(figsize=(10, 5))
for i, signal_points in enumerate(all_signals_points):
    x_points = [p[0] for p in signal_points]
    y_points = [img.shape[0] - p[1] for p in signal_points]
    plt.plot(x_points, y_points, label=f"Reconstructed Signal {i + 1}")

plt.legend()
plt.title("Reconstructed Signals")
plt.show()
