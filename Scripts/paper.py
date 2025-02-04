import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte

def load_and_scan_image(image_path):
    """Step 1 & 2: Load and scan the ECG image"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not load image")
    return img

def crop_roi(image):
    """Step 3: Crop region of interest"""
    # Find non-zero points to determine ROI
    coords = cv2.findNonZero(image)
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]

def binarize_image(image):
    """Step 4: Image Binarization"""
    # Apply adaptive thresholding for better binarization
    binary = cv2.adaptiveThreshold(
        image, 
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    return binary

def extract_features(image):
    """Step 5: Gradient-based Feature Extraction"""
    # Calculate gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    return img_as_ubyte(gradient_magnitude / gradient_magnitude.max())

def remove_noise(image):
    """Step 6: Noise Rejection"""
    # Apply morphological operations to remove noise
    kernel = np.ones((2,2), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing

def thin_image(image):
    """Step 7: Image Thinning"""
    # Convert to binary format required by skeletonize
    binary = image > 0
    skeleton = skeletonize(binary)
    return img_as_ubyte(skeleton)

def detect_edges(image):
    """Step 8: Edge Detection"""
    edges = cv2.Canny(image, 50, 150)
    return edges

def convert_to_vector(image):
    """Step 9: Pixel to vector conversion"""
    # Find all non-zero points
    points = np.column_stack(np.where(image > 0))
    
    # Convert to time-amplitude coordinates
    y_values = points[:, 0]
    x_values = points[:, 1]
    
    # Normalize coordinates
    x_norm = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
    y_norm = (np.max(y_values) - y_values) / (np.max(y_values) - np.min(y_values))
    
    # Remove duplicates and sort
    unique_x, indices = np.unique(x_norm, return_index=True)
    unique_y = y_norm[indices]
    
    sorted_indices = np.argsort(unique_x)
    return unique_x[sorted_indices], unique_y[sorted_indices]

def process_ecg_image(image_path, debug=False):
    """Complete ECG processing pipeline"""
    # Load and scan image
    img = load_and_scan_image(image_path)
    
    # Crop ROI
    img_cropped = crop_roi(img)
    
    # Binarize
    img_binary = binarize_image(img_cropped)
    
    # Extract features
    img_features = extract_features(img_binary)
    
    # Remove noise
    img_denoised = remove_noise(img_features)
    
    # Thin image
    img_thinned = thin_image(img_denoised)
    
    # Detect edges
    img_edges = detect_edges(img_thinned)
    
    # Convert to vector
    x_values, y_values = convert_to_vector(img_edges)
    
    if debug:
        # Plot intermediate steps
        plt.figure(figsize=(15, 10))
        images = [img, img_cropped, img_binary, img_features, 
                 img_denoised, img_thinned, img_edges]
        titles = ['Original', 'Cropped', 'Binarized', 'Features', 
                 'Denoised', 'Thinned', 'Edges']
        
        for i, (image, title) in enumerate(zip(images, titles)):
            plt.subplot(3, 3, i+1)
            plt.imshow(image, cmap='gray')
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Interpolate for smooth curve
    f = interp1d(x_values, y_values, kind='cubic')
    x_new = np.linspace(0, 1, 500)
    y_new = f(x_new)
    
    # Plot final result
    plt.figure(figsize=(15, 5))
    plt.plot(x_values, y_values, 'ro', label="Extracted Points", markersize=2)
    plt.plot(x_new, y_new, 'b-', label="Reconstructed Signal")
    plt.xlabel("Time (Normalized)")
    plt.ylabel("Amplitude (Normalized)")
    plt.title("Digitized ECG Signal")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return x_new, y_new

# Usage example
if __name__ == "__main__":
    image_path = "bathini_image.jpg"  # Replace with your image path
    x_values, y_values = process_ecg_image(image_path, debug=True)
    
    # Save digitized data
    data = np.column_stack((x_values, y_values))
    np.savetxt("digitized_ecg.csv", data, delimiter=",", 
               header="time,amplitude", comments="")