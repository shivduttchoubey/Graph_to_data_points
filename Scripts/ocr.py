import cv2
import pytesseract

# Simple OCR function to extract numbers from an image
def extract_numbers(image_path: str):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image file not found.")
        return []

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract text from the image
    text = pytesseract.image_to_string(gray, config='--psm 6')

    # Extract numbers only
    numbers = [num for num in text.split() if num.isdigit()]

    return numbers

# Example usage
image_path = "download (3).jpeg"  # Replace with your image path
print("Extracted Numbers:", extract_numbers(image_path))
