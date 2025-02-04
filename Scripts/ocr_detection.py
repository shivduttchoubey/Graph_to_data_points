import cv2
import easyocr

# Load the image
image = cv2.imread('Media\accu_sure_3_param_front.jpeg')

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Perform OCR on the image
result = reader.readtext(image)

# Print the detected text
for res in result:
    print(res[1])

# Display the image
cv2.imshow('Image', image)
cv2.waitKey(0)
# cv2.destroyAllWindows()
