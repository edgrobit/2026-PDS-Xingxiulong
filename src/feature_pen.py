import cv2
import numpy as np

# Load image
image = cv2.imread("/home/hakim/Desktop/ITU Course stuff/Project in DataScience/2026-PDS-Xingxiulong/data/imgs/PAT_1320_1135_471.png")

# Convert to HSV (better for color detection)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define blue color range (tune if needed)
lower_blue = np.array([80, 13, 13])
upper_blue = np.array([200, 200, 200])

# Create mask for blue pen marks
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Optional: clean up mask (remove noise)
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Inpaint to replace blue marks with surrounding skin
result = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)

# Save result
cv2.imwrite("output.jpg", result)

print("Done! Saved as output.jpg")