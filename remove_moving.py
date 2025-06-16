import cv2
import numpy as np

def remove_moving_object(img1_path, img2_path, save_path='result.png'):
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Resize to same size (just in case)
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference
    diff = cv2.absdiff(gray1, gray2)

    # Threshold to get moving regions
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Dilate mask to cover full object
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Create 3-channel mask for color image
    mask_3ch = cv2.merge([mask, mask, mask])

    # Invert mask
    mask_inv = cv2.bitwise_not(mask_3ch)

    # Combine images: remove object in img1 using pixels from img2
    background = cv2.bitwise_and(img2, mask_3ch)
    foreground = cv2.bitwise_and(img1, mask_inv)
    result = cv2.add(background, foreground)

    # Save and show result
    cv2.imwrite(save_path, result)
    cv2.imshow('Original Image 1', img1)
    cv2.imshow('Original Image 2', img2)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
remove_moving_object('img1.jpg', 'img2.jpg')
