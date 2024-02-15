import numpy as np
import cv2 as cv

def main(img, mask):
    # Create a blank image of the same size as the original images.
    result_image = np.zeros_like(mask[0])
    background = cv.imread('background.png')
    background = cv.resize(background, (640, 480))

    # Sum images
    for i in range(len(mask)):
        result_image = cv.add(result_image, mask[i])

    # Show result
    cv.imshow('Result Image', result_image)

    # Create inverse mask for objects
    inverse_mask = cv.bitwise_not(result_image)

    # Apply the inverse mask to the original image to obtain the white background.
    white_background = cv.bitwise_and(background, background, mask=inverse_mask)

    # Adding the objects from the original image to the white background
    result_image = cv.bitwise_or(white_background, cv.bitwise_and(img, img, mask=result_image))

    # Show the final result of inpainting process
    cv.imshow('Final Result', result_image)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main(img, mask)
