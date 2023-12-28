import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.filters import unsharp_mask
import numpy as np

# Define a function to check the image is blurred or not
def is_blurred(image,threshold=100):

    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur_metric=cv2.Laplacian(gray_image,cv2.CV_64F).var()
    return blur_metric<threshold

# Define a function to deblur the image
def deblur_image(image):
    # Unsharp masking for deblurring 
    deblurred_image = unsharp_mask(image, radius=1.5, amount=1.5)
    return deblurred_image
    

image_path=r"C:\Users\USER\Desktop\Deblurring\nature-blurred.jpeg"
image=cv2.imread(image_path)


if is_blurred(image):
    # If the image is blurred, deblur it
    deblurred_image = deblur_image(image)
    deblurred_image = (deblurred_image * 255).astype(np.uint8)

    grayA = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    deblurred_image_resized = cv2.resize(deblurred_image, (grayA.shape[1], grayA.shape[0]))
    grayB= cv2.cvtColor(deblurred_image_resized, cv2.COLOR_BGR2GRAY)   
    (score, diff) = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))         

    # Display the original, blurred, and deblurred images for comparison
    cv2.imshow("Original", image)
      
    cv2.imshow("Deblurred", deblurred_image)
    cv2.imwrite('Orginal_image.jpeg',image)
    cv2.imwrite('Deblurred Image.jpeg',deblurred_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    # If the image is not blurred, do nothing or display the original image
    cv2.imshow("Original", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
