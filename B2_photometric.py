import math
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np


def load_lights(name_file, list_numbers):
    file = open(name_file, "r")
    tab = []

    line = file.readline()
    while line :
        tab_line = line[:-1].split(" ")
        tab_temp = []
        tab_temp.append(float(tab_line[0]))
        tab_temp.append(float(tab_line[1]))
        tab_temp.append(float(tab_line[2]))
        tab.append(tab_temp)
        
        line = file.readline()

    file.close()

    S_list = []
    for i in list_numbers :
        S_list.append(tab[i])
    
    return S_list

def load_imgs_mask(list_numbers):
    img_files = []
    for i in list_numbers:
        img_files.append(f"resources/photometric/owl.{i}.png")

    imgs = [Image.open(img_file).convert("RGB") for img_file in img_files]
    for img in imgs:
        if img is None:
            print("Image loading has failed")
            exit()
            
    mask = Image.open("resources/photometric/owl.mask.png").convert("L")
    if mask is None:
        print("Mask loading has failed")
        exit()
    return imgs, mask


# Compute normals with lights on object with the photometric stereo method
def compute_photometric(S_list, imgs, mask):
    # Convert to gray scale and apply mask
    gray_imgs = []
    for img in imgs:
        gray_img = img.convert("L")
        masked = Image.fromarray(np.bitwise_and(np.array(gray_img), np.array(mask)))
        gray_imgs.append(masked)

    # Use the light sources coordinates
    S = np.array(S_list)
    S_t = S.T

    # Transform the gray images list into an array for the computation
    gray_imgs = np.array(gray_imgs)

    h, w = gray_imgs[0].shape[:2]

    # Matrix of all the normal coordinates at each pixel
    img_normal = np.zeros((h, w, 3))

    # Value of luminance at each pixel of the grey images
    I = np.zeros((len(S_list), 3))

    # For each pixel
    for x in range(w):
        for y in range(h):

            # And each of the 3 images
            for i in range(len(gray_imgs)):
                I[i] = gray_imgs[i][y][x]

            # Use the formula to compute N
            S_inv = np.linalg.inv(S_t)
            N = np.dot(S_inv, I).T
            N_norm = np.linalg.norm(N, axis=1)

            rho = N_norm * math.pi

            # Define a luminosity to display normals
            N_gray = N[0] * 0.30 + N[1] * 0.35 + N[2] * 0.35
            N_gray_norm = np.linalg.norm(N_gray)
            if N_gray_norm == 0:
                continue

            # Store the normals' values of the image
            img_normal[y][x] = N_gray / N_gray_norm

    # Convert to rgb for colored representation
    return ((img_normal * 0.5 + 0.5) * 255).astype(np.uint8)


# Display and save results
def show_and_display(list_numbers, imgs, img_normal_rgb, name_save):
    plt.figure(figsize=(30,8))
    plt.subplot(1, 2, 1)
    plt.imshow(imgs[0])
    plt.title(f"Example image with a specific light source (image [{list_numbers[0]}])")
    plt.subplot(1, 2, 2)
    plt.imshow(img_normal_rgb)
    plt.title("Surface Normals of the object")
    plt.savefig(name_save)
    plt.show()
