"""

©Copyright Code :
Chloé BRICE, INSA Lyon
Evahn LE GAL, ISIMA Clermont INP

Code co-written with Chloé BRICE, with the agreement of the professor, the report and the images will nevertheless be different for each student
Only the code was written and used as a team

Python 3.11.6

"""

import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



#-#-# // Question A-2 : Compute E \\ #-#-#

print("\n### ## # A-2 : Compute E # ## ###\n\n")

# Part 1 : Compute F in the same way as report1

from A2_epipolar_lines import load_points, construct_F, creation_of_img, draw_report1_epipolar_line

list_of_point_F = "resources/report1_F/list_of_points.txt"
points1, points2, colors = load_points(list_of_point_F)
F = construct_F(points1, points2)

# Part 2 : Create images and draw first epipolar lines of report1 in cyan

img1 = Image.open("resources/report1_F/img1.png")
img2 = Image.open("resources/report1_F/img2.png")
ax1, ax2 = creation_of_img(img1, img2)
draw_report1_epipolar_line(F, points1, points2, img1, img2, ax1, ax2)


# Part 3 : recover E and draw the associated epipolar lines

from A2_exploitation_of_E import load_E, compute_new_F, draw_epipolar_line_new_F

name_matrix = input("\n\nGive the name of the file with the matrix E : ")

E = load_E("resources/report1_F/A2_matrix_E/" + name_matrix)
print("\n E loaded = ", E)

K = np.array([[1,0,0],
              [0,1,0],
              [0,0,1]])

new_F = compute_new_F(E,K)
print("\n New F = ", new_F)


nb_elements = input("\nHow many items do you want to process? (choose the same number as for the calculation of E in C++, 5 by default for the 5 point algorithm) : ")
nb_elements = int(nb_elements)
if nb_elements < 5 :
    print("Error, can't use less than 5 points")
    exit()

draw_epipolar_line_new_F(new_F, nb_elements, points1, points2, colors, img1, img2, ax1, ax2)

plt.savefig("resources/output/epipolar_lines_from_E.png")
plt.show()




#-#-# // Question B-2 : Photometric Stereo Method \\ #-#-#

print("\n\n\n### ## # B-2 : Photometric Stereo Method # ## ###\n\n")
from B2_photometric import load_lights, load_imgs_mask, compute_photometric, show_and_display

# Part 1 : Extract lights and Load Images/Mask

list_numbers = []
print("Type the camera numbers (between 0 and 11, need 3 cameras)\n")
for i in range(3):
    ask = int(input("Camera number you want to use : "))
    list_numbers.append(ask)

S_list = load_lights("resources/photometric/lights.txt", list_numbers)
print("\nS_list : \n", S_list)

imgs, mask = load_imgs_mask(list_numbers)


# Part 2 : Compute the photomectric method

img_normal_rgb = compute_photometric(S_list, imgs, mask)


# Part 3 : Display and save
show_and_display(list_numbers, imgs, img_normal_rgb, "resources/output/surface_normals.png")


