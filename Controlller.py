import Test_dx
import Stable_img
import Stable2YOLO
import numpy as np
import matplotlib.pyplot as plt


def transitions(n):
    # Create an original list with n elements
    original_list = list(range(1, n + 1))

    # Create a new list to store the results
    modified_list = []

    # Add a zero at the beginning of the list
    modified_list.append(0)

    # Iterate over the original list
    for value in original_list:
        # Add the current value to the modified list
        modified_list.append(value)
        # Add a zero after each value, except at the end
        if value != original_list[-1]:
            modified_list.append(0)

    # Add a zero at the end of the list
    modified_list.append(0)

    # Print the original list and the modified list
    print("Original List:", original_list)
    print("Modified List:", modified_list)
    return modified_list



#* YOLO2Stable-> we obtain the values of what camera see
center_old, size_old, angle_old, prompts, clases_old, data_A,img,masks,vector_d,Z_Points,Points = Test_dx.main()
print(center_old)
print("Angles: {}".format(angle_old))
print(len(masks))
C_img = []
UD =[]
mg =[]
for i in range(len(masks)):
    #* If up=1, it is upwards
    mango=False
    up=1
    cx=center_old[i][0]
    cy=center_old[i][1]
    C_img.append([cx,cy])
    UD.append(up)
    mg.append(mango)
    
print("*******************")
#* Generate the .txt with the prompts
Bot_prompts = "Bot_prompts.txt"
with open (Bot_prompts, 'w') as archivo:
    for i in range(len(prompts)):
        fila = prompts[i]
        archivo.write(fila +'\n')

#* Execute Stable Baseline
Stable_img.main() #! Comment to have a real time execution

#* Stable2YOLO-> select the correct image with YOLO
center_f,size_f,angle_f,img_stable,masks_stable, vector_df, clases_new = Stable2YOLO.main(clases_old,size_old) 
#? Añadir profundidad
Z_Points_f = []
Points_f = []
for i in range(len(center_f)):
    Z_point = Z_Points[i] # Asumir misma profundidad que escena inicial
    Point =[center_f[i][0],center_f[i][1],Z_point]
    Z_Points_f.append(Z_point)
    Points_f.append(Point)

print("*******************")
#* Correct the angle depending of the direction
for i in range(len(UD)):
    if UD[i] == False:
        angle_old[i] = angle_old[i] + 180
    else:
        angle_old[i]=angle_old[i]

#? Export data
print("ACTUAL SCENE DATA")
print("Points: {}".format(Points))
print("Orientations: {}".format(angle_old))
print("Classes: {}".format(clases_old))
print("*******************")
print("NEW SCENE DATA")
print("Points: {}".format(Points_f))
print("Orientations: {}".format(angle_f))
print("Classes: {}".format(clases_new))
print("*******************")