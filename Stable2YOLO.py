from collections import Counter
import os
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw
import time

import speech_recognition as sr
import sounddevice
import re

import InPaint

def normalize(value):
    x_min, x_max = -1.0, 1.0
    y_min, y_max = -1.0, 1.0
    x = x_min + (value[0] / 640) * (x_max - x_min)
    y = y_min + (1.0 - value[1] / 480) * (y_max - y_min)

    return x,y

def extract_info(text):
    text = text.lower()  # Convert to lowercases
    if 'yes' in text:
        return True
    elif 'no' in text:
        return False
    else:
        return None

def speech_to_text():
    #* SPEECH RECOGNITION
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("DO YOU LIKE THIS SOLUTION?")
    print("Wait a few seconds")
    time.sleep(5)
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Adjusting noise ")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Recording for 10 seconds")
        recorded_audio = recognizer.listen(source, timeout=10)
        print("Done recording")

    try:
        print("Recognizing the text")
        text = recognizer.recognize_google(
                recorded_audio, 
                language="en-US"
            )

        print("Decoded Text : {}".format(text))

        extracted_info = extract_info(text)
        print("Extracted information:", extracted_info)

        return extracted_info

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

def checkExact(results,masks,img_path,clases_old):
    print("CheckExcat()")
    masks =[]
    # Extract clasess and coordinates
    for r in results:
        names_cls = r.names
        coordenadas = r.masks.xy   
        clases = r.boxes.cls.cpu().numpy()
        print(clases)

        # Convert arrays to sets and counters
        set1 = set(clases_old)
        set2 = set(clases)
        count1 = Counter(clases_old)
        count2 = Counter(clases)

        # Check similarity between the initial and the final objects
        if set1 == set2 and count1 == count2:
            print("The objects match the initial detection.")

            # Initialize variables
            center_f = []
            size_f = []
            angle_f = []
            vector_df = []
            clases_new =[]

            for i in range(len(coordenadas)):
                # Image size
                img = Image.open(img_path)
                width, height = img.size

                # Create blank image
                binary_image = Image.new("L", (width, height), 0)

                # Create object
                draw = ImageDraw.Draw(binary_image)

                #! Reorder objects
                print("Clase old:{}".format(clases_old[i]))
                index = np.where(clases == clases_old[i])[0]
                index = index[0]
                print("Index: {}".format(index))
                points_float = coordenadas[index]
                clases_new.append(clases[index])


                # Convert to integer
                points_int = [(int(round(x)), int(round(y)) ) for x, y in points_float]

                # Draw the polygon in white
                draw.polygon(points_int, outline=0, fill=255)

                # Save binary image
                binary_image.save("mask_" + str(i) + names_cls[clases_new[i]] + ".png")   

                cv2_mask = np.array(binary_image)

                ###*     Extraction OBB   ###

                # Load the image in OpenCV format
                img = cv2.imread(img_path)

                # Resize the image to 640x480
                img = cv2.resize(img, (640, 480))

                # Resize the binary_image to 640x480
                cv2_mask = cv2.resize(cv2_mask, (640, 480))
                masks.append(cv2_mask)

                # Find the contours of the objects in the binary_image
                contours, _ = cv2.findContours(cv2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Find the largest contour
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)

                    # Fit a rotated rectangle to the largest contour
                    rect = cv2.minAreaRect(largest_contour)
                    box_points = cv2.boxPoints(rect).astype(int)

                    # Calculate the angle
                    if rect[1][0]<rect[1][1]:
                        angle = 90 - rect[2] 
                        size = [rect[1][1],rect[1][0]]

                        x1,y1 = normalize(value=box_points[3])
                        x2,y2 = normalize(value=box_points[2])

                        dx = (x2-x1)
                        dy = (y2-y1)
                        vector_df.append([dx,dy])
                    else:
                        angle = 180 - rect[2]
                        size = [rect[1][0],rect[1][1]]

                        x1,y1 = normalize(value=box_points[3])
                        x2,y2 = normalize(value=box_points[0])
                        dx = (x2-x1)
                        dy = (y2-y1)
                        vector_df.append([dx,dy])

                    # Draw the rotated rectangle on the original image
                    cv2.drawContours(img, [box_points], 0, (0, 255, 0), 2)  # Draw the green rectangle
                    print(str(rect[0][0]) + ' ' + str(rect[0][1]) + ' ' + str(rect[1][0]) + ' ' + str(rect[1][1]) + ' ' + '(Angle)' + ' ' + str(angle) + '\n' )

                    # Display the image with the oriented bounding box
                    cv2.imshow('Oriented Bounding Box', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    # Save the variables 
                    center_f.append(rect[0])
                    size_f.append(size)
                    angle_f.append(angle)

                    # Close binary image
                    binary_image.close()
                    clases[index]=10    #! Huge value to avoid repeating the same object for each class
                    
                    
                    
                else:
                    print("No contours found in the binary_image.")
                    binary_image.close()
                    return None, None, None,None,None, None, None
            return center_f,size_f,angle_f,img,masks, vector_df, clases_new
        else:
            print("The objects do NOT match the initial detection.")
            return None, None, None,None,None, None, None


def checkAprox(results,masks,img_path,clases_old,cont):
    print("CheckAprox()")
    masks =[]
    # Extract clasess and coordinates
    for r in results:
        names_cls = r.names
        coordenadas = r.masks.xy    
        clases = r.boxes.cls.cpu().numpy() 
        print(clases)

        # Convert arrays to sets
        set1 = set(clases_old)
        set2 = set(clases)

        # Check similarity between the initial and the final objects
        print(set1==set2)
        print(len(clases_old))
        print(cont)
        print(len(clases_old)+cont == len(clases))
        if set1 == set2 and len(clases_old)+cont == len(clases):
            print("APPROXIMATE:Objects match the initial detection.")
            # Initialize variables
            center_f = []
            size_f = []
            angle_f = []
            vector_df = []
            clases_new = []

            for i in range(len(clases_old)):
                print("Iteration: {}".format(i))
                # Image size
                img = Image.open(img_path)
                width, height = img.size

                # Create blank image
                binary_image = Image.new("L", (width, height), 0)

                # Create object
                draw = ImageDraw.Draw(binary_image)

                #! Reorder objects
                print("Clase old:{}".format(clases_old[i]))
                print(len(coordenadas))
                index = np.where(clases == clases_old[i])[0]
                index = index[0]
                print("Index: {}".format(index))
                points_float = coordenadas[index]
                clases_new.append(clases[index])

                # Convert to integer
                points_int = [(int(round(x)), int(round(y)) ) for x, y in points_float]

                # Draw the polygon in white
                draw.polygon(points_int, outline=0, fill=255)

                # Save binary image
                binary_image.save("mask_" + str(i) + names_cls[clases_new[i]] + ".png")
                


                cv2_mask = np.array(binary_image)
                
                ###*     Extraction OBB   ###

                # Load the image in OpenCV format
                img = cv2.imread(img_path)

                # Resize the image to 640x480
                img = cv2.resize(img, (640, 480))

                # Resize the binary_image to 640x480
                cv2_mask = cv2.resize(cv2_mask, (640, 480))
                masks.append(cv2_mask)

                # Find the contours of the objects in the binary_image
                contours, _ = cv2.findContours(cv2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Find the largest contour
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)

                    # Fit a rotated rectangle to the largest contour
                    rect = cv2.minAreaRect(largest_contour)
                    box_points = cv2.boxPoints(rect).astype(int)

                    # Calculate the angle
                    if rect[1][0]<rect[1][1]:
                        angle = 90 - rect[2] 
                        size = [rect[1][1],rect[1][0]]

                        x1,y1 = normalize(value=box_points[3])
                        x2,y2 = normalize(value=box_points[2])

                        dx = (x2-x1)
                        dy = (y2-y1)
                        vector_df.append([dx,dy])
                    else:
                        angle = 180 - rect[2] 
                        size = [rect[1][0],rect[1][1]]

                        x1,y1 = normalize(value=box_points[3])
                        x2,y2 = normalize(value=box_points[0])
                        dx = (x2-x1)
                        dy = (y2-y1)
                        vector_df.append([dx,dy])

                    # Draw the rotated rectangle on the original image
                    cv2.drawContours(img, [box_points], 0, (0, 255, 0), 2)  # Draw the green rectangle
                    print(str(rect[0][0]) + ' ' + str(rect[0][1]) + ' ' + str(rect[1][0]) + ' ' + str(rect[1][1]) + ' ' + '(Angle)' + ' ' + str(angle) + '\n' )

                    # Display the image with the oriented bounding box
                    cv2.imshow('Oriented Bounding Box', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    # Save the variables
                    center_f.append(rect[0])
                    size_f.append(size)
                    angle_f.append(angle)

                    # Close binary image
                    binary_image.close()
                    clases[index]=10
                else:
                    print("APROXIMADO: No se han encontrado contornos en la imagen_binaria.")
                    binary_image.close()

                    return None, None, None,None,None, None, None
                clases[index]=10  #! Huge value to avoid repeating the same object for each class
            return center_f,size_f,angle_f,img,masks, vector_df, clases_new
        else:
            print("Condition APROX is not met")
            return None, None, None,None,None, None, None

def main(clases_old, size_old):
    print("FIND: {}".format(clases_old))
    # Load a model
    model = YOLO('yolov8x-seg.pt')  # load an official model

    # Image directory
    #directorio = 'Stable_images/' 
    directorio = 'Pruebas_buenas/' #! Modified for the test

    masks =[]
    newSolution = False

    # Iterate through the images
    for img_file in os.listdir(directorio):
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
        #if img_file.endswith('.jpg'):
            img_path = os.path.join(directorio, img_file)
            # Perform instance segmenation
            results = model(img_path, stream=True, conf=0.45)
            # Check for exact solution
            center_f,size_f,angle_f,img,masks,vector_df, clases_new = checkExact(results,masks,img_path,clases_old)
            if center_f!=None:
                # Inpainting process
                InPaint.main(img,masks)
                # Ask the user
                newSolution = speech_to_text()
                if newSolution == True:
                    break
                else: 
                        print("REPEAT YOUR ANSWER CLEARLY")
                        print("Wait a few seconds")
                        # Ask the user again
                        newSolution = speech_to_text()
                        if newSolution == True:
                            break
    cont=1
    while center_f==None:
        for img_file in os.listdir(directorio):
            if img_file.endswith('.jpg') or img_file.endswith('.png'):
            #if img_file.endswith('.jpg'):
                img_path = os.path.join(directorio, img_file)
                # Perform instance segmenation
                results = model(img_path, stream=True, conf=0.45)
                center_f,size_f,angle_f,img,masks,vector_df, clases_new = checkAprox(results,masks,img_path,clases_old,cont)
                if center_f!=None:
                    # Inpainting process
                    InPaint.main(img,masks)
                    # Ask the user
                    newSolution = speech_to_text()
                    if newSolution == True:
                        break
                    else: 
                        print("REPEAT YOUR ANSWER CLEARLY")
                        print("Wait a few seconds")
                        # Ask the user again
                        newSolution = speech_to_text()
                        if newSolution == True:
                            break
        cont=cont+1
    
    # Verify the coincidence
    if len(size_old) != len(size_f):
        print("Different scenes.")
    else:
    # Refine results and save them
        w = []
        h = []

        for i in range(len(size_old)):
            w.append(size_f[i][0] / size_old[i][0])
            h.append(size_f[i][1] / size_old[i][1])
        
        size_n=[]
        for i in range(len(size_f)):
            print(len(size_f))
            print(len(w))
            print(size_f[i][0] * w[i])
            size_n.append([size_f[i][0] / w[i], size_f[i][1] / h[i]])
            
        print(w)
        print(h)
        print(size_f)
        print(size_old)
        print(size_n)
        print(clases_new)
    
    return center_f,size_f,angle_f,img,masks,vector_df, clases_new

if __name__=="__main__":
    main(clases_old = clases_old, size_old = size_old)

