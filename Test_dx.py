from collections import Counter
import os
import numpy as np
import cv2
import time
from ultralytics import YOLO
from PIL import Image, ImageDraw

import DepthPoints 

import speech_recognition as sr
import sounddevice
import re


def normalize(value):
    x_min, x_max = -1.0, 1.0
    y_min, y_max = -1.0, 1.0
    x = x_min + (value[0] / 640) * (x_max - x_min)
    y = y_min + (1.0 - value[1] / 480) * (y_max - y_min)

    return x,y

def word_to_number(word):
    word = word.lower()
    numbers_dict = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10
    }
    return numbers_dict.get(word)

def extract_info(text):
    names = []
    text_words = text.split()
    for i in range(len(text_words)):
        number = word_to_number(text_words[i].lower())
        if number is not None:
            text_words[i] = str(number)
    text = " ".join(text_words)
    
    # Dictionary of names and associated regular expressions
    names_dict = {
        'person': 0,
        'bicycle': 1,
        'car': 2,
        'motorcycle': 3,
        'airplane': 4,
        'bus': 5,
        'train': 6,
        'truck': 7,
        'boat': 8,
        'traffic light': 9,
        'fire hydrant': 10,
        'stop sign': 11,
        'parking meter': 12,
        'bench': 13,
        'bird': 14,
        'cat': 15,
        'dog': 16,
        'horse': 17,
        'sheep': 18,
        'cow': 19,
        'elephant': 20,
        'bear': 21,
        'zebra': 22,
        'giraffe': 23,
        'backpack': 24,
        'umbrella': 25,
        'handbag': 26,
        'tie': 27,
        'suitcase': 28,
        'frisbee': 29,
        'skis': 30,
        'snowboard': 31,
        'sports ball': 32,
        'kite': 33,
        'baseball bat': 34,
        'baseball glove': 35,
        'skateboard': 36,
        'surfboard': 37,
        'tennis racket': 38,
        'bottle': 39,
        'wine glass': 40,
        'cup': 41,
        'fork': 42,
        'knife': 43,
        'spoon': 44,
        'bowl': 45,
        'banana': 46,
        'apple': 47,
        'sandwich': 48,
        'orange': 49,
        'broccoli': 50,
        'carrot': 51,
        'hot dog': 52,
        'pizza': 53,
        'donut': 54,
        'cake': 55,
        'chair': 56,
        'couch': 57,
        'potted plant': 58,
        'bed': 59,
        'dining table': 60,
        'toilet': 61,
        'tv': 62,
        'laptop': 63,
        'Mouse': 64,
        'remote': 65,
        'keyboard': 66,
        'cell phone': 67,
        'microwave': 68,
        'oven': 69,
        'toaster': 70,
        'sink': 71,
        'refrigerator': 72,
        'book': 73,
        'clock': 74,
        'vase': 75,
        'scissors': 76,
        'teddy bear': 77,
        'hair drier': 78,
        'toothbrush': 79
    }
    
    # Generate the regular expression to find names
    names_regex = "|".join(names_dict.keys())
    
    # Find all names in the text using the regular expression
    names_found = re.findall(names_regex, text)
    
    numbers = re.findall(r'\d+', text)
    
    min_length = min(len(numbers), len(names_found))
    
    for i in range(min_length):
        names.extend([names_dict[names_found[i]]] * int(numbers[i]))
    
    if not names:
        names = False
    
    return names



def main():

    # Dictionary to map from numbers to words
    number_words = {
        1: "One",
        2: "Two",
        3: "Three",
        4: "Four",
        5: "Five",
        6: "Six",
        7: "Seven",
        8: "Eight",
        9: "Nine",
        10: "Ten"
    }

    names_dict = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        4: 'airplane',
        5: 'bus',
        6: 'train',
        7: 'truck',
        8: 'boat',
        9: 'traffic light',
        10: 'fire hydrant',
        11: 'stop sign',
        12: 'parking meter',
        13: 'bench',
        14: 'bird',
        15: 'cat',
        16: 'dog',
        17: 'horse',
        18: 'sheep',
        19: 'cow',
        20: 'elephant',
        21: 'bear',
        22: 'zebra',
        23: 'giraffe',
        24: 'backpack',
        25: 'umbrella',
        26: 'handbag',
        27: 'tie',
        28: 'suitcase',
        29: 'frisbee',
        30: 'skis',
        31: 'snowboard',
        32: 'sports ball',
        33: 'kite',
        34: 'baseball bat',
        35: 'baseball glove',
        36: 'skateboard',
        37: 'surfboard',
        38: 'tennis racket',
        39: 'bottle',
        40: 'wine glass',
        41: 'cup',
        42: 'fork',
        43: 'knife',
        44: 'spoon',
        45: 'bowl',
        46: 'banana',
        47: 'apple',
        48: 'sandwich',
        49: 'orange',
        50: 'broccoli',
        51: 'carrot',
        52: 'hot dog',
        53: 'pizza',
        54: 'donut',
        55: 'cake',
        56: 'chair',
        57: 'couch',
        58: 'potted plant',
        59: 'bed',
        60: 'dining table',
        61: 'toilet',
        62: 'tv',
        63: 'laptop',
        64: 'mouse',
        65: 'remote',
        66: 'keyboard',
        67: 'cell phone',
        68: 'microwave',
        69: 'oven',
        70: 'toaster',
        71: 'sink',
        72: 'refrigerator',
        73: 'book',
        74: 'clock',
        75: 'vase',
        76: 'scissors',
        77: 'teddy bear',
        78: 'hair drier',
        79: 'toothbrush'
    }

    #* SPEECH RECOGNITION
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

    except Exception as ex:

        print(ex)


    sr.Microphone.list_microphone_names()

    extracted_info = extract_info(text)
    print("Extracted information:", extracted_info)


    #* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Load a model
    model = YOLO('yolov8x-seg.pt')  # load an official model

    # Initialize camera
    cap = cv2.VideoCapture(4)  # El argumento 0 selecciona la camara predeterminada

    # Wait 1 second
    time.sleep(1)

    # Get a frame
    ret, frame = cap.read()

    # Verify the frame
    if ret:
        print("Have image")
        results = model(frame, stream = True, conf = 0.5)
        cv2.imwrite('frame.jpg', frame)
        # Analize results
        for r in results:
            names_cls = r.names
            clases = r.boxes.cls.cpu().numpy() 
            print(clases)
            coordenadas = r.masks.xy
            if extracted_info != False:
                clases_old = extracted_info
                names_cls =  names_dict
            else: 
                clases_old = clases

            element_count = Counter(clases_old)  # Count of number of each object

            result_strings = []

            for element, count in element_count.items():
                result_strings.append(str(number_words[count])+ " " + str(names_cls[element]))

            # Concatenate the strings
            if len(result_strings) > 1:
                result = ", ".join(result_strings[:-1]) + " and " + result_strings[-1]
            else:
                result = result_strings[0]

            print(result)

            # Prompt generation
            prompt1 = result + ", top view of a wooden table, photography, ordered, best quality, photorealistic, hyperdetailed, realistic, 4k"
            prompt2 = "Top view of a white table with " + result + ", photography, ordered, best quality,  photorealistic, hyperdetailed, realistic, 4k"
            prompt3 = result + ", view from above of a white table, photography, ordered, best quality,  photorealistic, hyperdetailed, realistic, 4k"
            prompt4 = "View from above of a wooden table with " + result + ", photography, ordered, best quality,  photorealistic, hyperdetailed, realistic, 4k"
            print("Prompt 1: " + prompt1)
            print("Prompt 2: " + prompt2)
            print("Prompt 3: " + prompt3)
            print("Prompt 4: " + prompt4)

            # Initialize variables
            center_old = []
            size_old = []
            angle_old = []
            data_A = []
            masks=[]
            vector_d=[]
            clases_new =[]

            for i in range(len(clases_old)):
                # Image size
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                width, height = im_pil.size

                # Create blank image
                binary_image = Image.new("L", (width, height), 0)

                # Create object
                draw = ImageDraw.Draw(binary_image)

                # Reorder coordinates
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
                binary_image.save("maskT_" + str(i) + names_cls[clases[i]] + ".png")   

                cv2_mask = np.array(binary_image)

                ###!     OBB Extraction       ###

                # Resize the image to 640x480
                img = cv2.resize(frame, (640, 480))

                # Resize the binary_image to 640x480
                cv2_mask = cv2.resize(cv2_mask, (640, 480))
                masks.append(cv2_mask) #*Almacenamos las mascaras

                # Find the contours of the objects in the binary_image
                contours, _ = cv2.findContours(cv2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Find the largest contour
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)

                    # Fit a rotated rectangle to the largest contour
                    rect = cv2.minAreaRect(largest_contour)
                    # Extract vertices
                    box_points = cv2.boxPoints(rect).astype(int)

                    # Calculate the angle
                    if rect[1][0]<rect[1][1]:
                        angle = 90 - rect[2]
                        size = [rect[1][1],rect[1][0]]

                        x1,y1 = normalize(value=box_points[3])
                        x2,y2 = normalize(value=box_points[2])

                        dx = (x2-x1)
                        dy = (y2-y1)
                        vector_d.append([dx,dy])
                    else:
                        angle = 180 - rect[2]
                        size = [rect[1][0],rect[1][1]]

                        x1,y1 = normalize(value=box_points[3])
                        x2,y2 = normalize(value=box_points[0])
                        dx = (x2-x1)
                        dy = (y2-y1)
                        vector_d.append([dx,dy])

                    Point_A = [dx, dy]

                    # Draw the rotated rectangle on the original image
                    cv2.drawContours(img, [box_points], 0, (0, 255, 0), 2)  # Draw the green rectangle
                    #print(str(rect[0][0]) + ' ' + str(rect[0][1]) + ' ' + str(rect[1][0]) + ' ' + str(rect[1][1]) + ' ' + '(Angle)' + ' ' + str(angle) + '\n' )
                    center_old.append(rect[0])
                    #size_old.append(rect[1])
                    size_old.append(size)
                    angle_old.append(angle)
                    data_A.append(Point_A)
                    #print(data_A)
                    prompts = (prompt1, prompt2, prompt3, prompt4)

                    # Display the image with the oriented bounding box
                    cv2.imshow('Oriented Bounding Box', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    # Close binary image
                    binary_image.close() 
                    
                else:
                    print("No contours found in the binary_image.")
                    binary_image.close()
                clases[index]=10 #! Huge value to avoid repeating the same object for each class

    # Free the camera and CV windows
    cap.release()
    cv2.destroyAllWindows()

    #* Obtain depth information
    Z_Points=[]
    Points=[]
    Z_point = 0
    for i in range(len(center_old)):
        while Z_point == 0: 
            Z_point, Point = DepthPoints.main(int(center_old[i][0]), int(center_old[i][1]))
            #Z_point=0.5
        if clases_old[i] ==41:
            Z_point = Z_point -0.1
            Point[2]= Point[2] - 0.1
        Z_Points.append(Z_point)
        Points.append(Point)
        Z_point=0
    print("Z_Points: {}".format(Z_Points))
    print("Points: {}".format(Points))
    
    return center_old, size_old, angle_old, prompts, clases_old, data_A,img,masks, vector_d, Z_Points,Points

if __name__=="__main__":
    main()