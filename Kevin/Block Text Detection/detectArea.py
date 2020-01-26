# USING ANS FROM https://stackoverflow.com/questions/37771263/detect-text-area-in-an-image-using-python-and-opencv
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

global path, mask_path, working_path, out_path

path = "..\\original_images\\"
mask_path = "..\\masked_images\\"
working_path = "..\\working_images\\"
out_path = "..\\cropped_images\\"

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return (0) # or (0,0,0,0) ?
  return (1)

def maskImages():
    files = []
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            files += [file]

    for n, fName in enumerate(files):
        #load the image.
        print("Masking image: {}".format(fName))
        img = cv2.imread(path+fName)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #blur the image.
        img_hsv_blurred = cv2.GaussianBlur(img_hsv, (19, 19), cv2.BORDER_DEFAULT)
        #create a mask for the texts
        mask_text = cv2.inRange(img_hsv, (0, 0, 0), (180, 80, 60))
        mask_images = cv2.inRange(img_hsv_blurred, (0, 0, 60), (180, 80, 255))
        final_mask = cv2.bitwise_and(mask_text, mask_images)

        cv2.imwrite(mask_path+fName, final_mask)

        #convert to hsv color space.
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(img_hsv, (0, 0, 0), (180, 80, 60))
        blank_image = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
        masked_img = cv2.bitwise_and(mask1, blank_image)

        cv2.imwrite(mask_path+fName, masked_img)

def gridWhite():
    files = []
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            files += [file]
    for fName in files:
        print("Splicing Masked Image: {}".format(fName))
        fNameWJ = fName.split(".jpg")
        fNameWJ = fNameWJ[0]
        img = cv2.imread(path+fName)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        ROI_number = 0
        
        #mask_text = cv2.inRange(img_hsv, (0, 0, 255), (255, 0, 255))
        edges = cv2.Canny(img, 100, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
        dilate = cv2.dilate(edges, kernel, iterations=5)

        cnts = cv2.findContours(dilate, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            area = cv2.contourArea(c)
            if area > 10000:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(img_hsv, (x, y), (x + w, y + h), (36,255,12), 2)
                ROI = img[y:y+h, x:x+w]
                try:
                    cv2.imwrite(out_path+fNameWJ+'ROI_{}.jpg'.format(ROI_number), ROI)
                except:
                    print("Could not Write ROI: {}".format(ROI_number))
                ROI_number += 1

        cv2.imwrite(working_path+fNameWJ+'edges.jpg', edges)
        cv2.imwrite(working_path+fNameWJ+'dilate.jpg', dilate)
        cv2.imwrite(working_path+fNameWJ+'final.jpg', img_hsv)

def show(desc, img):
    scale_percent = 20 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    cv2.imshow(desc, resized)

def spliceImage():
    files = []
    for file in os.listdir(mask_path):
        if file.endswith(".jpg"):
            files += [file]

    for n, fName in enumerate(files):
        print("Splicing Masked Image: {}".format(fName))
        fNameWJ = fName.split(".jpg")
        fNameWJ = fNameWJ[0]
        mimage = cv2.imread(mask_path+fName)
        oimage = cv2.imread(path + fName)

        gray = cv2.cvtColor(mimage, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9,9), 0)
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,30)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
        dilate = cv2.dilate(thresh, kernel, iterations=5)

        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        ROI_number = 0

        boxList = []

        for c in cnts:
            area = cv2.contourArea(c)
            if area > 10000:
                x,y,w,h = cv2.boundingRect(c)
                boxList += [[x,y,w,h]]

        boxListCopy = list(boxList)
        removedList = []
        for i, boxa in enumerate(boxList):
            for boxb in boxList[i+1:]:
                # [boxa[0], boxa[1], boxa[0]+boxa[2], boxa[1]+boxa[3]], [boxb[0], boxb[1], boxb[0]+boxb[2], boxb[1]+boxb[3]]
                if intersection(boxa, boxb):
                    if (boxa[2]*boxa[3] > boxb[2]*boxb[3]):
                        if boxb not in removedList:
                            boxListCopy.remove(boxb)
                            removedList += [boxb]
                        else:
                            print("Failed to remove: {}".format(boxb))
                    elif (boxa[2]*boxa[3] < boxb[2]*boxb[3]):
                        if boxa not in removedList:
                            boxListCopy.remove(boxa)
                            removedList += [boxa]
                        else:
                            print("Failed to remove: {}".format(boxa))
        boxList = list(boxListCopy)

        for i, box in enumerate(boxList):
            [x, y, w, h] = box

            offset = 200
            y -= offset
            h += offset

            cv2.rectangle(oimage, (x, y), (x + w, y + h), (36,255,12), 3)
            ROI = oimage[y:y+h, x:x+w]
            try:
                cv2.imwrite(out_path+fNameWJ+'ROI_{}.jpg'.format(ROI_number), ROI)
            except:
                print("Could not Write ROI: {}".format(box))
            ROI_number += 1

        cv2.imwrite(working_path+fNameWJ+'thresh.jpg', thresh)
        cv2.imwrite(working_path+fNameWJ+'gray.jpg', gray)
        cv2.imwrite(working_path+fNameWJ+'dilate.jpg', dilate)
        cv2.imwrite(working_path+fNameWJ+'final.jpg', oimage)

#maskImages()
gridWhite()
#spliceImage()