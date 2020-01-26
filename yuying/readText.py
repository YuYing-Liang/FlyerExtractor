from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

img_dir = os.getcwd() + "\\images" # Enter Directory of all images
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
# newFile = open("extractedText.txt","w+")

# for f1 in files:
    # img = cv2.imread(f1)
    # data.append(img)
    # img = Image.open(f1)
    # text = pytesseract.image_to_string(f1)
    # txtArray = text.split("\n\n");
    #
    # print(txtArray)
    # for str in txtArray:
    #     if "$" in str:
    #         print(str)
    # newFile.write(text + "\n")

#load the image.
image1 = cv2.imread('week_1_page_1_item_100.jpg')
image2 = cv2.imread('week_1_page_1_item_2.jpg')
img1 = 'week_1_page_1_item_100.jpg'
img2 = "week_1_page_1_item_2.jpg"
img3 = "week_52_page_4_item_1.jpg"
#blank_image = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)

def filterName(img):
    #convert to hsv color space.
    img = cv2.imread(img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred_img = cv2.GaussianBlur(img_hsv, (11, 11), cv2.BORDER_DEFAULT)
    mask = cv2.inRange(blurred_img, (0, 0, 0), (180, 40, 100))
    fin_mask = cv2.bitwise_and(blurred_img, blurred_img, mask = mask)
    fin_mask = cv2.cvtColor(fin_mask, cv2.COLOR_HSV2BGR)

    text = pytesseract.image_to_string(fin_mask)
    return text
    # cv2.imwrite( "final.jpg", finfin_mask);
    # cv2.imshow('image', fin_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def getPriceInfo(img):
    img = cv2.imread(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Range for lower red
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv, lower_red,upper_red)

    # Range for upper range
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv,lower_red,upper_red)

    # Generating the final mask to detect red color
    mask1 = mask1+mask2

    #identify text with black background
    dst1 = cv2.bitwise_and(img, img, mask=mask1)

    #identify text with grey scale image
    res2 = cv2.bitwise_not(mask1)

    #read text
    text = pytesseract.image_to_string(res2)
    return text

    #show image
    # cv2.imshow("image", res2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# returns 2 arguments: discount information and description of product
def filterDescription(img):
    img = Image.open(img)
    img = img.convert('RGBA')
    pix = img.load()
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if pix[x, y][0] < 102 or pix[x, y][1] < 102 or pix[x, y][2] < 102:
                pix[x, y] = (0, 0, 0, 255)
            else:
                pix[x, y] = (255, 255, 255, 255)

    text = pytesseract.image_to_string(img)
    txtArray = text.split('\n')
    # print(txtArray);
    desArr = [];
    discount = [];
    des = []
    for str in txtArray:
        if "SAVE" in str:
            discount += [str];
        elif len(str.split(" ")) >= 4 and "Discount" not in str:
            des += [str];

    desArr += [discount] + [des]
    return desArr;

def getLeastUnitForPromotion(str):
    if "/" in str:
        dashIndex = str.find("/")
        if "." in str:
            # endStr = str[dashIndex+1:]
            # return endStr.split(".")[0]
            return "1"
        else:
            return str[dashIndex-1:dashIndex]
    else:
        return "1"

def getDescription(arr):
    des = ""
    for str in arr:
        des += str.replace("\n"," ")
    return des

# 0 - returns save per unit
# 1 - returns discount price
def getDiscounts(discount, unit):
    unit = 0
    try:
        unit = int(unit)
    except:
        unit = 1

    if unit == 0:
        unit = 1

    disc = 0
    if len(discount) > 0:
        str = discount[0]
        if str != "":
            if "$" in str:
                startInd = str.find("$")
                temp = str[startInd+1:]
                if "/" in temp:
                    disc = int(temp[:temp.find("/")])
                else:
                    disc = int(temp[:temp.find(" ")])
                savePerUnit = disc/unit
                discountPrice = disc/(disc * unit)

                return [savePerUnit, discountPrice]
    return ["", ""]

def getPrice(str):
    price = ""
    if "/" in str:
        temp = str.split("/")
        if "." in str:
            str = temp[0]
        else:
            str = temp[1]
    else:
        str = str.split(" ")[0]

    for c in str:
        if ord(c) >= 48 and ord(c) <= 57:
            price += c
    if len(price) > 2:
        price = price[:(len(price)//2)] + "." + price[2:]

    return price

# print(filterName(image1))
# print(filterName(image2))
# arr = filterDescription(img2)
# print(arr)
# print(getUnitofMeasurement(arr[0]))
filDes = filterDescription(img1)
# print(filDes)
# print(getDescription(filDes[1]))
price = getPriceInfo(img1)
getPrice(price)
unit = getLeastUnitForPromotion(price)
print(getDiscounts(filDes[0], unit))
