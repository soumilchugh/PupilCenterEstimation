import csv
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def loadData():
    normalizedImg = np.zeros((64, 64))
    normalizedImg1 = np.zeros((64, 64))
    with open('/home/soumil/eye-tracking-ml/set_office_crop_64/data.txt', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            img = cv2.imread('/home/soumil/eye-tracking-ml/set_office_crop_64/' + row[0], 0)
            equ = cv2.equalizeHist(img)
            #otsu_thresh_val,normalizedImg1 = cv2.threshold(equ, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU);
            #equ1 = cv2.Canny(equ,250,255)
            #high_thresh_val  = otsu_thresh_val
            #lower_thresh_val = otsu_thresh_val * 0.5;
            #equ1 = cv2.Canny( equ, lower_thresh_val, high_thresh_val );
            #equ = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            #equ = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
            #kernel = np.ones((5,5),np.float32)/25
            #equ1 = cv2.filter2D(equ,-1,kernel)
            #equ = cv2.blur(img,(5,5))
            #equ1 = cv2.medianBlur(equ,5)
            #equ1 = cv2.Laplacian(equ,cv2.CV_64F)
            #equ1 = cv2.bilateralFilter(equ,5,10,10)
            #equ1 = adjust_gamma(equ,0.5)
            #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
            #cl1 = clahe.apply(img)
            equ1 = adjust_gamma(equ,1.5)
            #g = 10 * (np.log(1 + np.float32(equ/255)))
            cv2.imwrite('/home/soumil/eye-tracking-ml/PowerLaw+Hist/' + row[0],equ1)            
              
def main():
    loadData()

if __name__ == '__main__':
    main()




