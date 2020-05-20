# gray FEATURES

#features de contorno

import cv2
import numpy as np
from scipy import stats
from skimage import feature
import math

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged


#mean_H,std_H,mean_S,std_S,mean_V,std_V,std_hist_H,kurt_hist_H,skew_hist_H,std_hist_S,kurt_hist_S,skew_hist_S,std_hist_V,kurt_hist_V,skew_hist_V,
#mean_R,std_R,mean_G,std_G,mean_B,std_B,std_hist_R,kurt_hist_R,skew_hist_R,std_hist_G,kurt_hist_G,skew_hist_G,std_hist_B,kurt_hist_B,skew_hist_B,
#hu_sobel1,hu_sobel2,hu_sobel3,hu_sobel4,hu_sobel5,hu_sobel6,hu_sobel7
def extract_color_features(img_rgb):
    #img_rgb = cv2.imread(arquivo)

    img_rgb_32F = np.float32(img_rgb)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    img_hsv_32F = cv2.cvtColor(img_rgb_32F, cv2.COLOR_BGR2HSV)
    img_hsv_32F = img_hsv_32F

    descritores = []

    B,G,R = cv2.split(img_rgb)
    H,S,V = cv2.split(img_hsv)
    H_32,S_32,V_32 = cv2.split(img_hsv_32F)
    H_32 = H_32 / 360.0 #Scaling to 0 to 1
    V_32 = V_32/255.0 #Scaling to 0 to 1

    hist_B,_ = np.histogram(B.ravel(),256,[0,256])
    hist_B = hist_B.astype(float) / sum(hist_B)
    hist_G,_ = np.histogram(G.ravel(),256,[0,256])
    hist_G = hist_G.astype(float) / sum(hist_G)
    hist_R,_ = np.histogram(R.ravel(),256,[0,256])
    hist_R = hist_R.astype(float) / sum(hist_R)

    H_32[0:10,0:10]

    hist_H,bins = np.histogram(H_32.ravel(),256,[0,1])
    hist_H = hist_H.astype(float) / sum(hist_H)

    hist_S,_ = np.histogram(S_32.ravel(),256,[0,1])
    hist_S = hist_S.astype(float) / sum(hist_S)

    hist_V,_ = np.histogram(V.ravel(),256,[0,256])
    hist_V = hist_V.astype(float) / sum(hist_V)

    #media e desvio H
    #media e desvio S
    #media e desvio V
    mean_H = stats.tmean(H_32.ravel())
    std_H = stats.tstd(H_32.ravel())
    mean_S = stats.tmean(S_32.ravel())
    std_S = stats.tstd(S_32.ravel())
    mean_V = stats.tmean(V_32.ravel())
    std_V = stats.tstd(V_32.ravel())

    descritores = descritores + [mean_H,std_H,mean_S,std_S,mean_V,std_V]

    #std kurt skew hist H
    #std kurt skew hist S
    #std kurt skew hist V
    std_hist_H = stats.tstd(hist_H)
    kurt_hist_H = stats.kurtosis(hist_H,fisher=False)
    skew_hist_H = stats.skew(hist_H)

    std_hist_S = np.std(hist_S)
    kurt_hist_S = stats.kurtosis(hist_S,fisher=False)
    skew_hist_S = stats.skew(hist_S)

    std_hist_V = np.std(hist_V)
    kurt_hist_V = stats.kurtosis(hist_V,fisher=False)
    skew_hist_V = stats.skew(hist_V)

    descritores = descritores + [std_hist_H,kurt_hist_H, skew_hist_H,std_hist_S,kurt_hist_S,skew_hist_S,std_hist_V,kurt_hist_V,skew_hist_V]

    #media e desvio R
    #media e desvio G
    #media e desvio B
    mean_R = stats.tmean(R.ravel())
    std_R = stats.tstd(R.ravel())
    mean_G = stats.tmean(G.ravel())
    std_G = stats.tstd(G.ravel())
    mean_B = stats.tmean(B.ravel())
    std_B = stats.tstd(B.ravel())

    descritores = descritores + [mean_R,std_R,mean_G,std_G,mean_B,std_B]

    #std kurt skew hist R
    #std kurt skew hist G
    #std kurt skew hist B
    std_hist_R = np.std(hist_R)
    kurt_hist_R = stats.kurtosis(hist_R,fisher=False)
    skew_hist_R = stats.skew(hist_R)

    std_hist_G = np.std(hist_G)
    kurt_hist_G = stats.kurtosis(hist_G,fisher=False)
    skew_hist_G = stats.skew(hist_G)

    std_hist_B = np.std(hist_B)
    kurt_hist_B = stats.kurtosis(hist_B,fisher=False)
    skew_hist_B = stats.skew(hist_B)

    descritores = descritores + [std_hist_R,kurt_hist_R, skew_hist_R,std_hist_G,kurt_hist_G,skew_hist_G,std_hist_B,kurt_hist_B,skew_hist_B]

    gray_image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_gray_32F = np.float32(gray_image)
    sobelx = cv2.Sobel(gray_image,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(gray_image,cv2.CV_64F,0,1)
    im_gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
    gmag = (im_gmag-im_gmag.min())/(im_gmag.max()-im_gmag.min())

    gmag_int = (255*gmag).astype('uint8')
    #cv2.imwrite('haha2.jpg',gmag_int)
    numpx = cv2.countNonZero(gmag_int)
    perc = float(numpx)/ img_gray_32F.size
    #gmag_int
    #ret,thresh = cv2.threshold(gmag_int,127,255,0)
    # print(im2, contours, hierarchy, gmag_int, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _,contours, hierarchy = cv2.findContours(gmag_int, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if (len(contours)> 0):
        cnt = contours[0]
        momentos = cv2.moments(cnt)
        inv_moments_1 = momentos.get('nu02')
        inv_moments_2 = momentos.get('nu02')
        inv_moments_3 = momentos.get('nu11')
        inv_moments_4 = momentos.get('nu12')
        inv_moments_5 = momentos.get('nu20')
        inv_moments_6 = momentos.get('nu21')
        inv_moments_7 = momentos.get('nu30')
    else:
        inv_moments_1 = 0
        inv_moments_2 = 0
        inv_moments_3 = 0
        inv_moments_4 = 0
        inv_moments_5 = 0
        inv_moments_6 = 0
        inv_moments_7 = 0

    descritores = descritores + [perc,inv_moments_1,inv_moments_2,inv_moments_3,inv_moments_4,inv_moments_5,inv_moments_6,inv_moments_7]

    #import cv2
    #gray_image = cv2.imread('./teste/croppedImage_77_cotenna_S-191005-10.jpg',0) #venatura
    #gray_image = cv2.imread('./teste/croppedImage_61_cotenna_S-191005-20.jpg',0) #non-venatura
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    canny_edge = cv2.Canny(blurred, 20, 60)
    #canny_edge
    #canny_edge = auto_canny(blurred)
    #canny_edge
    #cv2.imwrite('haha.jpg',canny_edge)

    numpx = cv2.countNonZero(canny_edge)
    perc = float(numpx)/ canny_edge.size
    _, contours,hierarchy = cv2.findContours(canny_edge.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if(len(contours) > 0):
        cnt = contours[0]
        momentos = cv2.moments(cnt)

        inv_moments_1 = momentos.get('nu02')
        inv_moments_2 = momentos.get('nu02')
        inv_moments_3 = momentos.get('nu11')
        inv_moments_4 = momentos.get('nu12')
        inv_moments_5 = momentos.get('nu20')
        inv_moments_6 = momentos.get('nu21')
        inv_moments_7 = momentos.get('nu30')
    else:
        inv_moments_1 = 0
        inv_moments_2 = 0
        inv_moments_3 = 0
        inv_moments_4 = 0
        inv_moments_5 = 0
        inv_moments_6 = 0
        inv_moments_7 = 0

    descritores = descritores + [perc,inv_moments_1,inv_moments_2,inv_moments_3,inv_moments_4,inv_moments_5,inv_moments_6,inv_moments_7]

    return descritores
