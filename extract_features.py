import cv2
import imutils
import numpy as np
import os
import glob
import csv
from extractor_gray import extract_gray_features
from extractor_meta import extract_meta_features
from extractor_color import extract_color_features
import sys

# input_folder = sys.argv[1]
input_folder = '/home/remid/Documents/JESSICA-Projects/Projeto Venatura/Images Int - Segment/Class 3/TS'
# output_file = sys.argv[2]
output_file = '/home/remid/Documents/JESSICA-Projects/Projeto Venatura/Images Int - Segment/Dataset/3_TS.csv'

#print (input_folder)
#print (output_file)

#Image folder
files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]
#print files

csv_complete = []

meta_features_names = ['attributes','samples','dimension','cor_RG','cor_RB','cor_RH','cor_RS','cor_RV','cor_RI','cor_GB','cor_GH','cor_GS','cor_GV','cor_GI','cor_BH','cor_BS','cor_BV','cor_BI','cor_HS','cor_HV','cor_HI','cor_SV','cor_SI','cor_VI']
color_features_names = ['mean_H','std_H','mean_S','std_S','mean_V','std_V','std_hist_H','kurt_hist_H','skew_hist_H','std_hist_S','kurt_hist_S','skew_hist_S','std_hist_V','kurt_hist_V','skew_hist_V']
color_features_names = color_features_names + ['mean_R','std_R','mean_G','std_G','mean_B','std_B','std_hist_R','kurt_hist_R','skew_hist_R','std_hist_G','kurt_hist_G','skew_hist_G','std_hist_B','kurt_hist_B','skew_hist_B']
color_features_names = color_features_names + ['nump_sobel','hu_sobel1','hu_sobel2','hu_sobel3','hu_sobel4','hu_sobel5','hu_sobel6','hu_sobel7']
color_features_names = color_features_names + ['nump_canny','hu_canny1','hu_canny2','hu_canny3','hu_canny4','hu_canny5','hu_canny6','hu_canny7']
gray_features_names = ['mean_I','std_I','entropy_I','std_hist_I','kurt_hist_I','skew_hist_I','lbp_0','lbp_1','lbp_2','lbp_3','lbp_4','lbp_5','lbp_6','lbp_7',
                'lbp_8','lbp_9','com_entropy','com_inertia','com_energy','com_correlation','com_homogeniety','FFT_energy','FFT_entropy','FFT_intertia','FFT_homogeneity', 'SNM','EME']

features_names = ['Name_file'] + meta_features_names + color_features_names + gray_features_names
#rotulos = []

#with open('rotulos_berkley.csv', 'rb') as csvfile:
#    spamreader = csv.reader(csvfile)
#    for index,row in enumerate(spamreader):
#        rotulos.append(row)

#rotulos = [ row[1:] for row in rotulos ] #remove nome do arquivo


#features_names = features_names + rotulos[0]
csv_complete.append(features_names)

for index,arquivo in enumerate(files):
    nome_arquivo_completo = os.path.basename(arquivo)
    print (nome_arquivo_completo)
    if(cv2.imread(arquivo) is not None):
        img_rgb = cv2.imread(arquivo)
        img_gray = cv2.imread(arquivo,0)
        linha_csv = [nome_arquivo_completo]
        linha_csv = linha_csv + extract_meta_features(img_rgb)
        linha_csv = linha_csv + extract_color_features(img_rgb)
        linha_csv = linha_csv + extract_gray_features(img_gray)
        #indice_rotulo = index/6 +1
        #linha_csv = linha_csv + rotulos[indice_rotulo]
        csv_complete.append(linha_csv)
    #print nome_arquivo_completo



with open(output_file, 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for i in csv_complete:
        wr.writerow(i)
