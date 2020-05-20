import cv2
import numpy as np
from scipy import stats
from skimage import feature
import math


def desvio(x):
    vet = x.ravel()
    desvio = stats.tstd(vet)
    return np.ones(len(vet))*desvio


def block_process(img, X, fun):
    img_array = img.ravel()
    height, width = img.shape[:2]
    length = len(img_array)
    B = np.zeros(length).reshape(img.shape[:2])
    process_length = X*X
    for i in range(0, height, X):
        limit_h = i + X
        if(i+X > height):
            limit_h = height
        for j in range(0, width, X):
            limit_w = j + X
            if(j+X > width):
                limit_w = width
            bloco = img[i:limit_h, j:limit_w]
            bloco_ravel = bloco.ravel()
            # print "i:",i,"j:",j
            if(len(bloco_ravel) < process_length):
                zeros = np.zeros(process_length - len(bloco_ravel))
                bloco_ravel = np.concatenate((bloco_ravel, zeros))
            block = fun(bloco_ravel)
            # print block.reshape(bloco.shape[:2])
            h, w = bloco.shape[:2]
            limite_total = h*w
            B[i:limit_h, j:limit_w] = block[0:limite_total].reshape(
                bloco.shape[:2])

    return B


def SNM(img):
    L_ldr = img.astype("double")
    u = stats.tmean(L_ldr.ravel())
    I1 = block_process(L_ldr, 11, desvio)
    I1 = I1.reshape(img.shape[:2])
    sig = stats.tmean(I1.ravel())
    phat_1 = 4.4
    phat_2 = 10.1
    beta_mode = (phat_1 - 1)/(phat_1 + phat_2 - 2)
    C_0 = stats.beta.pdf(beta_mode, phat_1, phat_2)
    C = stats.beta.pdf(sig/64.29, phat_1, phat_2)
    pc = C/C_0
    # pc
    muhat = 115.94
    sigmahat = 27.99
    B_0 = stats.norm.pdf(muhat, muhat, sigmahat)
    B = stats.norm.pdf(u, muhat, sigmahat)
    pb = B/B_0
    N = pb*pc
    return N


def eme(img, M, L):
    how_many = int(math.floor(M/L))
    E = 0
    B1 = np.zeros(L)
    m1 = 0
    for m in range(0, how_many):
        n1 = 0
        for n in range(0, how_many):
            B1 = img[m1:(m1+L), n1:(n1+L)]
            b_min = np.min(B1.ravel())
            b_max = np.max(B1.ravel())
            if b_min > 0:
                b_ratio = b_max/b_min
                E = E + 20*np.log(b_ratio)
            n1 = n1 + L
        m1 = m1 + L
    E = (E/how_many)/how_many
    return E


#mean_I,std_I,entropy_I,std_hist_I,kurt_hist_I,skew_hist_I,lbp_0,lbp_1,lbp_2,lbp_3,lbp_4,lbp_5,lbp_6,lbp_7,lbp_8,lbp_9,com_entropy,com_inertia,com_energy,com_correlation,com_homogeniety,FFT_energy,FFT_entropy,FFT_intertia,FFT_homogeneity, SNM,EME

def extract_gray_features(img):
    #img = cv2.imread(arquivo,0)
    histogram, bins = np.histogram(img.ravel(), 256, [0, 256])
    normalized_histogram = histogram.astype(
        float) / sum(histogram)  # Normalizing Histogram

    descritores = []

    # Espaciais
    #-----------------------#
    media_img = stats.tmean(img.ravel())
    std_img = stats.tstd(img.ravel())
    entropy_img = stats.entropy(histogram, base=2)
    descritores.append(media_img)
    descritores.append(std_img)
    descritores.append(entropy_img)
    #------------------------#

    # histogram
    #----------------------#
    std_hist = np.std(normalized_histogram)
    kurt_hist = stats.kurtosis(normalized_histogram, fisher=False)
    skew_hist = stats.skew(normalized_histogram)
    descritores.append(std_hist)
    descritores.append(kurt_hist)
    descritores.append(skew_hist)
    #----------------------#

    # LBP
    #----------------------#

    #lbp_default = feature.local_binary_pattern(img,8,1,method="default")
    #lbp_ror = feature.local_binary_pattern(img,8,1,method="ror")
    #lbp_nriuniform = feature.local_binary_pattern(img,8,1,method="nri_uniform")
    #lbp_var = feature.local_binary_pattern(img,8,1,method="var")
    lbp_uniform = feature.local_binary_pattern(img, 8, 1, method="uniform")
    (hist, _) = np.histogram(lbp_uniform.ravel(), 10, [0, 10])
    hist = hist/np.linalg.norm(hist)
    descritores = descritores + hist.tolist()

    #----------------------#
    # Co-ocorrence MATRIX
    #---------------------------------#
    co_ocurrence_matrix = feature.greycomatrix(
        img, [1], [0], levels=256, normed=True)
    co_ocurrence_matrix_nonorm = feature.greycomatrix(
        img, [1], [0], levels=256, normed=False)
    co_ocurrence_matrix2 = co_ocurrence_matrix[:, :, 0, 0]
    co_ocurrence_matrix_nonorm = co_ocurrence_matrix_nonorm[:, :, 0, 0]

    co_ocurrence_matrix_nonorm[co_ocurrence_matrix_nonorm >= 1] = 1

    hist_glcm, bins = np.histogram(
        co_ocurrence_matrix_nonorm.ravel(), 256, [0, 1])
    norm_hist_glcm = hist_glcm.astype(float) / sum(hist_glcm)
    soma = 0
    for i in range(0, 256):
        if(norm_hist_glcm[i] != 0):
            soma = soma + norm_hist_glcm[i]*np.log2(norm_hist_glcm[i])

    com_entropy = -1*soma
    com_inertia = feature.greycoprops(co_ocurrence_matrix, 'contrast')
    com_correlation = feature.greycoprops(co_ocurrence_matrix, 'correlation')
    com_energy = feature.greycoprops(co_ocurrence_matrix, 'energy')
    com_energy = com_energy * com_energy
    com_homogeniety = 0

    for i in range(0, co_ocurrence_matrix2.shape[0]):
        for j in range(0, co_ocurrence_matrix2.shape[1]):
            com_homogeniety = com_homogeniety + \
                co_ocurrence_matrix2[i, j] / (1 + abs(i-j))

    descritores.append(com_entropy)
    descritores.append(com_inertia.ravel()[0])
    descritores.append(com_energy.ravel()[0])
    descritores.append(com_correlation.ravel()[0])
    descritores.append(com_homogeniety)

    #---------------------------------#

    # Fourier Transform
    #---------------------------------#
    F = np.fft.fft2(img)
    NFP = F

    spectral_energy = sum(sum(abs(F)*abs(F)))
    spectral_energy = 1.0/F.ravel().size * spectral_energy

    F = np.fft.fftshift(F)
    F = abs(F)
    F = np.log(F+1)

    out = np.zeros(F.ravel().size, np.double)
    F = cv2.normalize(F, out, 1.0, 0.0, cv2.NORM_MINMAX)
    NFP = abs(np.real(NFP))/math.sqrt(abs(spectral_energy))

    NFP_energy = 0
    NFP_entropy = 0
    NFP_inertia = 0
    NFP_homogeneity = 0
    for i in range(0, NFP.shape[0]):
        for j in range(0, NFP.shape[1]):
            NFP_entropy = NFP_entropy + NFP[i, j] * np.log(NFP[i, j])

            NFP_energy = NFP_energy + (NFP[i, j] * NFP[i, j])

            NFP_inertia = NFP_inertia + (i-j) * NFP[i, j]

            NFP_homogeneity = NFP_homogeneity + NFP[i, j]/(1 + abs(i-j))

    NFP_energy = NFP_energy / NFP.ravel().size
    NFP_homogeneity = NFP_homogeneity / NFP.ravel().size
    NFP_entropy = NFP_entropy / NFP.ravel().size
    NFP_inertia = NFP_inertia / NFP.ravel().size

    descritores.append(NFP_energy)
    descritores.append(NFP_entropy)
    descritores.append(NFP_inertia)
    descritores.append(NFP_homogeneity)
    #-----------------------------------------#

    # QUALIDADE
    #---------------------------------------#
    # SMN

    descritores.append(SNM(img))

    # EME
    nrows = img.shape[0]
    ncols = img.shape[1]

    if(nrows > ncols):
        csize = ncols
    else:
        csize = nrows

    if(csize % 2 != 0):
        csize = csize - 1

    xmin = int(round(ncols/2)) - int(round(csize/2))
    ymin = int(round(nrows/2)) - int(round(csize/2))

    crop = img[ymin:ymin+csize, xmin:xmin+csize]
    EME_original = eme(crop.astype("float"), crop.shape[0], 8)

    descritores.append(EME_original)

    #-------------------------------------------#
    return descritores
