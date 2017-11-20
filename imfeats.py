import sys
import numpy as np
import skimage.io as imgio
from skimage import color
from skimage.feature import greycomatrix, greycoprops
from scipy.stats import describe

PERCENTILES = [1, 10, 25, 75, 90, 99]

GLCM_DISTANCES = [1, 3, 5, 10, 15]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_ANGLES_DEG = [int(np.rad2deg(x)) for x in GLCM_ANGLES]
GLCM_PROPS = ['contrast', 'dissimilarity', 'homogeneity',
              'energy', 'correlation', 'ASM']

im = imgio.imread(sys.argv[1])

# Se a imagem for colorida, converter para cinza
im = color.rgb2gray(im)

features = {}

# Estatísticas
stats = describe(im, axis=None)
features['mean'] = stats.mean
features['variance'] = stats.variance
features['skewness'] = stats.skewness
features['kurtosis'] = stats.kurtosis

# Percentis do histograma
for perc in PERCENTILES:
    features['percentile_%d' % perc] = np.percentile(im, perc, axis=None)

# GLCM
glcm = greycomatrix(im, GLCM_DISTANCES, GLCM_ANGLES)
for prop in GLCM_PROPS:
    glcm_props = greycoprops(glcm, prop=prop)
    for dist_ix, dist in enumerate(GLCM_DISTANCES):
        for ang_ix, ang in enumerate(GLCM_ANGLES_DEG):
            name = 'glcm_%s_%d_%d' % (prop, dist, ang)
            features[name] = glcm_props[dist_ix, ang_ix]

# Resultado
print(features)
