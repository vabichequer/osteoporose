# -*- coding: utf-8 -*-
import sys
import numpy as np
import skimage.io as imgio
from skimage import color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from scipy.stats import describe
from PIL import Image
from sklearn import decomposition, datasets
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction import image
from keras.models import Sequential
from keras.layers import Dense
import csv

PERCENTILES = [1, 10, 25, 75, 90, 99]

GLCM_DISTANCES = [1, 3, 5, 10, 15, 20, 25]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_ANGLES_DEG = [int(np.rad2deg(x)) for x in GLCM_ANGLES]
GLCM_PROPS = ['contrast', 'dissimilarity', 'homogeneity',
              'energy', 'correlation', 'ASM']
			  
def gerarDados(im):
	# Checar se essa conversão não mata dados demais no histograma. 
	# Se não for um histograma homogêneo, eu posso ta perdendo muitos dados na conversão
	im = img_as_ubyte(im) 
	#features = {}
	features = []
	
	# Estatísticas
	stats = describe(im, axis=None)
	#features['mean'] = stats.mean
	features.append(stats.mean)
	#features['variance'] = stats.variance
	features.append(stats.variance)
	#features['skewness'] = stats.skewness
	features.append(stats.skewness)
	#features['kurtosis'] = stats.kurtosis
	features.append(stats.kurtosis)

	# Percentis do histograma
	for perc in PERCENTILES:
		#features['percentile_%d' % perc] = np.percentile(im, perc, axis=None)
		features.append(np.percentile(im, perc, axis=None))

	# GLCM
	glcm = greycomatrix(im, GLCM_DISTANCES, GLCM_ANGLES)
	for prop in GLCM_PROPS:
		glcm_props = greycoprops(glcm, prop=prop)
		for dist_ix, dist in enumerate(GLCM_DISTANCES):
			for ang_ix, ang in enumerate(GLCM_ANGLES_DEG):
				#name = 'glcm_%s_%d_%d' % (prop, dist, ang)
				features.append(glcm_props[dist_ix, ang_ix])
	return features

def lerImagem(prefixo, classe):
	im = []
	if classe:
		im = Image.open('Class1\Image_1_' + str(prefixo) + '.tif')
	else:
		im = Image.open('Class0\Image_0_' + str(prefixo) + '.tif')
	
	return im.getdata()

def lerImagens(classe, imagensTreino, imagensTeste, targetTreino, targetTeste):
	for i in range(1, 30):
		imagensTreino.append(lerImagem(i, classe))
		imagensTeste.append(lerImagem(i + 29, classe))
		targetTeste.append(classe)
		targetTreino.append(classe)

def lerDados():
	imagensTreino = []
	imagensTeste = []
	targetTreino = []
	targetTeste = []
	
	lerImagens(0, imagensTreino, imagensTeste, targetTreino, targetTeste)
	lerImagens(1, imagensTreino, imagensTeste, targetTreino, targetTeste)
		
	imagensTreino = np.array(imagensTreino)
	imagensTeste = np.array(imagensTeste)
	targetTreino = np.array(targetTreino)
	targetTeste = np.array(targetTeste)
	
	imagensTreino = imagensTreino.reshape(len(imagensTreino), 400, 400)
	imagensTeste = imagensTeste.reshape(len(imagensTeste), 400, 400)
	
	return imagensTreino, imagensTeste, targetTeste, targetTreino

def avaliar(avaliador, imagensTreino, targetTreino, imagensTeste, targetTeste):
	print("# Solver: " + avaliador)
	clf = MLPClassifier(solver=avaliador, alpha=1e-5, random_state=1)
	
	clf.fit(imagensTreino, targetTreino)
	
	predito = clf.predict(imagensTeste)
	
	scores = cross_val_score(clf, imagensTeste, targetTeste, cv = 5, verbose = 0, scoring='accuracy')
	print("# Validacao cruzada: " + str(np.mean(scores)))
	return np.mean(scores)

imagensTreino = []
imagensTeste = []
targetTreino = []
targetTeste = []
featuresTreino = []
featuresTeste = []

[imagensTreino, imagensTeste, targetTeste, targetTreino] = lerDados()
	
for i in range(0, 29):
	featuresTreino.append(gerarDados(imagensTreino[i]))
	featuresTeste.append(gerarDados(imagensTeste[i]))
		
for i in range(0, 29):
	featuresTreino.append(gerarDados(imagensTreino[i]))
	featuresTeste.append(gerarDados(imagensTeste[i]))

featuresTreino = np.array(featuresTreino)
featuresTeste = np.array(featuresTeste)
	
# csv_file = open("resultadosMCO.csv", 'wb')
# writer = csv.writer(csv_file)
# writer.writerow(["LBFGS", "VC", "SGD", "VC", "ADAM", "VC"])

# print("\n###########################################################\n#")
# cross_val_lbfgs = avaliar("lbfgs", featuresTreino, targetTreino, featuresTeste, targetTeste)
# print("#\n#----------------------------------------------------------\n#")
# cross_val_sgd = avaliar("sgd", featuresTreino, targetTreino, featuresTeste, targetTeste)
# print("#\n#----------------------------------------------------------\n#")
# cross_val_adam = avaliar("adam", featuresTreino, targetTreino, featuresTeste, targetTeste)
# writer.writerow(["", cross_val_lbfgs, "", cross_val_sgd, "", cross_val_adam])
# print("#\n###########################################################\n")

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
model.train_on_batch(featuresTreino, targetTreino)
loss_and_metrics = model.evaluate(featuresTeste, targetTeste, batch_size=128)
classes = model.predict(featuresTeste, batch_size=128)