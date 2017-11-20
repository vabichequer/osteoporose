# -*- coding: utf-8 -*-
import sys
import numpy as np
import skimage.io as imgio
import os
import csv

from skimage import color
from scipy.stats import describe
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

def descrever(nomeDoCSV, imClass, data):
	csv_file = open(nomeDoCSV, 'wb')
	writer = csv.writer(csv_file)
	i = 0
	writer.writerow(["Mean", "Variance", "Skewness", "Kurtosis"])
	
	# Se a imagem for colorida, converter para cinza
	for image in imClass:
		image = color.rgb2gray(image)
		# Estatística
		stats = describe(image, axis=None)
		# Resultado
		writer.writerow([stats.mean, stats.variance, stats.skewness, stats.kurtosis])
		data.append([stats.mean, stats.variance, stats.skewness, stats.kurtosis])
		i = i + 1

def buscarArquivos(path, imclass):
	# Busca todos arquivos na pasta
	for file in os.listdir(path):
		imclass.append(imgio.imread(os.path.join(path, file)))

def avaliar(avaliador, hDataTrain, hTargetTrain, hDataEval, hTargetEval):
	print("\n###########################################################\n")
	print("Resultado com o solver " + avaliador)
	clf = MLPClassifier(solver=avaliador, alpha=1e-5, random_state=1)
	
	clf.fit(hDataTrain, hTargetTrain)
	
	predito = clf.predict(hDataEval)
	print("Resultado: " + str(np.mean(predito == hTargetEval)))
	
	scores = cross_val_score(clf, hDataEval, hTargetEval, cv = 5, verbose = 0, scoring='accuracy')
	print("Validacao cruzada: " + str(np.mean(scores)))
		
class0 = "C:\Users\Vicenzo\Desktop\ia-ec-2017-2-tp3-master\Nayanne_Vicenzo\Class0"
class1 = "C:\Users\Vicenzo\Desktop\ia-ec-2017-2-tp3-master\Nayanne_Vicenzo\Class1"

imClass0 = []
imClass1 = []

data = []
target = []

buscarArquivos(class0, imClass0)
buscarArquivos(class1, imClass1)

print("Descrevendo a Classe 0...")
descrever("classe0.csv", imClass0, data)
for i in range(0, 58):
	target.append(0)
print("Descrevendo a Classe 1...")
descrever("classe1.csv", imClass1, data)
for i in range(0, 58):
	target.append(1)

data = np.asarray(data)
target = np.asarray(target)

hDataTrain = []
hTargetTrain = []

hDataEval = []
hTargetEval = []

print(data.shape)
print(target.shape)

for i in range(0, 29):
	hDataTrain.append(data[i])
	hTargetTrain.append(target[i])
	hDataEval.append(data[i + 29])
	hTargetEval.append(target[i + 29])
	
for i in range(58, 87):
	hDataTrain.append(data[i])
	hTargetTrain.append(target[i])
	hDataEval.append(data[i + 29])
	hTargetEval.append(target[i + 29])

hDataTrain = np.asarray(hDataTrain)
hTargetTrain = np.asarray(hTargetTrain)
hDataEval = np.asarray(hDataEval)
hTargetEval = np.asarray(hTargetEval)

#http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier

avaliar("lbfgs", hDataTrain, hTargetTrain, hDataEval, hTargetEval)
avaliar("sgd", hDataTrain, hTargetTrain, hDataEval, hTargetEval)
avaliar("adam", hDataTrain, hTargetTrain, hDataEval, hTargetEval)

