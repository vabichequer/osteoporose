# -*- coding: utf-8 -*-
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn import decomposition
from sklearn import datasets
from math import floor, sqrt
import csv


def lerImagem(prefixo, classe):
	im = []
	if classe:
		im = Image.open('Class1/Image_1_' + str(prefixo) + '.tif')
	else:
		im = Image.open('Class0/Image_0_' + str(prefixo) + '.tif')
	
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
	
	imagensTreino = imagensTreino.reshape(len(imagensTreino), 400 * 400)
	imagensTeste = imagensTeste.reshape(len(imagensTeste), 400 * 400)
	
	return imagensTreino, imagensTeste, targetTeste, targetTreino

def pca(ar, numeroDeComponentes):
	pca = decomposition.PCA(n_components=numeroDeComponentes)
	pca.fit(ar)
	ar = pca.transform(ar)
	return ar
	
def avaliar(avaliador, multiplicador, imagensTreino, targetTreino, imagensTeste, targetTeste):
	print("# Solver: " + avaliador)

	numNeuronios = int(floor(sqrt(multiplicador * 2)))

	clf = MLPClassifier(hidden_layer_sizes = numNeuronios, solver=avaliador, alpha=1e-5, random_state=1)
	
	clf.fit(imagensTreino, targetTreino)
	
	predito = clf.predict(imagensTeste)
	print("# Resultado: " + str(np.mean(predito == targetTeste)))
	#partial_fit(training_set, training_result, classes=cls)
	
	scores = cross_val_score(clf, imagensTeste, targetTeste, cv = 5, verbose = 0, scoring='accuracy')
	print("# Validacao cruzada: " + str(np.mean(scores)))
	return np.mean(scores), np.mean(predito == targetTeste)

imagensTreino = []
imagensTreinoPCA = []
imagensTeste = []
imagensTestePCA = []
targetTreino = []
targetTeste = []

[imagensTreino, imagensTeste, targetTeste, targetTreino] = lerDados()

print("Imagens:")
print("\t Treino: " + str(imagensTreino.shape))
print("\t Teste: " + str(imagensTeste.shape))
print("Target:")
print("\t Treino: " + str(targetTreino.shape))
print("\t Teste: " + str(targetTeste.shape))

csv_file = open("resultadosComPCA.csv", 'wb')
writer = csv.writer(csv_file)
writer.writerow(["PCA", "Camada oculta", "LBFGS", "VC", "Media", "SGD", "VC", "Media", "ADAM", "VC", "Media"])

for i in range(1, 101):
	imagensTreinoPCA = pca(imagensTreino, i)
	imagensTestePCA = pca(imagensTeste, i)
	
	print("\n###########################################################\n#")
	print("# Usando PCA com " + str(i) + " componentes.")
	print("#\n#----------------------------------------------------------\n#")
	[cross_val_lbfgs, mean_lbfgs] = avaliar("lbfgs", i, imagensTreinoPCA, targetTreino, imagensTestePCA, targetTeste)
	print("#\n#----------------------------------------------------------\n#")
	[cross_val_sgd, mean_sgd] = avaliar("sgd", i, imagensTreinoPCA, targetTreino, imagensTestePCA, targetTeste)
	print("#\n#----------------------------------------------------------\n#")
	[cross_val_adam, mean_adam] = avaliar("adam", i, imagensTreinoPCA, targetTreino, imagensTestePCA, targetTeste)
	writer.writerow([i, int(floor(sqrt(i * 2))), "", cross_val_lbfgs, mean_lbfgs, "", cross_val_sgd, mean_sgd, "", cross_val_adam, mean_adam])
	print("#\n###########################################################\n")
