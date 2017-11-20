# -*- coding: utf-8 -*-
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn import decomposition
from sklearn import datasets
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
	
def avaliar(avaliador, imagensTreino, targetTreino, imagensTeste, targetTeste):
	print("# Solver: " + avaliador)
	clf = MLPClassifier(hidden_layer_sizes=200, solver=avaliador, alpha=1e-5, random_state=1)
	
	clf.fit(imagensTreino, targetTreino)
	
	predito = clf.predict(imagensTeste)
	print("# Resultado: " + str(np.mean(predito == targetTeste)))
	
	scores = cross_val_score(clf, imagensTeste, targetTeste, cv = 5, verbose = 0, scoring='accuracy')
	print("# Validacao cruzada: " + str(np.mean(scores)))
	return np.mean(scores), np.mean(predito == targetTeste)

imagensTreino = []
imagensTeste = []
targetTreino = []
targetTeste = []

[imagensTreino, imagensTeste, targetTeste, targetTreino] = lerDados()

print("Imagens:")
print("\t Treino: " + str(imagensTreino.shape))
print("\t Teste: " + str(imagensTeste.shape))
print("Target:")
print("\t Treino: " + str(targetTreino.shape))
print("\t Teste: " + str(targetTeste.shape))

csv_file = open("resultadosSemPCA.csv", 'wb')
writer = csv.writer(csv_file)
writer.writerow(["LBFGS", "VC", "Media", "SGD", "VC", "Media", "ADAM", "VC", "Media"])

print("\n###########################################################\n#")
[cross_val_lbfgs, mean_lbfgs] = avaliar("lbfgs", imagensTreino, targetTreino, imagensTeste, targetTeste)
print("#\n#----------------------------------------------------------\n#")
[cross_val_sgd, mean_sgd] = avaliar("sgd", imagensTreino, targetTreino, imagensTeste, targetTeste)
print("#\n#----------------------------------------------------------\n#")
[cross_val_adam, mean_adam] = avaliar("adam", imagensTreino, targetTreino, imagensTeste, targetTeste)
writer.writerow(["", cross_val_lbfgs, mean_lbfgs, "", cross_val_sgd, mean_sgd, "", cross_val_adam, mean_adam])
print("#\n###########################################################\n")
