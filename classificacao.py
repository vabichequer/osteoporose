# -*- coding: utf-8 -*-
import pandas as pd

print ("Importando pacotes... (sklearn.ensemble)");
from sklearn.ensemble import RandomForestClassifier
print ("Importando pacotes... (sklearn.model_selection)");
from sklearn.model_selection import train_test_split
print ("Importando pacotes... (sklearn.metrics)");
from sklearn.metrics import accuracy_score

print ("Iniciando leitura...");
# Lê a base de dados e define os campos usados na classificação
df = pd.read_csv('animals.csv')

print ("Leitura concluída.");

print ("Iniciando separação dos dados...");
# Separa 1/3 dos registros para teste, o resto para treino
campos_treino, campos_teste, classes_treino, classes_teste = train_test_split(
    df[df.columns[0:-1]], df[df.columns[-1]], test_size=0.33)
print ("Separação concluída.");

# Treina o classificador
print ("Iniciando treinamento...");
print ("Passo 1...");
cls = RandomForestClassifier()
print ("Passo 2...");
cls.fit(campos_treino, classes_treino)
print ("Passo 3...");
predito = cls.predict(campos_teste)
print ("Classificação concluída.");

print ("Avaliação...");
# Avalia o resultado
print("Acuracia:", accuracy_score(classes_teste, predito))

print (predito)
