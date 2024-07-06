import pandas as pd
import numpy as np
import matplotlib as mt
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn as sk
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score

#Descrição do atributo

#Idade em anos
#Sexo 1 = masculino, 0= feminino
#Tipo de dor torácica   1 = angina típica, 2= angina atípica, 3= dor não anginosa, 4= assintomático
#descansando bp s em mm Hg
#colesterol em mg/dl
#Açúcar no sangue em jejum, (glicemia em jejum > 120 mg/dl) (1 = verdadeiro; 0 = falso)
#Eletro Em repouso  0= normal, 1= ter anormalidade na onda ST-T (inversões da onda Te/ou elevação ou depressão de ST > 0,05 mV)
# 2= mostrando provável ou definitiva hipertrofia pelos critérios de Estes
#frequência cardíaca máxima entre 71 a 202
#Exercício induzido angina 1 = sim, 0 = não
#pico antigo em númerico
#inclinação ST  1= inclinação ascendente, 2= plano, 3= descida
#target  1 = doença cardíaca, 0 = Normal





dataset = 'heart_statlog_cleveland_hungary_final.csv'
path = './'

#importar o arquivo
def importarArquivo(path, nomeArquivo):
    print("Caminho do arquivo: %s" % path)
    print("Nome do Arquivo: %s" % nomeArquivo)

    dataframe = pd.read_csv(path+nomeArquivo, delimiter=',', encoding='utf-8')

    return dataframe
    
def tratamentoDados(dataframe):

    # renomear colunas
    dataframe.rename(columns={'sex': 'sexo', 'chest pain type': 'tipo dor no peito'}, inplace=True)    
    dataframe.rename(columns={'age': 'idade', 'resting bp s': 'descansando bp s'}, inplace=True) 
    dataframe.rename(columns={'cholesterol': 'colesterol', 'fasting blood sugar': 'açúcar no sangue em jejum'}, inplace=True)
    dataframe.rename(columns={'resting ecg': 'eletro em repouso', 'max heart rate': 'frequência cardíaca máxima'}, inplace=True)
    dataframe.rename(columns={'exercise angina': 'angina de exercício', 'oldpeak': 'pico antigo'}, inplace=True)
    dataframe.rename(columns={'ST slope': 'Inclinação ST'}, inplace=True)

    print(dataframe.head())    
    print(dataframe.tail())
    print(dataframe.info())
    print(dataframe.describe())

    # remover duplicatas do dataset
    dataframe.drop_duplicates()
    
    # remover dados nulos (se existir)
    dataframe.dropna(inplace=True)
        
    return dataframe

def visualizacaoGrafica(dataframe):
    dadosGrupo = dataframe.groupby('target').groups

    print(dadosGrupo)

    lb = []
    vl = []

    for grp in dadosGrupo:
        print(grp)
        lb.append(str(grp))
        print(len(dadosGrupo[grp]))
        vl.append(len(dadosGrupo[grp]))

    bar_colors = ['tab:red', 'tab:blue']
    plt.bar(lb, vl, color=bar_colors)
    plt.show()
    return dataframe

def visualizacaoGrafica2(dataframe):
    # Verificar se as colunas 'sexo' e 'target' existem no DataFrame
    if dataframe is None:
        print("Erro: O DataFrame é None.")
        return

    if 'sexo' not in dataframe.columns or 'target' not in dataframe.columns:
        print("Erro: As colunas 'sexo' ou 'target' não existem no DataFrame.")
        print("Colunas do DataFrame:", dataframe.columns)
        return
    
    #dataframe['sexo'] = dataframe['sexo'].map({0: 'feminino', 1: 'masculino'})

    # Manter os valores únicos da coluna 'target' como numéricos
    cardiaco = dataframe['target'].unique()

    print("Valores únicos na coluna 'target':", cardiaco)
    
    # Agrupar por dois tipos de colunas
    dadosGrupo = dataframe.groupby(['sexo', 'target']).size().unstack(fill_value=0)

    print("Dados agrupados por 'sexo' e 'target':\n", dadosGrupo)

    width = 0.6  # Largura das barras

    fig, ax = plt.subplots()
    bottom = np.zeros(len(cardiaco))
    

    for sex in dadosGrupo.index:
        arr = dadosGrupo.loc[sex].values
        p = ax.bar(cardiaco, arr, width, label=sex, bottom=bottom)
        bottom += arr

        ax.bar_label(p, label_type='center')
    
    ax.set_title('cárdiaco ou não por sexo')
    ax.set_xlabel('0 = Normal  1 = doença cardíaca')
    ax.set_ylabel('Contagem')
    ax.legend()

    plt.show()
    return dataframe

def visualizacaoGrafica3(dataframe):

    fig, ax = plt.subplots()
    ax.set_ylabel('boxplot variáveis da idade')

    print(dataframe['idade'])

    print(dataframe.columns)

    for n, col in enumerate(dataframe.columns):
        if col == 'idade':
            ax.boxplot(dataframe[col], positions=[n+1])

    plt.show()
    return dataframe



if __name__ == '__main__':

    # importar o dataset
    dataframe = importarArquivo(path, dataset)
    
    # executar o tratamentos dos dados
    dataframe = tratamentoDados(dataframe)

    # executar gráfico
    dataframe = visualizacaoGrafica(dataframe)
    dataframe = visualizacaoGrafica2(dataframe)
    dataframe = visualizacaoGrafica3(dataframe)

    #print(dataframe)

    X = dataframe.drop(['target'],axis=1)
    y = dataframe['target']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        shuffle=True, random_state=42, 
                                                        stratify=y)
    print(x_train)
    print(x_test)
    print(y_train)
    print(y_test)

    train, test = train_test_split(dataframe, test_size=0.3, 
                                                        shuffle=True, random_state=42, 
                                                        stratify=y)
    print(train)
    print(test)

    # quebra simplificada: do teste e validação em 50%
    y = test['target']
    test, validacao = train_test_split(test, test_size=0.5, stratify=y)

    print(test)
    print(validacao)
    

    # geração dos arquivos de treino teste e validação
    #df_train = pd.DataFrame(train)
    #df_test = pd.DataFrame(test)
    #df_validacao = pd.DataFrame(validacao)

    #df_train.to_csv("train.csv")
    #df_test.to_csv("test.csv")
    #df_validacao.to_csv("validacao.csv")

cls1 = RandomForestClassifier(n_estimators=100)
cls2 = LogisticRegression(max_iter=1000)
cls3 = GaussianNB()
cls4 = svm.SVC()
cls5 = GradientBoostingClassifier(n_estimators=100)
cls6 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), max_iter=2000)

cls1.fit(x_train, y_train)
cls2.fit(x_train, y_train)
cls3.fit(x_train, y_train)
cls4.fit(x_train, y_train)
cls5.fit(x_train, y_train)
cls6.fit(x_train, y_train)

y_pred1 = cls1.predict(x_test)
y_pred2 = cls2.predict(x_test)
y_pred3 = cls3.predict(x_test)
y_pred4 = cls4.predict(x_test)
y_pred5 = cls5.predict(x_test)
y_pred6 = cls6.predict(x_test)

print("Acuracia cls1:", accuracy_score(y_pred1, y_test))
print("Acuracia cls2:", accuracy_score(y_pred2, y_test))
print("Acuracia cls3:", accuracy_score(y_pred3, y_test))
print("Acuracia cls4:", accuracy_score(y_pred4, y_test))
print("Acuracia cls5:", accuracy_score(y_pred5, y_test))
print("Acuracia cls6:", accuracy_score(y_pred6, y_test))
