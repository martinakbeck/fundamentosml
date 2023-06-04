# Fundamentos de Machine Learning

## Aprendizado Supervisionado

### KNN
* Classifica uma instância com base nas classes de seus vizinhos mais próximos.

``` 
# Seleção das features
x_train = features = columns que pode ter alguma similidariedade de distância com a variável resposta
y_train = label = variável resposta

# Definir parametros de treinamento
k = 7
knn_classifier = KNeighborsClassifier(n_neighbors = k)

# Treinamento do algoritmo
knn_classifier.fit(x_train, y_train)

# Previsão das observações
y_pred = knn_classifier.predict(x_train)

#manual
df_result['acertos'] = (df_result.loc[:, ['id_cliente', 'limite_adicional', 'classificacao']]
                        .apply( lambda x: 1 if x['limite_adicional'] == x['classificacao'] else 0, axis=1))
```

### Linear Regression
* A relação entre variáveis através de uma reta.

``` 
#seleção de features
x_train = df.loc[:, features]
y_train = df.loc[:, label]

#model definition
lr_model = LinearRegression()

#model fit (ajuste treinamento)
lr_model.fit( x_train, y_train)

#previsao
y_pred = lr.model.predict (x_train)

#manual
np.sum(( x_train.loc[0,].values * lr_model.coef_)) + lr_model.intercept_
```

## Aprendizado Não-Supervisionado

### K-Means
* Dividi o conjunto de dados em grupos (cluesters)

```
#definir o algoritmo
kmeans = ct.KMeans(
        n_clusters = n_clusters, #numero de clusters que achamos que tem
        init = 'random', #como é o processo de inicialização
        n_init = 10, #quantas vezes queremos que inicialize
        random_state = 0, #inicilize eles sempre a partir da mesma origem aleatória
        )

# fit - training
labels = kmeans.fit_predict(X)

#perfomance (test)
silhouette_avg = mt.silhouette_score(X, labels)
```