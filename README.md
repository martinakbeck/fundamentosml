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