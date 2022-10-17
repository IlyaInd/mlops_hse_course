# Домашняя работа №1

Выполнение задания в рамках курса MLOps по реализации API. 
- Реализована документация в swagger с помощью flask-restx
- Пока не реализовано использование линтера
- Нет функции предикта
- Нет функции предоставления списка всех моделей
- Чего-то еще точно нет и я про это забыл, но это короче пока v0.1 всё же :)

Доступные модели:
```models_id = {1: Ridge, 2: RandomForestRegressor}```



## Использование

```bash
# upload train data in csv format
$ curl -X PUT http://127.0.0.1:5000/data/train -d 'filename=winequality-red.csv' -d 'target_name=quality'

# configure model parameters
$ curl -X PUT http://127.0.0.1:5000/models/1 -d -d '{"alpha": 0.1, "fit_intercept": true}' 

# inspect current model configuration
$ curl -X GET  http://127.0.0.1:5000/models/1                                                                   

# fit model on train data
$ curl -X PUT http://127.0.0.1:5000/models/1/fit  

# validate model on 5-fold cross-validation
$ curl -X GET http://127.0.0.1:5000/models/1/validate    

```
