# Домашняя работа №1

Выполнение задания в рамках курса MLOps по реализации API. 


## Main features
- Доступны для обучения 2 типа моделей:```model_types = [ridge, random_forest]```
- Поддержка нескольких экземпляров одного и того же типа моделей с разными гиперпараметрами и обучением на разных датасетах
- Подсчет RMSE на тренировочных данных для оценки качества обучения

## How to run
Необходим `poetry >= 1.1.15`

Запуск без развертывания виртуального окружения:
 ```bash
 $ poetry run python main.py
 ```

Для установки всех зависимостей в виртуальном окружении и запуска сервера:
 ```bash
 $ cd mlops_hse_course
 $ poetry install
 $ python run main.py
 ```

## Usage 

Получение списка всех текущих моделей в сервисе:
```bash
curl -X GET http://127.0.0.1:5001/models
```

Вывод информации о конкретной модели:
```bash
$ curl -X GET http://127.0.0.1:5001/models/1
```

Обучение модели с передачей данных в виде списка списков:
<details>
<summary>Данные из примера в табличном виде</summary>

| data | target |
|------|--------|
| 1    | 1      |
| 2    | 3      |
| 5    | 8      |

</details>

```bash
$ curl -X PUT http://127.0.0.1:5001/models/0/fit \
  -H 'Content-Type: application/json' \
  -d '{"train_data": [[1], [2], [5]],
       "target": [1, 3, 8]}'
```

Конфигурирование модели с передачей гиперпараметров в виде JSON:
```bash
$ curl -X PUT http://127.0.0.1:5001/models/0 \
-H 'Content-Type: application/json' \
-d '{"params": {"alpha": 10}}'
```

## License :)
[MIT](LICENCE)
