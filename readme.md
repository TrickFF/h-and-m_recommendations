## Рекомендательная система (+EDA) на основе [датасета H&M](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)

### Подробная [презентация по проекту](https://docs.google.com/presentation/d/1tfh8iZZZ1sbf3mtkmAQ1Iaqj4__Q3-YToo31DbVIq0A/edit?usp=sharing)

### Основные использованные библиотеки:
1. Numpy 1.22.0
2. Numba 0.56.4
3. Implicit 0.6.1
4. Pandas 1.5.3
5. Scipy 1.7.3
6. CatBoost 1.1.1

### Задача: Предсказать 12 категорий покупок для покупателей на неделю вперед
Метрика оценки - MAP@12.

Для решения задачи использована двухуровневая рекомендательная система:
- Модель 1го уровня - ансамбль моделей ALS implicit OwnRec + user-user + item-item. Выдет список из 500 рекомендаций для каждого покупателя.
- Модель 2го уровня - 4 модели CatBoostClassifier по возрасной группе покупателей ранжирует соотвтетсвующие рекомендации модели 1го уровня и выдет top12.

#### Лучшее решение по метрике MAP@12 на валидации: **0.0294**

### Описание данных
##### Папки:
1. "archive" - папка под файлы данных, сюда необходимо загрузить [данные из датасета](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data), если требуется воспроизвести работу кода.
2. "catboost_model" - содержит обученные модели для 4х возрастных групп.
3. "rec_lib" - содержит файлы с функциями обработки данных, выдачи предсказаний и расчета метрик:
	- "metrics.py" - метрики,
	- "utils.py" - обрабока данных, создание фичей,
	- "models.py" - функции выдачи предсказаний классификаторами.
##### Файлы:
1. "1._ETL.ipynb" - группировка дублирующихся записей в transactions_train, сжатие данных, сохранение в формате parquet.
2. "2._train-test_lvl1_model_top500_recs.ipynb" - получение и сохранение рекомендаций моделей 1го уровня для train-test.
3. "3._train-test_lvl2_model_top12_recs.ipynb" - получение и сохранение рекомендаций моделей 2го уровня для train-test.
4. "4._validation_lvl1_model_top500_recs.ipynb" - получение и сохранение рекомендаций моделей 1го уровня для валидационной выборки.
5. "5._validation_lvl2_model_top12_recs.ipynb" - получение и сохранение рекомендаций моделей 2го уровня для валидационной выборки.