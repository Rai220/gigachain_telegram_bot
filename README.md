# gigachain_telegram_bot
Пример телеграм-бота технической поддержки по вопросам GigaChain / GigaChat API

За основу взять модифицированый adaptive RAG.

## Описание проекта

* db_builder - набор утилит и данных для заполнения векторной базы данных знаниями
* bot.py - telegram bot
* graph_2.py - текущее описание графа мультиагентной системы в нотакции GigaGraph / LangGraph
-----
Другие версии бота:
* graph.py - Adaptive Rag
* graph_3.py — support_bot v3 (с поиском в интернете и поддержкой Small Talk)

## Evaluation в GigaLogger
Ниже приведены ноутбуки, показывающие как можно автоматически замерять качество
нашего бота:
1. [Генерация синтетического датасета из наших документов](evaluation/1_generate_dataset.ipynb)
2. [Загрузка этого датасета в GigaLogger](evaluation/2_gigalogger_create_dataset.ipynb)
3. [Оценка разных подходов к ответам на вопросы с помощью датасета и LLM судьи,
которая сравнивает ответы из датасета и ответы наших LLM цепочек](evaluation/3_evaluation.ipynb)