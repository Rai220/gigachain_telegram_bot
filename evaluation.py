import csv
import os

from dotenv import find_dotenv, load_dotenv
from langchain import hub
from langchain.evaluation import CotQAEvalChain
from langchain_community.chat_models import GigaChat
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.evaluation import evaluate, evaluate_comparative

load_dotenv(find_dotenv())
os.environ["LANGCHAIN_PROJECT"] = "gigachain_telegram_bot"

COT_PROMPT = PromptTemplate(
    input_variables=["query", "context", "result"],
    template="""Ты учитель, оценивающий тест.

Тебе дан вопрос, корректный ответ и ответ студента. Тебе нужно оценить ответ студента как ПРАВИЛЬНЫЙ или НЕПРАВИЛЬНЫЙ, основываясь на корректном ответе.
Опиши пошагово своё рассуждение, чтобы убедиться, что твой вывод правильный. Избегай просто указывать правильный ответ с самого начала.

Вот базовая информация из конкретной области этого теста:
GigaChat - это большая языковая модель (LLM) от Сбера.
GigaChat API (апи) - это API для взаимодействия с GigaChat по HTTP с помощью REST запросов.
GigaChain - это SDK на Python для работы с GigaChat API. Русскоязычный форк библиотеки LangChain.
GigaGraph - это дополнение для GigaChain, который позволяет создавать мультиагентные системы, описывая их в виде графов.
Обучение GigaChat выполняется командой разработчиков. Дообучение и файнтюнинг для конечных пользователей на данный момент не доступно.
Для получения доступа к API нужно зарегистрироваться на developers.sber.ru и получить авторизационные данные.

Опирайся на эту базовую информацию, если тебе не хватает информации для проверки теста.

Пример формата:
QUESTION: здесь вопрос
TRUE ANSWER: здесь корректный ответ
STUDENT ANSWER: здесь ответ студента
EXPLANATION: пошаговое рассуждение здесь
GRADE: CORRECT или INCORRECT здесь

Тебе будем дан только один ответ студента, не несколько.
Оценивай ответ студента ТОЛЬКО на основе их фактической точности. 
Игнорируй различия в пунктуации и формулировках между ответом студента и правильным ответом.
Ответ студента может содержать больше информации, чем правильный ответ, если в нём нет противоречивых утверждений, то он корректен. Начнём!

QUESTION: "{query}"
TRUE ANSWER: "{context}"
STUDENT ANSWER: "{result}"
EXPLANATION:""",
)


eval_llm = ChatOpenAI(temperature=0, model="gpt-4o-2024-08-06")

# Create dataset
client = Client()
dataset_name = "gigachain_telegram_bot"

if not client.has_dataset(dataset_name=dataset_name):
    with open("./evaluation/dataset_cleared.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        inputs = []
        outputs = []

        for row in reader:
            inputs.append(row["question"])
            outputs.append(row["reference_answer"])

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="",
    )
    top_k = 10
    client.create_examples(
        inputs=[{"question": q} for q in inputs][slice(top_k)],
        outputs=[{"answer": a} for a in outputs][slice(top_k)],
        dataset_id=dataset.id,
    )

gigachat_lite = llm = GigaChat(
    model="GigaChat",
    temperature=0.00001,
    profanity_check=False,
    max_tokens=8000,
)

gigachat_pro = GigaChat(
    model="GigaChat-Pro",
    temperature=0.00001,
    profanity_check=False,
    max_tokens=8000,
)


def predict_gigachat_lite(example: dict):
    response = gigachat_lite.invoke(example["question"])
    return {"answer": response.content}


def predict_gigachat_pro(example: dict):
    response = gigachat_pro.invoke(example["question"])
    return {"answer": response.content}


def cot_evaluator(run, example) -> dict:
    input_question = example.inputs["question"]
    reference = example.outputs["answer"]
    prediction = run.outputs["answer"]

    cot_qa_chain = CotQAEvalChain.from_llm(llm=eval_llm, prompt=COT_PROMPT)

    response = cot_qa_chain.invoke(
        {
            "query": input_question,
            "context": reference,
            "result": prediction,
        }
    )
    parsed_response = cot_qa_chain._prepare_output(response)

    return {"key": "cot_qa_score", "score": parsed_response["score"]}


def evaluate_pairwise(runs: list, example) -> dict:
    """
    A simple evaluator for pairwise answers based on LCEL code solutions
    """
    scores = {}

    for i, run in enumerate(runs):
        scores[run.id] = i

    question = example.inputs["question"]
    answer_a = runs[0].outputs["answer"]
    answer_b = runs[1].outputs["answer"]

    grade_prompt = hub.pull("rlm/pairwise-evaluation-lcel-answers")
    answer_grader = grade_prompt | eval_llm

    response = answer_grader.invoke(
        {"question": question, "answer_a": answer_a, "answer_b": answer_b}
    )
    score = response["Preference"]

    if score == 1:  # Assistant A is preferred
        scores[runs[0].id] = 1
        scores[runs[1].id] = 0
    elif score == 2:  # Assistant B is preferred
        scores[runs[0].id] = 0
        scores[runs[1].id] = 1
    else:
        scores[runs[0].id] = 0
        scores[runs[1].id] = 0

    return {"key": "ranked_preference", "scores": scores}


sample_question = "Что такое gigachain"
print(gigachat_lite.invoke(sample_question))
print(gigachat_pro.invoke(sample_question))


experiment_gigachat_lite = evaluate(
    predict_gigachat_lite,
    data=dataset_name,
    evaluators=[cot_evaluator],
    experiment_prefix="rag-qa-gigachat-lite",
    max_concurrency=4,
    num_repetitions=1,
)

experiment_gigachat_pro = evaluate(
    predict_gigachat_pro,
    data=dataset_name,
    evaluators=[cot_evaluator],
    experiment_prefix="rag-qa-gigachat-pro",
    max_concurrency=4,
    num_repetitions=1,
)


evaluate_comparative(
    [
        experiment_gigachat_lite.experiment_name,
        experiment_gigachat_pro.experiment_name,
    ],
    evaluators=[evaluate_pairwise],
)
