from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith.evaluation import evaluate, LangChainStringEvaluator

# Target task definition
prompt = ChatPromptTemplate.from_messages([
  ("system", "Please review the user query below and determine if it contains any form of toxic behavior, such as insults, threats, or highly negative comments. Respond with 'Toxic' if it does, and 'Not toxic' if it doesn't."),
  ("user", "{question}")
])
chat_model = ChatOpenAI(model="gpt-4o")
output_parser = StrOutputParser()

chain = prompt | chat_model | output_parser

# The name or UUID of the LangSmith dataset to evaluate on.
# Alternatively, you can pass an iterator of examples
data = "PomoGigaRegression"

# A string to prefix the experiment name with.
# If not provided, a random string will be generated.
experiment_prefix = "PomoGigaRegression"

# List of evaluators to score the outputs of target task
evaluators = [
  LangChainStringEvaluator("cot_qa")
]

# Evaluate the target task
results = evaluate(
  chain.invoke,
  data=data,
  evaluators=evaluators,
  experiment_prefix=experiment_prefix,
)