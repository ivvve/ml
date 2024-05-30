"""
pip install langchain-experimental
pip install tabulate
"""
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI

import os
assert os.environ.get("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-pro")
df = pd.read_csv("./data/booksv_02.csv")

agent = create_pandas_dataframe_agent(
    llm,
    df,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    return_intermediate_steps=True,
    verbose=True, # 추론 과정 출력 설정
)

query = "어떤 제품의 ratings_count가 제일 높아?"
answer = agent.invoke(query)
print(f"{query=}")
print(f"{answer=}")


query = "가장 최근에 출간된 책은?"
answer = agent.invoke(query)
print(f"{query=}")
print(f"{answer=}")
