from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

try:
    llm = ChatOpenAI(model="gpt-5.6-luna", max_tokens=10)
    response = llm.invoke("Say OK")
    print("✅ API key works! Response:", response.content)

except Exception as e:
    print("❌ API check failed:", e)