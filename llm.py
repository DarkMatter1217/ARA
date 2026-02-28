import os
from functools import lru_cache
from dotenv import load_dotenv

from cerebras.cloud.sdk import Cerebras
from groq import Groq

load_dotenv()

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not CEREBRAS_API_KEY:
    raise RuntimeError("CEREBRAS_API_KEY not set")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

_cerebras_client = Cerebras(api_key=CEREBRAS_API_KEY)
_groq_client = Groq(api_key=GROQ_API_KEY)


class CerebrasLLM:
    def __init__(self, model="gpt-oss-120b"):
        self.model = model

    def invoke(self, prompt: str):
        completion = _cerebras_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=1,
            max_completion_tokens=2048,
        )

        class Response:
            def __init__(self, text):
                self.content = text

        return Response(completion.choices[0].message.content)


class GroqLLM:
    def __init__(self, model="openai/gpt-oss-120b"):
        self.model = model

    def invoke(self, prompt: str):
        completion = _groq_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=1,
            max_tokens=2048,
        )

        class Response:
            def __init__(self, text):
                self.content = text

        return Response(completion.choices[0].message.content)


@lru_cache(maxsize=4)
def get_llm(agent_type: str = "verification"):
    if agent_type == "verification":
        return CerebrasLLM()
    elif agent_type == "summarization":
        return GroqLLM(model="moonshotai/kimi-k2-instruct")
    return GroqLLM()
