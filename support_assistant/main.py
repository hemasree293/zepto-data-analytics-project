from fastapi import FastAPI
from pydantic import BaseModel

from support_assistant.graph import graph, validate_response


app = FastAPI(title="Zepto Support Assistant")


class AskRequest(BaseModel):
    query: str


@app.post("/ask")
def ask(request: AskRequest):
    state = {
        "query": request.query,
        "intent": "",
        "answer": "",
        "sources": [],
        "confidence": 0.0,
    }

    result = graph.invoke(state)

    validated = validate_response(result)

    return validated
