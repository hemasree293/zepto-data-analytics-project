from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from typing import TypedDict
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb


class SupportState(TypedDict):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float
    
class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    
def classify_intent(state: SupportState) -> SupportState:
    query = state["query"].lower()

    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours",
    ]

    if any(keyword in query for keyword in policy_keywords):
        state["intent"] = "policy_question"
    else:
        state["intent"] = "general_question"

    return state

# Load the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to the existing ChromaDB database
chroma_path = Path(__file__).parent / "chroma_db"
client = chromadb.PersistentClient(path=str(chroma_path))

collection = client.get_collection(
    name="zepto_policies"
)


def retrieve_and_answer(state: SupportState) -> SupportState:
    query = state["query"]

    # Embed the user's query
    query_embedding = embedding_model.encode([query])[0]

    # Retrieve the top 3 most similar chunks
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )

    retrieved_ids = results["ids"][0]
    retrieved_documents = results["documents"][0]

    # Mock-mode answer using the top retrieved chunk
    top_chunk_snippet = retrieved_documents[0][:200]

    state["answer"] = (
        f"Based on the retrieved context: {top_chunk_snippet}"
    )

    state["sources"] = retrieved_ids
    state["confidence"] = 1.0

    return state

def direct_answer(state: SupportState) -> SupportState:
    # Fixed mock response for general questions
    state["answer"] = "I can only answer questions about Zepto policies right now."
    state["sources"] = []
    state["confidence"] = 1.0

    return state

# -----------------------------
# LangGraph StateGraph
# -----------------------------

def route_by_intent(state: SupportState) -> str:
    """Route the query based on the classified intent."""
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"


workflow = StateGraph(SupportState)

# Add the three required nodes
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

# Start with intent classification
workflow.set_entry_point("classify_intent")

# Conditional routing after classification
workflow.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    },
)

# Both branches finish the graph
workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

# Compile the graph
graph = workflow.compile()

def validate_response(state: SupportState) -> AnswerResponse:
    """Validate the final graph output against the required schema."""
    return AnswerResponse(
        answer=state["answer"],
        sources=state["sources"],
        confidence=state["confidence"],
    )
    
    