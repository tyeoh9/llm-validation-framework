"""
llm_validation_framework — composable LLM validation pipeline.

Primary interface:
    from llm_validation_framework import ValidationFramework, LLMProvider, Pipe

Agents (import individually as needed):
    from llm_validation_framework import ToxicityAgent, PrivacyAgent
    from llm_validation_framework import AccuracyAgent, RelevancyAgent, BiasAgent

RAG support (bring your own RAG implementation):
    from llm_validation_framework import RAGProvider
    # Pass any retriever with .invoke(query: str) -> List[Document]
    rag = RAGProvider(your_retriever)
    AccuracyAgent(rag=rag)

Types:
    from llm_validation_framework.models import EvaluationResult, ValidationSummary
"""

from llm_validation_framework.validation_framework import ValidationFramework
from llm_validation_framework.llm_provider import LLMProvider, DeepEvalLLMProvider
from llm_validation_framework.pipe import Pipe
from llm_validation_framework.toxicity_agent import ToxicityAgent
from llm_validation_framework.privacy_agent import PrivacyAgent
from llm_validation_framework.accuracy_agent import AccuracyAgent
from llm_validation_framework.relevancy_agent import RelevancyAgent
from llm_validation_framework.bias_agent import BiasAgent
from llm_validation_framework.online_data import OnlineData
from llm_validation_framework.rag_provider import RAGProvider
from llm_validation_framework.models import EvaluationResult, GuardrailSummary, ValidationSummary

__version__ = "0.1.0"

__all__ = [
    # Core orchestration
    "ValidationFramework",
    "LLMProvider",
    "DeepEvalLLMProvider",
    "Pipe",
    # Agents
    "ToxicityAgent",
    "PrivacyAgent",
    "AccuracyAgent",
    "RelevancyAgent",
    "BiasAgent",
    "OnlineData",
    "RAGProvider",
    # Types
    "EvaluationResult",
    "GuardrailSummary",
    "ValidationSummary",
]
