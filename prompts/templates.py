"""Prompt templates for RAG pipeline."""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re


# Default templates
DEFAULT_RAG_TEMPLATE = """Based on the following context, answer the question.
If the answer cannot be found in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""

DEFAULT_RAG_TEMPLATE_CN = """已知信息:
{context}

根据上述已知信息，简洁和专业的来回答用户的问题。
如果无法从中得到答案，请说"根据已知信息无法回答该问题"。
不允许在答案中添加编造成分。

问题: {question}

回答:"""

DEFAULT_QA_TEMPLATE = """You are a helpful assistant. Answer the following question based on the provided information.

Information:
{context}

Question: {question}

Provide a clear and concise answer:"""


@dataclass
class PromptTemplate:
    """Basic prompt template."""

    template: str
    input_variables: List[str]

    def format(self, **kwargs) -> str:
        """Format the template with provided variables.

        Args:
            **kwargs: Variable values.

        Returns:
            Formatted prompt string.
        """
        # Validate all required variables are provided
        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        return self.template.format(**kwargs)

    @classmethod
    def from_template(cls, template: str) -> "PromptTemplate":
        """Create PromptTemplate from template string.

        Args:
            template: Template string with {variable} placeholders.

        Returns:
            PromptTemplate instance.
        """
        # Extract variables from template
        variables = re.findall(r'\{(\w+)\}', template)
        return cls(template=template, input_variables=list(set(variables)))


class RAGPromptTemplate:
    """Specialized prompt template for RAG applications."""

    def __init__(
        self,
        template: Optional[str] = None,
        language: str = "en",
        include_sources: bool = True
    ):
        """Initialize RAG prompt template.

        Args:
            template: Custom template string.
            language: Language for default template ('en' or 'cn').
            include_sources: Whether to include source citations.
        """
        if template:
            self.template = template
        elif language == "cn":
            self.template = DEFAULT_RAG_TEMPLATE_CN
        else:
            self.template = DEFAULT_RAG_TEMPLATE

        self.language = language
        self.include_sources = include_sources

    def format(
        self,
        question: str,
        context_docs: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """Format prompt with question and retrieved documents.

        Args:
            question: User question.
            context_docs: List of retrieved documents with 'content' and optional 'source'.
            **kwargs: Additional template variables.

        Returns:
            Formatted prompt string.
        """
        # Format context from documents
        context_parts = []
        for i, doc in enumerate(context_docs):
            content = doc.get('content', '')
            source = doc.get('source', '')

            if self.include_sources and source:
                context_parts.append(f"[{i+1}] {content}\n(Source: {source})")
            else:
                context_parts.append(content)

        context = "\n\n".join(context_parts)

        return self.template.format(
            context=context,
            question=question,
            **kwargs
        )

    def format_with_results(
        self,
        question: str,
        retrieval_results: List,
        **kwargs
    ) -> str:
        """Format prompt using RetrievalResult objects.

        Args:
            question: User question.
            retrieval_results: List of RetrievalResult objects.
            **kwargs: Additional template variables.

        Returns:
            Formatted prompt string.
        """
        context_docs = [
            {
                'content': r.content,
                'source': r.source,
                'score': r.score
            }
            for r in retrieval_results
        ]
        return self.format(question, context_docs, **kwargs)


class QAPromptTemplate:
    """Question-answering prompt template with terminology support."""

    def __init__(
        self,
        template: Optional[str] = None,
        terminology_dict: Optional[Dict[str, str]] = None
    ):
        """Initialize QA prompt template.

        Args:
            template: Custom template string.
            terminology_dict: Dictionary of term -> explanation mappings.
        """
        self.template = template or DEFAULT_QA_TEMPLATE
        self.terminology_dict = terminology_dict or {}

    def extract_terminology(self, text: str) -> List[str]:
        """Extract known terminology from text.

        Args:
            text: Text to search for terminology.

        Returns:
            List of found terms.
        """
        found_terms = []
        text_lower = text.lower()

        for term in self.terminology_dict.keys():
            if term.lower() in text_lower:
                found_terms.append(term)

        return found_terms

    def format(
        self,
        question: str,
        context: str,
        include_terminology: bool = True,
        **kwargs
    ) -> str:
        """Format QA prompt.

        Args:
            question: User question.
            context: Retrieved context.
            include_terminology: Whether to include terminology explanations.
            **kwargs: Additional variables.

        Returns:
            Formatted prompt.
        """
        # Add terminology explanations
        terminology_context = ""
        if include_terminology:
            terms = self.extract_terminology(question)
            if terms:
                term_explanations = []
                for term in terms:
                    explanation = self.terminology_dict.get(term, "")
                    if explanation:
                        term_explanations.append(f"- {term}: {explanation}")

                if term_explanations:
                    terminology_context = "Terminology:\n" + "\n".join(term_explanations) + "\n\n"

        full_context = terminology_context + context

        return self.template.format(
            context=full_context,
            question=question,
            **kwargs
        )


class ConversationalPromptTemplate:
    """Prompt template for multi-turn conversations."""

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        template: Optional[str] = None,
        max_history: int = 5
    ):
        """Initialize conversational prompt template.

        Args:
            system_prompt: System message for the conversation.
            template: Custom template for the user turn.
            max_history: Maximum number of history turns to include.
        """
        self.system_prompt = system_prompt
        self.template = template or DEFAULT_RAG_TEMPLATE
        self.max_history = max_history

    def format(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Format conversational prompt with history.

        Args:
            question: Current user question.
            context: Retrieved context.
            history: List of previous turns [{"role": "user/assistant", "content": "..."}].
            **kwargs: Additional variables.

        Returns:
            Dictionary with system message and messages list.
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add history (limited to max_history)
        if history:
            for turn in history[-self.max_history:]:
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", "")
                })

        # Format current question with context
        user_content = self.template.format(
            context=context,
            question=question,
            **kwargs
        )

        messages.append({"role": "user", "content": user_content})

        return {
            "system": self.system_prompt,
            "messages": messages
        }

    def format_simple(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Format as a simple string (for non-chat models).

        Args:
            question: Current question.
            context: Retrieved context.
            history: Conversation history.

        Returns:
            Formatted string prompt.
        """
        parts = [f"System: {self.system_prompt}\n"]

        if history:
            for turn in history[-self.max_history:]:
                role = turn.get("role", "user").capitalize()
                content = turn.get("content", "")
                parts.append(f"{role}: {content}\n")

        user_content = self.template.format(context=context, question=question)
        parts.append(f"User: {user_content}\n")
        parts.append("Assistant:")

        return "\n".join(parts)
