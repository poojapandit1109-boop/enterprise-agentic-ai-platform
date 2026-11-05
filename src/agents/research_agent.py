from src.config import get_settings
from src.llm.base import BaseLLMClient
from src.llm.factory import get_reliable_llm_client
from src.llm.prompts import get_prompt
from src.llm.structured_client import generate_structured_output
from src.llm.structured_outputs import ResearchOutput
from src.agents.research_mock_client import ResearchMockLLMClient
from src.orchestration.state import AgentResult


class ResearchAgent:
    """Research agent that generates validated structured findings."""

    def __init__(self, client: BaseLLMClient | None = None) -> None:
        if client is not None:
            self.client = client
        elif get_settings().llm_provider.strip().lower() == "mock":
            self.client = ResearchMockLLMClient()
        else:
            self.client = get_reliable_llm_client()

        self.prompt_template = get_prompt("research_agent")

    def run(self, topic: str) -> AgentResult:
        """Research a topic and return a validated agent result."""

        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("Research topic cannot be empty.")

        prompt = self.prompt_template.render(topic=topic.strip())

        result = generate_structured_output(
            client=self.client,
            prompt=prompt,
            schema=ResearchOutput,
        )

        if not isinstance(result, ResearchOutput):
            raise TypeError(
                "Research agent received an unexpected output schema."
            )

        return {
            "agent_name": "ResearchAgent",
            "status": result.status,
            "output": result.model_dump(),
            "error": None,
        }