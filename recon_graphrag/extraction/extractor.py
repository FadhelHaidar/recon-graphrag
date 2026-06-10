"""LLM-based graph extraction from text."""

from recon_graphrag.extraction.parser import GraphExtractionParser
from recon_graphrag.extraction.prompts import SchemaPromptBuilder
from recon_graphrag.extraction.schema import GraphSchema
from recon_graphrag.extraction.types import GraphExtraction


class LLMGraphExtractor:
    def __init__(self, llm, prompt_builder: SchemaPromptBuilder | None = None):
        self.llm = llm
        self.prompt_builder = prompt_builder or SchemaPromptBuilder()
        self.parser = GraphExtractionParser()

    async def extract(self, text: str, schema: GraphSchema) -> GraphExtraction:
        prompt = self.prompt_builder.build_prompt(text=text, schema=schema)
        response = await self.llm.ainvoke(prompt)
        return self.parser.parse(response.content)
