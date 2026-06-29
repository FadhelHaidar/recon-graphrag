"""DRIFT search: Dynamic Reasoning and Inference with Flexible Traversal.

Paper-aligned iterative DRIFT that begins with semantic community-report
retrieval, generates follow-up questions, performs local subgraph searches
for each follow-up, and reduces all gathered evidence into a final answer.

Pipeline:
1. Primer: embed query → vector search community reports → LLM generates
   initial answer, score, and follow-up questions (strict JSON).
2. Traversal: breadth-first expansion of follow-up questions via local
   entity retrieval + LLM scoring, respecting depth/LLM-call/concurrency
   limits.
3. Reduction: sort completed actions by score, token-pack, LLM synthesizes
   final answer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Optional

from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.llm.base import BaseLLM
from recon_graphrag.models.types import SearchResult
from recon_graphrag.retrieval.base import BaseRetriever
from recon_graphrag.retrieval.citations import (
    resolve_chunk_citations,
    resolve_reference_citations,
)
from recon_graphrag.retrieval.community_levels import (
    CommunityLevelSelector,
    resolve_community_level,
)
from recon_graphrag.retrieval.drift_types import (
    DriftAction,
    DriftQueryState,
    DriftSearchConfig,
)
from recon_graphrag.retrieval.hybrid import HybridEntityRetriever, HybridRanker
from recon_graphrag.retrieval.local import _source_chunk_ids_from_result
from recon_graphrag.utils.tokens import (
    ApproximateTokenCounter,
    TokenCounter,
    pack_items,
    PackItem,
)


logger = logging.getLogger(__name__)


class _LLMCallLimitReached(RuntimeError):
    """Internal signal used to enforce a per-search hard call budget."""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PRIMER_PROMPT = """\
You are answering a question using community reports from a knowledge graph.

## Question
{query}
{conversation_history}
## Community Reports
{report_context}

## Instructions
1. Answer the question using ONLY information from the reports above.
2. Rate your confidence from 0 to 100.
3. Generate up to 3 follow-up questions that would help refine the answer.
4. Cite report IDs you used.
5. Return valid JSON only.

JSON format:
{{
  "answer": "Your initial answer...",
  "score": 75,
  "follow_ups": ["follow-up question 1", "follow-up question 2"],
  "report_ids": ["report:0:0"],
  "references": [
    {{"target_id": "entity-id", "target_type": "entity"}}
  ]
}}

If the reports contain no relevant information, set score to 0 and
answer to "No relevant information found."
"""

ACTION_PROMPT = """\
You are refining an answer using local graph context.

## Original Question
{query}
{conversation_history}
## Follow-up Question
{action_query}

## Parent Answer
{parent_answer}

## Local Context
{local_context}

## Instructions
1. Answer the follow-up question using ONLY the local context above.
2. Rate your confidence from 0 to 100.
3. Generate up to 2 additional follow-up questions if the context suggests
   new angles worth exploring.
4. Return valid JSON only.

JSON format:
{{
  "answer": "Your refined answer...",
  "score": 60,
  "follow_ups": ["new follow-up 1"],
  "references": [
    {{"target_id": "entity-or-claim-id", "target_type": "entity"}}
  ]
}}

If the context doesn't help, set score to 0.
"""

REDUCE_PROMPT = """\
You are synthesizing multiple partial answers into a comprehensive final answer.

## Original Question
{query}
{conversation_history}
## Partial Answers (sorted by confidence)
{action_context}

## Instructions
1. Combine these partial answers into one coherent, comprehensive answer.
2. Remove redundancy and resolve contradictions.
3. Preserve specific details and cite sources where appropriate.
4. If no partial answers contain relevant information, say so.

Final Answer:"""

REPAIR_PROMPT = """\
Your previous response had invalid JSON. Fix it and return valid JSON.

## Errors
{errors}

## Your previous response
{raw_content}

Return ONLY valid JSON. Do not include markdown fences.
"""


# ---------------------------------------------------------------------------
# Local fallback prompt
# ---------------------------------------------------------------------------

FALLBACK_ANSWER_PROMPT = """Answer the query using the local graph findings.

Query: {query}

=== Local Findings ===
{context}

Answer:"""


# ---------------------------------------------------------------------------
# DriftSearchRetriever
# ---------------------------------------------------------------------------


class DriftSearchRetriever(BaseRetriever):
    """DRIFT search: iterative traversal with community report priming."""

    def __init__(
        self,
        graph_store: GraphStore,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        retrieval_query: Optional[str] = None,
        reduce_prompt: Optional[str] = None,
        vector_index_name: str = "entity-embeddings",
        fulltext_index_name: str = "entity-names",
        graph_name: str = "entity-graph",
        config: DriftSearchConfig | None = None,
        token_counter: TokenCounter | None = None,
    ):
        self.graph_store = graph_store
        self.llm = llm
        self.embedder = embedder
        self.retrieval_query = retrieval_query
        self.reduce_prompt = reduce_prompt or REDUCE_PROMPT
        self.vector_index_name = vector_index_name
        self.fulltext_index_name = fulltext_index_name
        self.graph_name = graph_name
        self.config = config or DriftSearchConfig()
        self.counter = token_counter or ApproximateTokenCounter()
        self._retriever = self._build_retriever()

    def _build_retriever(self) -> HybridEntityRetriever:
        return HybridEntityRetriever(
            graph_store=self.graph_store,
            embedder=self.embedder,
            vector_index_name=self.vector_index_name,
            fulltext_index_name=self.fulltext_index_name,
            retrieval_query=self.retrieval_query,
            graph_name=self.graph_name,
            context_mode="drift",
        )

    async def search(
        self,
        query: str,
        top_k: int = 10,
        community_level: CommunityLevelSelector = None,
        query_vector: list[float] | None = None,
        effective_search_ratio: int = 1,
        query_params: dict | None = None,
        ranker: HybridRanker | str = "naive",
        alpha: float | None = None,
        synthesize_citation_metadata: bool = False,
        synthesis_metadata_keys: list[str] | None = None,
        synthesize_response: bool = True,
        conversation_history: str = "",
    ) -> SearchResult:
        """Run iterative DRIFT search.

        Args:
            query: User question.
            top_k: Number of entities for local follow-up retrieval.
            community_level: Which community level to use.
            query_vector: Optional precomputed query vector.
            effective_search_ratio: Over-fetch multiplier for entity retrieval.
            query_params: Optional dict forwarded to the hybrid entity retriever.
            ranker: Hybrid ranker: "naive" or "linear".
            alpha: Required for the "linear" ranker.
            synthesize_citation_metadata: Include resolved citation metadata
                in local follow-up context.
            synthesis_metadata_keys: Citation metadata keys to include.
            synthesize_response: If False, skip final reduction and return
                context + trace without synthesis.
            conversation_history: Optional conversation history string injected
                into prompts.
        """
        start = time.monotonic()
        state = DriftQueryState(query=query)
        call_lock = asyncio.Lock()

        config = self.config
        target_selector = (
            community_level if community_level is not None else config.community_level
        )
        resolved_level = resolve_community_level(
            self.graph_store, self.graph_name, target_selector
        )

        # --- Phase 1: Primer ---
        primer_result = await self._primer_phase(
            query,
            state,
            resolved_level,
            config.primer_top_k,
            conversation_history,
            query_vector,
            config,
            call_lock,
        )

        if primer_result.get("fallback"):
            return await self._fallback_search(
                query=query,
                state=state,
                top_k=top_k,
                effective_search_ratio=effective_search_ratio,
                query_params=query_params,
                ranker=ranker,
                alpha=alpha,
                synthesize_response=synthesize_response,
                reason=primer_result.get("fallback_reason", "missing_report_embeddings"),
                call_lock=call_lock,
            )

        # --- Phase 2: Traversal ---
        await self._traversal_phase(
            query, state, config, top_k, effective_search_ratio,
            query_params, ranker, alpha, conversation_history,
            synthesize_citation_metadata, synthesis_metadata_keys, call_lock,
            synthesize_response,
        )

        # --- Phase 3: Reduction ---
        elapsed_ms = int((time.monotonic() - start) * 1000)
        state.phase_tokens["elapsed_ms"] = elapsed_ms

        return await self._reduction_phase(
            query=query,
            state=state,
            synthesize_response=synthesize_response,
            conversation_history=conversation_history,
            call_lock=call_lock,
        )

    # ------------------------------------------------------------------
    # Phase 1: Primer
    # ------------------------------------------------------------------

    async def _primer_phase(
        self,
        query: str,
        state: DriftQueryState,
        resolved_level: int | None,
        primer_top_k: int,
        conversation_history: str,
        query_vector: list[float] | None,
        config: DriftSearchConfig,
        call_lock: asyncio.Lock,
    ) -> dict:
        """Embed query, search community reports, parse primer JSON."""
        if query_vector is None:
            try:
                query_vector = await self.embedder.async_embed_query(query)
            except Exception as e:
                state.failures.append(f"query embedding failed: {e}")
                return {"fallback": True, "fallback_reason": "query_embedding_failed"}

        if resolved_level is None:
            return {"fallback": True, "fallback_reason": "no_community_level"}

        try:
            reports = self.graph_store.vector_search_community_reports(
                query_vector=query_vector,
                graph_name=self.graph_name,
                top_k=primer_top_k,
                level=resolved_level,
            )
        except Exception as e:
            state.failures.append(f"report vector search failed: {e}")
            return {"fallback": True, "fallback_reason": "missing_report_embeddings"}

        if not reports:
            return {"fallback": True, "fallback_reason": "no_report_results"}

        state.primer_reports = reports

        report_context = self._format_primer_reports(reports)
        prompt = PRIMER_PROMPT.format(
            query=query,
            report_context=report_context,
            conversation_history=self._format_history(conversation_history),
        )

        try:
            response = await self._invoke_llm(
                prompt, state, config.max_llm_calls, call_lock, "primer"
            )
            data = await self._parse_json_strict(
                response.content,
                state,
                phase="primer",
                max_llm_calls=config.max_llm_calls,
                call_lock=call_lock,
            )
        except Exception as e:
            state.failures.append(f"primer LLM failed: {e}")
            return {"fallback": True, "fallback_reason": "primer_llm_failed"}

        primer_action = DriftAction(
            id="primer",
            parent_id=None,
            depth=0,
            query=query,
            answer=data.get("answer", ""),
            score=float(data.get("score", 0)),
            status="completed",
            context=report_context,
            follow_ups=data.get("follow_ups", [])[:3],
            references=data.get("references", []),
            report_ids=data.get("report_ids", []),
        )
        state.actions.append(primer_action)

        return {"fallback": False}

    # ------------------------------------------------------------------
    # Phase 2: Traversal
    # ------------------------------------------------------------------

    async def _traversal_phase(
        self,
        query: str,
        state: DriftQueryState,
        config: DriftSearchConfig,
        top_k: int,
        effective_search_ratio: int,
        query_params: dict | None,
        ranker: HybridRanker | str,
        alpha: float | None,
        conversation_history: str,
        synthesize_citation_metadata: bool,
        synthesis_metadata_keys: list[str] | None,
        call_lock: asyncio.Lock,
        synthesize_response: bool,
    ) -> None:
        """Breadth-first expansion of follow-up questions."""
        # Seed from primer follow-ups
        pending: list[DriftAction] = []
        action_counter = 0
        seen_questions = {self._normalize_question(query)}
        traversal_call_limit = max(
            config.max_llm_calls - (1 if synthesize_response else 0), 0
        )

        primer = state.actions[0] if state.actions else None
        if primer and primer.follow_ups and config.max_depth >= 1:
            for fq in primer.follow_ups[: config.max_followups]:
                normalized = self._normalize_question(fq)
                if not normalized or normalized in seen_questions:
                    continue
                seen_questions.add(normalized)
                action_counter += 1
                pending.append(
                    DriftAction(
                        id=f"action:{action_counter}",
                        parent_id="primer",
                        depth=1,
                        query=fq,
                    )
                )

        while pending:
            # Check limits
            if state.total_llm_calls >= traversal_call_limit:
                state.stopping_reason = "max_llm_calls_reached"
                break

            # Process batch up to concurrency
            available_calls = traversal_call_limit - state.total_llm_calls
            if available_calls <= 0:
                state.stopping_reason = "max_llm_calls_reached"
                break
            batch_size = min(config.action_concurrency, available_calls)
            batch = pending[:batch_size]
            pending = pending[batch_size:]

            tasks = [
                self._process_action(
                    action, query, state, config, top_k,
                    effective_search_ratio, query_params,
                    ranker, alpha, conversation_history,
                    synthesize_citation_metadata, synthesis_metadata_keys,
                    call_lock, traversal_call_limit,
                )
                for action in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for action, result in zip(batch, results):
                if isinstance(result, Exception):
                    action.status = "failed"
                    state.failures.append(f"action {action.id} failed: {result}")
                state.actions.append(action)

                # Queue follow-ups
                if (
                    action.status == "completed"
                    and action.score >= config.min_expand_score
                    and action.depth < config.max_depth
                    and action.follow_ups
                ):
                    for fq in action.follow_ups[: config.max_followups]:
                        # Deduplicate by normalized query
                        normalized = self._normalize_question(fq)
                        if normalized and normalized not in seen_questions:
                            seen_questions.add(normalized)
                            action_counter += 1
                            pending.append(
                                DriftAction(
                                    id=f"action:{action_counter}",
                                    parent_id=action.id,
                                    depth=action.depth + 1,
                                    query=fq,
                                )
                            )

        if not state.stopping_reason:
            state.stopping_reason = "all_actions_completed"

    async def _process_action(
        self,
        action: DriftAction,
        original_query: str,
        state: DriftQueryState,
        config: DriftSearchConfig,
        top_k: int,
        effective_search_ratio: int,
        query_params: dict | None,
        ranker: HybridRanker | str,
        alpha: float | None,
        conversation_history: str,
        synthesize_citation_metadata: bool,
        synthesis_metadata_keys: list[str] | None,
        call_lock: asyncio.Lock,
        llm_call_limit: int,
    ) -> None:
        """Process a single follow-up action."""
        # Local retrieval
        try:
            retriever_result = await self._retriever.search(
                query_text=action.query,
                top_k=top_k,
                effective_search_ratio=effective_search_ratio,
                query_params=query_params,
                ranker=ranker,
                alpha=alpha,
            )
        except Exception as e:
            action.status = "failed"
            state.failures.append(f"retrieval failed for {action.id}: {e}")
            return

        # Resolve chunk citations
        chunk_ids = _source_chunk_ids_from_result(retriever_result)
        if chunk_ids:
            try:
                action.citations = resolve_chunk_citations(
                    self.graph_store, self.graph_name, chunk_ids
                )
            except Exception:
                pass

        from recon_graphrag.retrieval.local import _format_entity_context

        local_context = _format_entity_context(
            retriever_result,
            citations=(action.citations if synthesize_citation_metadata else None),
            citation_metadata_keys=synthesis_metadata_keys,
            drift=True,
        )
        action.context = local_context

        # Get parent answer for context
        parent_answer = ""
        if action.parent_id:
            parent = next(
                (a for a in state.actions if a.id == action.parent_id), None
            )
            if parent:
                parent_answer = parent.answer

        # LLM scoring
        prompt = ACTION_PROMPT.format(
            query=original_query,
            action_query=action.query,
            parent_answer=parent_answer,
            local_context=local_context,
            conversation_history=self._format_history(conversation_history),
        )

        try:
            response = await self._invoke_llm(
                prompt, state, llm_call_limit, call_lock, "action"
            )
            data = await self._parse_json_strict(
                response.content,
                state,
                phase="action",
                max_llm_calls=llm_call_limit,
                call_lock=call_lock,
            )
            action.answer = data.get("answer", "")
            action.score = float(data.get("score", 0))
            action.follow_ups = data.get("follow_ups", [])[: config.max_followups]
            action.references = data.get("references", [])
            if action.references:
                try:
                    ref_citations = resolve_reference_citations(
                        self.graph_store, self.graph_name, action.references
                    )
                    action.citations = self._dedupe_citations(
                        action.citations + ref_citations
                    )
                except Exception as citation_error:
                    state.failures.append(
                        f"citation resolution failed for {action.id}: {citation_error}"
                    )
            action.status = "completed"
        except Exception as e:
            action.status = "failed"
            state.failures.append(f"action LLM failed for {action.id}: {e}")

    # ------------------------------------------------------------------
    # Phase 3: Reduction
    # ------------------------------------------------------------------

    async def _reduction_phase(
        self,
        query: str,
        state: DriftQueryState,
        synthesize_response: bool,
        conversation_history: str,
        call_lock: asyncio.Lock,
    ) -> SearchResult:
        """Sort actions by score, pack context, synthesize final answer."""
        completed = [a for a in state.actions if a.status == "completed" and a.answer]
        completed.sort(key=lambda a: (-a.score, a.id))

        # Pack action context into budget
        action_items = [
            PackItem(
                id=a.id,
                text=f"[Score {a.score:.0f}] Q: {a.query}\nA: {a.answer}",
                priority=a.score,
            )
            for a in completed
        ]
        packed = pack_items(
            action_items, self.config.reduce_budget_tokens, self.counter
        )

        action_context = "\n\n".join(i.text for i in packed.included)

        # Union citations from packed actions only
        all_citations = []
        seen_cite_keys: set[tuple[str, str]] = set()
        packed_ids = {i.id for i in packed.included}
        for a in completed:
            if a.id not in packed_ids:
                continue
            for cite in a.citations:
                key = (cite.document_id, cite.chunk_id)
                if key not in seen_cite_keys:
                    seen_cite_keys.add(key)
                    all_citations.append(cite)

        # Resolve references only from a packed primer action and from reports
        # the primer explicitly selected.
        report_refs = (
            self._extract_report_refs_from_state(state)
            if "primer" in packed_ids
            else []
        )
        if report_refs:
            try:
                ref_citations = resolve_reference_citations(
                    self.graph_store, self.graph_name, report_refs
                )
                for cite in ref_citations:
                    key = (cite.document_id, cite.chunk_id)
                    if key not in seen_cite_keys:
                        seen_cite_keys.add(key)
                        all_citations.append(cite)
            except Exception as e:
                state.failures.append(f"report citation resolution failed: {e}")

        if not synthesize_response:
            trace = self._build_trace(state)
            return SearchResult(
                query=query,
                mode="drift",
                answer="",
                context=action_context,
                citations=all_citations,
                metadata={
                    "synthesize_response": False,
                    "response_synthesis_skipped": True,
                    "drift_trace": trace,
                },
            )

        # Reduce LLM
        if not action_context:
            trace = self._build_trace(state)
            return SearchResult(
                query=query,
                mode="drift",
                answer="No relevant information found.",
                context="",
                citations=all_citations,
                metadata={"drift_trace": trace},
            )

        prompt = self.reduce_prompt.format(
            query=query,
            action_context=action_context,
            conversation_history=self._format_history(conversation_history),
        )

        try:
            response = await self._invoke_llm(
                prompt,
                state,
                self.config.max_llm_calls,
                call_lock,
                "reduce",
            )
            answer = response.content
        except _LLMCallLimitReached:
            state.stopping_reason = "max_llm_calls_reached"
            state.failures.append("reduce LLM skipped: max_llm_calls reached")
            answer = completed[0].answer if completed else "No relevant information found."
        except Exception as e:
            state.failures.append(f"reduce LLM failed: {e}")
            answer = "Failed to synthesize final answer."

        trace = self._build_trace(state)
        return SearchResult(
            query=query,
            mode="drift",
            answer=answer,
            context=action_context,
            citations=all_citations,
            metadata={"drift_trace": trace},
        )

    # ------------------------------------------------------------------
    # Local fallback
    # ------------------------------------------------------------------

    async def _fallback_search(
        self,
        query: str,
        state: DriftQueryState,
        top_k: int,
        effective_search_ratio: int,
        query_params: dict | None,
        ranker: HybridRanker | str,
        alpha: float | None,
        synthesize_response: bool,
        reason: str,
        call_lock: asyncio.Lock,
    ) -> SearchResult:
        """Fall back to local-style retrieval when report embeddings missing."""
        logger.warning("DRIFT falling back to local-style retrieval: %s", reason)
        retriever_result = await self._retriever.search(
            query_text=query,
            top_k=top_k,
            effective_search_ratio=effective_search_ratio,
            query_params=query_params,
            ranker=ranker,
            alpha=alpha,
        )

        from recon_graphrag.retrieval.local import _format_entity_context

        entity_context = _format_entity_context(retriever_result, drift=True)
        chunk_ids = _source_chunk_ids_from_result(retriever_result)
        citations = []
        if chunk_ids:
            try:
                citations = resolve_chunk_citations(
                    self.graph_store, self.graph_name, chunk_ids
                )
            except Exception:
                pass

        if not synthesize_response:
            trace = self._build_trace(state)
            trace["fallback_reason"] = reason
            return SearchResult(
                query=query,
                mode="drift",
                answer="",
                context=entity_context,
                citations=citations,
                metadata={
                    "synthesize_response": False,
                    "response_synthesis_skipped": True,
                    "drift_fallback_reason": reason,
                    "drift_trace": trace,
                },
            )

        prompt = FALLBACK_ANSWER_PROMPT.format(
            query=query,
            context=entity_context,
        )
        try:
            response = await self._invoke_llm(
                prompt,
                state,
                self.config.max_llm_calls,
                call_lock,
                "fallback",
            )
            answer = response.content
        except Exception as error:
            state.failures.append(f"fallback LLM failed: {error}")
            answer = "Failed to generate answer."

        trace = self._build_trace(state)
        trace["fallback_reason"] = reason
        return SearchResult(
            query=query,
            mode="drift",
            answer=answer,
            context=entity_context,
            citations=citations,
            metadata={
                "drift_fallback_reason": reason,
                "drift_trace": trace,
            },
        )

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    async def _parse_json_strict(
        self,
        content: str,
        state: DriftQueryState,
        *,
        phase: str,
        max_llm_calls: int,
        call_lock: asyncio.Lock,
    ) -> dict:
        """Parse and validate a DRIFT response, with exactly one repair attempt."""
        try:
            return self._validate_drift_json(self._decode_json(content), phase)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            repair_prompt = REPAIR_PROMPT.format(
                errors=str(error),
                raw_content=content[:2000],
            )
            response = await self._invoke_llm(
                repair_prompt,
                state,
                max_llm_calls,
                call_lock,
                f"{phase}_repair",
            )
            try:
                repaired = self._decode_json(response.content)
                return self._validate_drift_json(repaired, phase)
            except (ValueError, TypeError, json.JSONDecodeError) as repaired_error:
                raise ValueError(
                    f"{phase} JSON remained invalid after one repair: {repaired_error}"
                ) from repaired_error

    def _decode_json(self, content: str) -> dict:
        content = content.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if fenced:
            content = fenced.group(1).strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = self._extract_json_object(content)
        if not isinstance(data, dict) or not data:
            raise ValueError("response is not a non-empty JSON object")
        return data

    @staticmethod
    def _validate_drift_json(data: dict, phase: str) -> dict:
        answer = data.get("answer")
        score = data.get("score")
        follow_ups = data.get("follow_ups")
        if not isinstance(answer, str):
            raise ValueError("answer must be a string")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("score must be numeric")
        if not 0 <= float(score) <= 100:
            raise ValueError("score must be between 0 and 100")
        if not isinstance(follow_ups, list) or not all(
            isinstance(question, str) and question.strip()
            for question in follow_ups
        ):
            raise ValueError("follow_ups must be a list of non-empty strings")
        references = data.get("references", [])
        if not isinstance(references, list) or not all(
            isinstance(ref, dict)
            and str(ref.get("target_id", "")).strip()
            and ref.get("target_type") in {"entity", "relationship", "claim"}
            for ref in references
        ):
            raise ValueError("references must contain typed target_id objects")
        report_ids = data.get("report_ids", [])
        if phase == "primer" and (
            not isinstance(report_ids, list)
            or not all(isinstance(report_id, str) for report_id in report_ids)
        ):
            raise ValueError("report_ids must be a list of strings")
        data["score"] = float(score)
        data["follow_ups"] = [question.strip() for question in follow_ups]
        data["references"] = references
        data["report_ids"] = report_ids if isinstance(report_ids, list) else []
        return data

    async def _invoke_llm(
        self,
        prompt: str,
        state: DriftQueryState,
        max_llm_calls: int,
        call_lock: asyncio.Lock,
        phase: str,
    ):
        async with call_lock:
            if state.total_llm_calls >= max_llm_calls:
                raise _LLMCallLimitReached("max_llm_calls reached")
            state.total_llm_calls += 1
        response = await self.llm.ainvoke(prompt)
        state.phase_tokens[phase] = state.phase_tokens.get(phase, 0) + self.counter.count(
            prompt + response.content
        )
        return response

    @staticmethod
    def _extract_json_object(content: str) -> dict:
        """Extract the first JSON object from a response with surrounding prose."""
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", content):
            try:
                data, _ = decoder.raw_decode(content[match.start() :])
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue
        return {}

    @staticmethod
    def _normalize_question(question: str) -> str:
        return " ".join(re.sub(r"[^\w]+", " ", question.casefold()).split())

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_primer_reports(reports: list[dict]) -> str:
        parts = []
        for r in reports:
            rid = r.get("id", "?")
            level = r.get("level", "?")
            report_text = r.get("report_text", "")
            parts.append(f"Report {rid} (level {level}):\n{report_text}")
        return "\n\n".join(parts) if parts else "No community reports available."

    @staticmethod
    def _format_history(history: str) -> str:
        if not history:
            return ""
        return f"\n## Conversation History\n{history}\n"

    # ------------------------------------------------------------------
    # Citation extraction
    # ------------------------------------------------------------------

    def _extract_report_refs_from_state(self, state: DriftQueryState) -> list[dict]:
        """Extract references from primer reports via JSON or [refs: ...] text."""
        refs: list[dict] = []
        seen: set[tuple[str, str]] = set()
        allowed_types = {"entity", "relationship", "claim"}

        primer = next((action for action in state.actions if action.id == "primer"), None)
        selected_report_ids = set(primer.report_ids if primer else [])
        available_report_ids = {
            str(report.get("id")) for report in state.primer_reports if report.get("id")
        }
        selected_report_ids &= available_report_ids
        if not selected_report_ids:
            selected_report_ids = available_report_ids

        for ref in primer.references if primer else []:
            key = (str(ref.get("target_type", "")), str(ref.get("target_id", "")))
            if key[0] in allowed_types and key[1] and key not in seen:
                seen.add(key)
                refs.append({"target_type": key[0], "target_id": key[1]})

        for report in state.primer_reports:
            if str(report.get("id")) not in selected_report_ids:
                continue
            report_json = report.get("report_json", "")
            json_refs: list[dict] = []
            if report_json:
                try:
                    from recon_graphrag.retrieval.global_search import (
                        GlobalSearchRetriever,
                    )

                    json_refs = GlobalSearchRetriever._extract_report_json_refs(report_json)
                    for ref in json_refs:
                        key = (ref["target_type"], ref["target_id"])
                        if key not in seen and ref["target_type"] in allowed_types:
                            seen.add(key)
                            refs.append(ref)
                except Exception as error:
                    state.failures.append(
                        f"report reference JSON failed for {report.get('id')}: {error}"
                    )
            if not json_refs:
                report_text = report.get("report_text", "")
                if report_text:
                    try:
                        from recon_graphrag.retrieval.global_search import (
                            GlobalSearchRetriever,
                        )

                        text_refs = GlobalSearchRetriever._extract_report_text_refs(report_text)
                        for ref in text_refs:
                            key = (ref["target_type"], ref["target_id"])
                            if key not in seen and ref["target_type"] in allowed_types:
                                seen.add(key)
                                refs.append(ref)
                    except Exception:
                        pass

        return refs

    @staticmethod
    def _dedupe_citations(citations: list) -> list:
        seen: set[tuple[str, str, int | None, int | None]] = set()
        deduped = []
        for citation in citations:
            key = (
                citation.document_id,
                citation.chunk_id,
                citation.page_start,
                citation.page_end,
            )
            if key not in seen:
                seen.add(key)
                deduped.append(citation)
        return deduped

    # ------------------------------------------------------------------
    # Trace
    # ------------------------------------------------------------------

    @staticmethod
    def _build_trace(state: DriftQueryState) -> dict:
        return {
            "primer_report_ids": [r.get("id") for r in state.primer_reports],
            "generated_questions": [
                a.query for a in state.actions if a.parent_id == "primer"
            ],
            "actions": [
                {
                    "id": a.id,
                    "parent_id": a.parent_id,
                    "depth": a.depth,
                    "query": a.query,
                    "score": a.score,
                    "status": a.status,
                }
                for a in state.actions
            ],
            "stopping_reason": state.stopping_reason,
            "total_llm_calls": state.total_llm_calls,
            "phase_tokens": dict(state.phase_tokens),
            "failures": list(state.failures),
        }
