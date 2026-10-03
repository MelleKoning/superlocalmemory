# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Summarizer — Mode A heuristic, Mode B/C the configured LLM provider.

Generates cluster summaries and search synthesis. All LLM failures
fall back to heuristic silently — never crashes the caller.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class Summarizer:
    """Generate summaries using heuristic or LLM based on mode."""

    def __init__(self, config) -> None:
        self._config = config
        self._mode = config.mode.value if hasattr(config.mode, 'value') else str(config.mode)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize_cluster(self, members: list[dict]) -> str:
        """Generate a human-readable cluster summary.

        Args:
            members: List of dicts with 'content' key.

        Returns:
            Summary string (2-3 sentences).
        """
        texts = [m.get("content", "") for m in members if m.get("content")]
        if not texts:
            return "Empty cluster."
        if self._mode in ("b", "c") and self._has_llm():
            try:
                prompt = self._cluster_prompt(texts[:10])
                return self._call_llm(prompt, max_tokens=150)
            except Exception as exc:
                logger.warning("LLM cluster summary failed, using heuristic: %s", exc)
        return self._heuristic_summary(texts[:5])

    def synthesize_answer(self, query: str, facts: list[dict]) -> str:
        """Generate a synthesized answer from query + retrieved facts.

        Returns empty string in Mode A (no LLM available).
        """
        if self._mode == "a" or not self._has_llm():
            return ""
        texts = [f.get("content", "") for f in facts if f.get("content")]
        if not texts:
            return ""
        try:
            prompt = self._synthesis_prompt(query, texts[:8])
            return self._call_llm(prompt, max_tokens=250)
        except Exception as exc:
            logger.warning("LLM synthesis failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Heuristic (Mode A — always available)
    # ------------------------------------------------------------------

    def _heuristic_summary(self, texts: list[str]) -> str:
        """First sentence from top-3 texts, joined."""
        sentences = []
        for text in texts[:3]:
            first = self._first_sentence(text)
            if first and first not in sentences:
                sentences.append(first)
        return " ".join(sentences)[:300] if sentences else "No summary available."

    @staticmethod
    def _first_sentence(text: str) -> str:
        """Extract first sentence (up to period, question mark, or 100 chars)."""
        text = text.strip()
        match = re.match(r'^(.+?[.!?])\s', text)
        if match:
            return match.group(1).strip()
        return text[:100].strip()

    # ------------------------------------------------------------------
    # LLM calls (Mode B/C)
    # ------------------------------------------------------------------

    def _backbone(self):
        """The configured provider, and only the configured provider.

        Mode B is the local Ollama in ``config.llm``; Mode C is whatever
        cloud provider the user chose. There is no built-in host and no key
        borrowed from the environment for another provider: a provider's key
        only ever goes to that provider. The backbone sends through the
        outbound gate, so memory text bound for another machine is screened.
        """
        from superlocalmemory.llm.backbone import LLMBackbone

        llm_config = getattr(self._config, "llm", None)
        if llm_config is None or not getattr(llm_config, "provider", ""):
            return None
        try:
            backbone = LLMBackbone(llm_config)
        except ValueError as exc:
            logger.warning("Summarizer: LLM provider not usable: %s", exc)
            return None
        return backbone if backbone.is_available() else None

    def _has_llm(self) -> bool:
        """True in Mode B/C when the configured provider is ready."""
        return self._mode in ("b", "c") and self._backbone() is not None

    def _call_llm(self, prompt: str, max_tokens: int = 200) -> str:
        """One request to the configured provider; raises if there is none.

        A single attempt: every caller falls back to the heuristic, and the
        store path must not wait out retries against a provider that is down.
        """
        backbone = self._backbone()
        if backbone is None:
            raise RuntimeError("no LLM provider configured")
        return backbone.generate(
            prompt, temperature=0.3, max_tokens=max_tokens, attempts=1,
        ).strip()

    # ------------------------------------------------------------------
    # Prompt templates
    # ------------------------------------------------------------------

    @staticmethod
    def _cluster_prompt(texts: list[str]) -> str:
        numbered = "\n".join(f"{i+1}. {t[:200]}" for i, t in enumerate(texts))
        return (
            "Summarize the following related memories in 2-3 concise sentences. "
            "Focus on the common theme and key facts.\n\n"
            f"Memories:\n{numbered}\n\n"
            "Summary:"
        )

    @staticmethod
    def _synthesis_prompt(query: str, texts: list[str]) -> str:
        numbered = "\n".join(f"- {t[:200]}" for t in texts)
        return (
            f"Based on these stored memories, answer the question concisely.\n\n"
            f"Question: {query}\n\n"
            f"Relevant memories:\n{numbered}\n\n"
            "Answer (2-3 sentences):"
        )
