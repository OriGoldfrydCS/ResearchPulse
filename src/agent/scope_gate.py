"""
Scope Gate - Classifies user messages to enforce arXiv-only paper discovery scope.

ResearchPulse is a focused paper-finding assistant for arXiv.
This module ensures the chat stays on-topic by classifying every user
message before it reaches the heavy agent pipeline.

Classification:
    IN_SCOPE                  → proceed to normal agent flow
    OUT_OF_SCOPE_ARXIV_ONLY   → research-adjacent but not arXiv-accessible
    OUT_OF_SCOPE_GENERAL      → completely unrelated to research paper discovery
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Scope classification enum
# =============================================================================

class ScopeClass(str, Enum):
    """Result of classifying a user message."""
    IN_SCOPE = "IN_SCOPE"
    OUT_OF_SCOPE_ARXIV_ONLY = "OUT_OF_SCOPE_ARXIV_ONLY"
    OUT_OF_SCOPE_GENERAL = "OUT_OF_SCOPE_GENERAL"


# =============================================================================
# Classification result
# =============================================================================

class ScopeResult:
    """Container for a scope-gate classification decision."""

    __slots__ = ("scope", "reason", "suggested_rewrite", "response")

    def __init__(
        self,
        scope: ScopeClass,
        reason: str,
        suggested_rewrite: Optional[str] = None,
        response: Optional[str] = None,
    ):
        self.scope = scope
        self.reason = reason
        self.suggested_rewrite = suggested_rewrite
        self.response = response

    def __repr__(self) -> str:
        return f"ScopeResult(scope={self.scope.value}, reason={self.reason!r})"


# =============================================================================
# Response templates (constants – consistent tone, never hallucinate sources)
# =============================================================================

RESPONSE_OUT_OF_SCOPE_GENERAL = (
    "arXiv primarily provides papers in fields such as computer science, "
    "mathematics, physics, statistics, quantitative biology, quantitative "
    "finance, and related technical domains.\n\n"
    "The requested topic does not appear to fall within arXiv's coverage, "
    "and ResearchPulse cannot assist with this topic in its current version.\n\n"
    "If you believe your topic is related to one of these fields, try rephrasing "
    "your query with more specific scientific keywords."
)

RESPONSE_OUT_OF_SCOPE_ARXIV_ONLY = (
    "Right now I can only search and summarize papers on arXiv. "
    "If you want, I can look for arXiv preprints related to your topic. "
    "Tell me the keywords (and optional category like cs.CL, cs.LG, stat.ML)."
)

RESPONSE_MISSING_TOPIC = (
    "I can help you find arXiv papers, but I need a topic. "
    "What keywords should I use (and any category or time window)?"
)

RESPONSE_NEED_ARXIV_LINK = (
    "I can summarize arXiv papers. Please provide an arXiv link or ID "
    "(e.g. arXiv:2301.00001 or https://arxiv.org/abs/2301.00001)."
)

RESPONSE_NON_ARXIV_VENUE = (
    "I can't search that venue directly, but I can look for related "
    "preprints on arXiv. What keywords should I use?"
)


# =============================================================================
# Keyword / pattern banks (lowercase, compiled once)
# =============================================================================

# ---- In-scope signals -------------------------------------------------------
_ARXIV_PATTERNS = [
    r"arxiv",
    r"arxiv\.org",
    r"\d{4}\.\d{4,5}",                       # arXiv ID like 2301.12345
    r"paper[s]?",
    r"preprint[s]?",
    r"find\s.*paper",
    r"search\s.*paper",
    r"recent\s.*paper",
    r"latest\s.*paper",
    r"new\s.*paper",
    r"top\s.*paper",
    r"show\s.*paper",
    r"get\s.*paper",
    r"fetch\s.*paper",
    r"look\s.*paper",
    r"recommend\s.*paper",
    r"suggest\s.*paper",
    r"summarize\s.*paper",
    r"summarise\s.*paper",
    r"abstract",
    r"research\s.*interest",
    r"research\s.*topic",
    r"what.s new",
    r"track\s.*author",
    r"track\s.*keyword",
    r"alert\s+me",
    r"notify\s+me",
    r"reading\s*list",
    r"diffusion\s*model",
    r"neural\s*network",
    r"deep\s*learning",
    r"machine\s*learning",
    r"reinforcement\s*learning",
    r"computer\s*vision",
    r"natural\s*language",
    r"cs\.[A-Z]{2}",                           # arXiv category codes
    r"stat\.\w+",
    r"math\.\w+",
    r"eess\.\w+",
    r"q-bio\.\w+",
    r"q-fin\.\w+",
    r"quant-ph",
    r"hep-\w+",
    r"cond-mat",
    r"astro-ph",
]

# ResearchPulse operational features that are always in-scope
_OPERATIONAL_KEYWORDS = [
    "colleague", "colleagues", "share", "sharing",
    "email", "inbox", "reminder", "calendar",
    "delivery", "policy", "setting", "settings",
    "profile", "schedule", "run", "execute",
    "report", "artifact", "reading list",
    "category", "categories", "subscribe",
    "health", "status",
]

# ---- Out-of-scope: non-arXiv venues -----------------------------------------
_NON_ARXIV_VENUES = [
    "google scholar", "pubmed", "ieee", "acm",
    "scopus", "web of science", "nature", "science",
    "springer", "elsevier", "wiley", "jstor",
    "semantic scholar", "crossref", "dblp",
    "biorxiv", "medrxiv", "ssrn", "repec",
]

# ---- Out-of-scope: general requests -----------------------------------------
_GENERAL_OFF_TOPIC = [
    r"write\s+(my|me|a)\s+(code|email|essay|homework|report|letter|resume|cv)",
    r"help\s+me\s+(write|code|program|debug|fix)",
    r"explain\s+(to\s+me\s+)?(what|how|why)\s+(?!.*paper)",
    r"tell\s+me\s+a\s+(joke|story|fact)",
    r"what\s+is\s+the\s+(weather|time|date|news)",
    r"who\s+(is|are|was|were)\b",
    r"recipe[s]?",
    r"play\s+(a\s+)?(game|song|music)",
    r"translate\s",
    r"personal\s+advice",
    r"relationship",
    r"movie[s]?",
    r"travel",
    r"restaurant[s]?",
    r"shopping",
    r"buy\s",
    r"price\s",
    r"stock\s+(market|price)",
    r"crypto",
    r"bitcoin",
    r"sports?(\s|$)",
    r"score[s]?\s",
    r"celebrity",
    r"gossip",
    r"horoscope",
    # Broad non-scientific topics that don't map to arXiv categories
    r"\banimals?\b",
    r"\bpets?\b",
    r"\bcooking\b",
    r"\bfood\b",
    r"\bfashion\b",
    r"\bclothing\b",
    r"\bgardening\b",
    r"\bfitness\b",
    r"\bworkout[s]?\b",
    r"\bdiet(?:ing|s)?\b",
    r"\bmusic\b",
    r"\bsong[s]?\b",
    r"\bdancing?\b",
    r"\bpoetry\b",
    r"\bnovels?\b(?!\s*(?:method|approach|algorithm|architecture|framework|technique|model|loss|representation|scheme|formulation|contribution|idea|metric|design|mechanism|strategy|solution|way|pipeline))",
    r"\bhistory\s+of\b",
    r"\bpolitics?\b",
    r"\belection[s]?\b",
    r"\breligion[s]?\b",
    r"\bastrology\b",
    r"\bcraft[s]?\b",
    r"\bDIY\b",
    r"\bhobby|hobbies\b",
    r"\bvideo\s*game[s]?\b",
    r"\bgaming\b",
    r"\bfurniture\b",
    r"\breal\s+estate\b",
    r"\bmortgage\b",
    r"\binsurance\b",
    r"\blegal\s+advice\b",
    r"\blawyer[s]?\b",
    r"\bsoccer\b",
    r"\bfootball\b",
    r"\bbasketball\b",
    r"\bbaseball\b",
    r"\btennis\b",
    r"\bcinema\b",
    r"\btv\s+show[s]?\b",
    r"\bcelebrities\b",
]

# Compile once for performance
_ARXIV_RE = re.compile("|".join(_ARXIV_PATTERNS), re.IGNORECASE)
_GENERAL_RE = re.compile("|".join(_GENERAL_OFF_TOPIC), re.IGNORECASE)

# Strong arXiv signals that should override an off-topic match — these prove
# the user is actually talking about arXiv / scientific research, not the
# off-topic subject itself.  Generic words like "paper" or "research" are
# intentionally excluded so that "find papers on cooking recipes" still
# triggers the off-topic gate.
_STRONG_ARXIV_PATTERNS = [
    r"arxiv",
    r"arxiv\.org",
    r"\d{4}\.\d{4,5}",                       # arXiv ID like 2301.12345
    r"preprint[s]?",
    r"cs\.[A-Z]{2}",
    r"stat\.\w+",
    r"math\.\w+",
    r"eess\.\w+",
    r"q-bio\.\w+",
    r"q-fin\.\w+",
    r"quant-ph",
    r"hep-\w+",
    r"cond-mat",
    r"astro-ph",
]
_STRONG_ARXIV_RE = re.compile("|".join(_STRONG_ARXIV_PATTERNS), re.IGNORECASE)

# Scientific / technical terms that prove the user is asking about research,
# even if the query also contains an off-topic keyword (e.g. "animal
# locomotion in robotics").  These override the off-topic gate in step 1.
_SCIENTIFIC_OVERRIDE_WORDS = {
    "algorithm", "algorithms", "architecture", "benchmark", "classification",
    "contrastive", "cnn", "dataset", "detection", "diffusion", "embedding",
    "evaluation", "federated", "fine-tuning", "finetune", "gan",
    "generative", "gradient", "inference", "llm", "locomotion", "model",
    "models", "multimodal", "multi-modal", "network", "neural", "optimization",
    "pretrain", "pretraining", "regression", "reinforcement", "representation",
    "robotics", "segmentation", "self-supervised", "simulation", "supervised",
    "transformer", "training", "unsupervised", "vae",
}

# ---- Weak vs strong arXiv pattern split (for step 6 topic validation) -------
# "Weak" patterns are generic words like "paper", "research" that appear in
# both scientific and non-scientific requests.  When ONLY weak patterns match,
# step 6 requires the topic itself to contain arXiv-domain vocabulary.
_WEAK_ARXIV_PATTERNS = {
    r"paper[s]?",
    r"find\s.*paper", r"search\s.*paper", r"recent\s.*paper",
    r"latest\s.*paper", r"new\s.*paper", r"top\s.*paper",
    r"show\s.*paper", r"get\s.*paper", r"fetch\s.*paper",
    r"look\s.*paper", r"recommend\s.*paper", r"suggest\s.*paper",
    r"summarize\s.*paper", r"summarise\s.*paper",
    r"abstract", r"research\s.*interest", r"research\s.*topic",
    r"what.s new",
}
_WEAK_ARXIV_RE = re.compile("|".join(_WEAK_ARXIV_PATTERNS), re.IGNORECASE)

# Patterns from _ARXIV_PATTERNS that are NOT weak — if any of these match,
# the query is definitively arXiv-related and needs no topic validation.
_STRONG_TOPIC_PATTERNS = [p for p in _ARXIV_PATTERNS if p not in _WEAK_ARXIV_PATTERNS]
_STRONG_TOPIC_RE = re.compile("|".join(_STRONG_TOPIC_PATTERNS), re.IGNORECASE)

# ---- arXiv domain vocabulary ------------------------------------------------
# Words/phrases that map to arXiv's actual coverage areas.  Built from arXiv
# category names + common research terms.  Used in step 6 to validate that
# "papers on X" has an X that arXiv actually covers.  This is an allowlist
# approach — if none of these words appear in the topic, we reject.
_ARXIV_DOMAIN_VOCAB = {
    # --- Computer Science ---
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "neural", "nlp", "natural language", "computer vision",
    "reinforcement learning", "robotics", "automation", "autonomous",
    "cryptography", "security", "databases", "distributed",
    "parallel", "cluster", "computing", "computational",
    "software", "programming", "compiler", "operating system",
    "information retrieval", "data structures", "networking",
    "internet", "multimedia", "graphics", "human-computer",
    "multiagent", "game theory", "logic", "formal",
    "complexity", "geometry", "numerical", "symbolic",
    # --- Physics ---
    "physics", "quantum", "relativity", "cosmology", "astrophysics",
    "astronomy", "galaxy", "galaxies", "stellar", "solar",
    "condensed matter", "superconductivity", "magnetism",
    "particle", "hadron", "collider", "lattice", "phenomenology",
    "nuclear", "plasma", "optics", "photonics", "laser",
    "fluid dynamics", "thermodynamics", "entropy", "turbulence",
    "semiconductor", "nanoscale", "mesoscale", "gravitation",
    "gravitational", "dark matter", "dark energy", "neutrino",
    "boson", "fermion", "quark", "lepton", "higgs",
    "string theory", "field theory", "gauge", "hamiltonian",
    "lagrangian", "perturbation", "scattering", "spectroscopy",
    "wavelength", "electromagnetic", "radiation",
    # --- Mathematics ---
    "mathematics", "math", "algebra", "topology", "calculus",
    "differential", "equations", "probability", "stochastic",
    "combinatorics", "number theory", "group theory", "manifold",
    "homology", "cohomology", "category theory", "operator",
    "functional analysis", "harmonic", "fourier", "integral",
    "theorem", "conjecture", "proof", "lemma",
    "optimization", "convex", "linear", "nonlinear",
    "dynamical systems", "chaos", "bifurcation", "ergodic",
    # --- Statistics ---
    "statistics", "statistical", "bayesian", "frequentist",
    "hypothesis", "regression", "estimation", "variance",
    "inference", "sampling", "monte carlo", "markov",
    "time series", "causal", "econometrics",
    # --- Quantitative Biology ---
    "genomics", "proteomics", "bioinformatics", "molecular",
    "biomolecular", "protein", "gene", "genome", "dna", "rna",
    "cell", "cellular", "neuroscience", "cognition", "neuron",
    "neurons", "brain", "evolution", "population genetics",
    "epidemiology", "systems biology", "biochemistry",
    # --- Quantitative Finance ---
    "portfolio", "risk management", "derivatives", "pricing",
    "financial", "econometric", "trading", "market microstructure",
    # --- Electrical Engineering ---
    "signal processing", "image processing", "video processing",
    "speech", "audio", "control systems", "circuits",
    "antenna", "wireless", "radar", "communications",
    # --- Common research/ML terms ---
    "transformer", "transformers", "attention", "bert", "gpt", "llm", "llms", "lora",
    "rag", "retrieval-augmented generation", "retrieval augmented generation",
    "cnn", "cnns", "rnn", "rnns", "lstm", "gan", "gans", "vae", "vaes", "autoencoder",
    "diffusion", "generative", "adversarial", "contrastive",
    "self-supervised", "semi-supervised", "unsupervised", "supervised",
    "federated", "meta-learning", "few-shot", "zero-shot",
    "transfer learning", "fine-tuning", "pretraining",
    "embedding", "embeddings", "representation", "encoder", "decoder",
    "segmentation", "detection", "classification", "recognition",
    "tracking", "localization", "reconstruction", "synthesis",
    "benchmark", "dataset", "evaluation", "metric",
    "gradient", "backpropagation", "loss function",
    "regularization", "dropout", "batch normalization",
    "convolution", "pooling", "activation",
    "hyperparameter", "architecture", "backbone",
    "simulation", "modeling", "modelling", "model", "models",
    "algorithm", "algorithms", "heuristic", "approximation",
    "language", "graph", "network", "networks", "node", "edge",
    "knowledge graph", "ontology", "semantic",
    "sentiment", "translation", "summarization",
    "question answering", "dialogue", "chatbot",
    "recommendation", "collaborative filtering",
    "anomaly detection", "outlier", "fraud",
    "clustering", "dimensionality reduction", "pca",
    "uncertainty", "calibration", "robustness",
    "fairness", "bias", "interpretability", "explainability",
    "privacy", "differential privacy", "encryption",
    "blockchain", "consensus", "decentralized",
    "energy", "renewable", "solar cell", "photovoltaic",
    "battery", "catalyst", "materials science",
    "drug discovery", "drug design", "clinical",
    "medical imaging", "radiology", "pathology",
    "climate", "atmospheric", "ocean", "geophysics",
    "seismology", "remote sensing", "satellite",
    # --- Common abbreviations & model names ---
    "rlhf", "dpo", "gnn", "gnns", "svm", "svms", "knn",
    "mlp", "mlps", "resnet", "efficientnet", "vit", "u-net", "unet",
    "ner", "sbert", "clip", "nerf", "slam", "ppo",
    "peft", "moe", "nas", "automl", "xai",
    "xgboost", "ocr", "alphafold",
    # --- ML / NLP concepts ---
    "support vector machine", "support vector machines",
    "k-nearest neighbors", "random forests", "random forest",
    "multilayer perceptron", "multilayer perceptrons",
    "perceptron", "perceptrons",
    "word embeddings", "sentence embeddings",
    "information extraction", "coreference resolution",
    "coreference", "dependency parsing", "parsing",
    "prompt engineering", "chain of thought",
    "in-context learning", "instruction tuning",
    "mixture of experts", "speculative decoding", "decoding",
    "knowledge distillation", "distillation",
    "image generation", "super resolution", "style transfer",
    "image inpainting", "inpainting",
    "video understanding", "optical flow",
    "point clouds", "point cloud", "scene understanding",
    "multimodal learning", "image captioning", "captioning",
    "text-to-image", "text to image",
    "actor-critic", "imitation learning",
    "robot manipulation", "manipulation",
    "motion planning", "drone", "drones",
    "legged locomotion",
    # --- Math / theory ---
    "information theory", "pac learning",
    "spectral methods", "spectral",
    "random matrix theory", "random matrix",
    "topological", "insulator", "insulators",
    "bose-einstein", "condensate",
    # --- Bio / chem ---
    "crispr", "variant calling",
    "drug-target interaction", "drug target interaction",
    # --- Crypto / formal methods ---
    "zero-knowledge", "zero knowledge",
    "smart contracts", "smart contract",
    "sat solver", "sat solvers", "satisfiability",
    "constraint satisfaction",
    # --- Learning paradigms ---
    "active learning", "curriculum learning",
    "continual learning", "catastrophic forgetting",
    "data augmentation", "augmentation",
    "label noise", "noisy labels",
    "domain adaptation", "multi-task learning",
    "counterfactual", "counterfactual reasoning",
    # --- Catch-all research identifiers ---
    "preprint", "preprints", "arxiv",
}

# Pre-split multi-word vocab entries for efficient substring matching.
# Single words are checked via set intersection; multi-word phrases via 'in'.
_DOMAIN_SINGLE_WORDS = {v for v in _ARXIV_DOMAIN_VOCAB if " " not in v}
_DOMAIN_PHRASES = [v for v in _ARXIV_DOMAIN_VOCAB if " " in v]


def _topic_matches_arxiv_domain(text_lower: str) -> bool:
    """Return True if *text_lower* contains at least one arXiv-domain term.

    Checks single-word entries via fast set intersection and multi-word
    phrases via substring search.  This is intentionally generous — if
    ANY domain word appears, the topic is accepted.
    """
    words = set(w.strip(".,;:!?()[]{}\"'") for w in text_lower.split())
    if words & _DOMAIN_SINGLE_WORDS:
        return True
    for phrase in _DOMAIN_PHRASES:
        if phrase in text_lower:
            return True
    return False


# =============================================================================
# "Explain X" detector — research redirect
# =============================================================================

_EXPLAIN_RE = re.compile(
    r"^(explain|describe|what\s+is|what\s+are|define|tell\s+me\s+about|how\s+does)\b",
    re.IGNORECASE,
)

_SUMMARIZE_RE = re.compile(
    r"^(summarize|summarise|summary\s+of|give\s+me\s+a\s+summary)\b",
    re.IGNORECASE,
)

_ARXIV_ID_RE = re.compile(
    r"(arxiv[:\s]?\d{4}\.\d{4,5}|https?://arxiv\.org/\w+/\d{4}\.\d{4,5})",
    re.IGNORECASE,
)


# =============================================================================
# Core classification function
# =============================================================================

def classify_user_request(
    message: str,
    conversation_context: Optional[str] = None,
) -> ScopeResult:
    """
    Classify a user message for scope-gating.

    Args:
        message: The raw user message text.
        conversation_context: Optional prior conversation summary (unused
            in v1 but reserved for future multi-turn awareness).

    Returns:
        A ``ScopeResult`` with scope, reason, optional rewrite, and response.
    """
    text = message.strip()
    text_lower = text.lower()

    # ------------------------------------------------------------------
    # 0. Empty / trivially short messages → ask for topic
    # ------------------------------------------------------------------
    if len(text) < 3:
        return ScopeResult(
            scope=ScopeClass.IN_SCOPE,
            reason="message_too_short",
            response=RESPONSE_MISSING_TOPIC,
        )

    # ------------------------------------------------------------------
    # 1. Check for clearly general / off-topic requests FIRST
    # ------------------------------------------------------------------
    if _GENERAL_RE.search(text_lower):
        # Only override if the message contains a *strong* arXiv signal
        # (arXiv ID, category code, "preprint", etc.) OR an unambiguous
        # scientific/technical term (e.g. "robotics", "segmentation").
        # Generic words like "paper" or "research" are NOT enough —
        # "find papers on cooking recipes" must still be rejected.
        words = set(w.strip(".,;:!?()[]{}\"'") for w in text_lower.split())
        if _STRONG_ARXIV_RE.search(text_lower) or (words & _DOMAIN_SINGLE_WORDS) or _topic_matches_arxiv_domain(text_lower):
            pass  # fall through to in-scope checks
        else:
            logger.info("[SCOPE_GATE] scope=OUT_OF_SCOPE_GENERAL, reason=general_off_topic")
            return ScopeResult(
                scope=ScopeClass.OUT_OF_SCOPE_GENERAL,
                reason="general_off_topic",
                response=RESPONSE_OUT_OF_SCOPE_GENERAL,
            )

    # ------------------------------------------------------------------
    # 2. Non-arXiv venue mentioned?
    # ------------------------------------------------------------------
    for venue in _NON_ARXIV_VENUES:
        if venue in text_lower:
            # If they also mention arXiv, keep in scope
            if "arxiv" in text_lower:
                break
            logger.info(
                "[SCOPE_GATE] scope=OUT_OF_SCOPE_ARXIV_ONLY, "
                f"reason=non_arxiv_venue:{venue}"
            )
            return ScopeResult(
                scope=ScopeClass.OUT_OF_SCOPE_ARXIV_ONLY,
                reason=f"non_arxiv_venue:{venue}",
                suggested_rewrite=(
                    f"I can't search {venue}, but I can look for arXiv "
                    f"preprints related to your topic. What keywords should I use?"
                ),
                response=RESPONSE_NON_ARXIV_VENUE,
            )

    # ------------------------------------------------------------------
    # 3. "Explain X" without asking for papers → redirect
    #    (checked BEFORE broad arXiv keyword match so "What is a CNN?"
    #     hits the explain branch, not the neural-network keyword)
    # ------------------------------------------------------------------
    if _EXPLAIN_RE.search(text_lower):
        # If they explicitly mention paper(s) / arXiv, keep in scope
        if not re.search(r"paper|arxiv|preprint", text_lower):
            topic = _EXPLAIN_RE.sub("", text).strip().rstrip("?.!")
            logger.info(
                "[SCOPE_GATE] scope=OUT_OF_SCOPE_ARXIV_ONLY, reason=explain_without_papers"
            )
            return ScopeResult(
                scope=ScopeClass.OUT_OF_SCOPE_ARXIV_ONLY,
                reason="explain_without_papers",
                suggested_rewrite=(
                    f"Find arXiv papers that explain {topic}" if topic else None
                ),
                response=(
                    "I can find arXiv papers that explain it. "
                    "What subtopic and level are you interested in?"
                ),
            )

    # ------------------------------------------------------------------
    # 4. "Summarize" without an arXiv ID/link → ask for link
    # ------------------------------------------------------------------
    if _SUMMARIZE_RE.search(text_lower):
        if not _ARXIV_ID_RE.search(text):
            # Check if general paper mention (still in scope, but prompt for ID)
            if _ARXIV_RE.search(text_lower):
                pass  # has paper-related keywords, let it proceed
            else:
                logger.info(
                    "[SCOPE_GATE] scope=IN_SCOPE, reason=summarize_missing_arxiv_id"
                )
                return ScopeResult(
                    scope=ScopeClass.IN_SCOPE,
                    reason="summarize_missing_arxiv_id",
                    response=RESPONSE_NEED_ARXIV_LINK,
                )

    # ------------------------------------------------------------------
    # 5. Vague "find papers" / "latest papers" without a real topic → ask
    #    (checked BEFORE broad arXiv match so bare "find papers" triggers
    #     the missing-topic follow-up instead of silently proceeding)
    # ------------------------------------------------------------------
    _vague_paper = re.search(
        r"^(find|search|get|show|fetch|look\s+for|latest|recent|new)\s+(papers?|articles?|preprints?|research)\s*$",
        text_lower,
    )
    if _vague_paper:
        logger.info("[SCOPE_GATE] scope=IN_SCOPE, reason=missing_topic")
        return ScopeResult(
            scope=ScopeClass.IN_SCOPE,
            reason="missing_topic",
            response=RESPONSE_MISSING_TOPIC,
        )

    # ------------------------------------------------------------------
    # 6. Positive match: arXiv / paper-related keywords
    #    If a *strong* pattern matched (arXiv IDs, category codes, specific
    #    technical terms), accept immediately.  If only a *weak* pattern
    #    matched ("paper", "recent paper", etc.), verify the topic itself
    #    maps to an arXiv domain — this prevents "papers on flowers" from
    #    slipping through while keeping "papers on quantum computing" in.
    # ------------------------------------------------------------------
    if _ARXIV_RE.search(text_lower):
        if _STRONG_TOPIC_RE.search(text_lower) or _topic_matches_arxiv_domain(text_lower):
            logger.info("[SCOPE_GATE] scope=IN_SCOPE, reason=arxiv_keyword_match")
            return ScopeResult(
                scope=ScopeClass.IN_SCOPE,
                reason="arxiv_keyword_match",
            )
        # Before rejecting, check if this is an operational request that
        # happens to mention "paper" (e.g. "share this paper with my
        # colleague").  Let those fall through to step 7.
        if any(kw in text_lower for kw in _OPERATIONAL_KEYWORDS):
            pass  # fall through to step 7 (operational keywords)
        else:
            # Only weak patterns matched and topic has no arXiv domain words.
            logger.info(
                "[SCOPE_GATE] scope=OUT_OF_SCOPE_GENERAL, "
                "reason=weak_arxiv_signal_no_domain_topic"
            )
            return ScopeResult(
                scope=ScopeClass.OUT_OF_SCOPE_GENERAL,
                reason="weak_arxiv_signal_no_domain_topic",
                response=RESPONSE_OUT_OF_SCOPE_GENERAL,
            )

    # ------------------------------------------------------------------
    # 7. Operational keywords (colleague, email, settings, etc.)
    # ------------------------------------------------------------------
    for kw in _OPERATIONAL_KEYWORDS:
        if kw in text_lower:
            logger.info(
                f"[SCOPE_GATE] scope=IN_SCOPE, reason=operational_keyword:{kw}"
            )
            return ScopeResult(
                scope=ScopeClass.IN_SCOPE,
                reason=f"operational_keyword:{kw}",
            )

    # ------------------------------------------------------------------
    # 8. Fallback heuristic: if the message contains research-y words
    #    that didn't match earlier patterns, give benefit of the doubt
    #    but only if it's short enough to be a query
    # ------------------------------------------------------------------
    _RESEARCH_HINTS = {
        "model", "models", "algorithm", "algorithms", "benchmark",
        "dataset", "training", "inference", "evaluation", "attention",
        "transformer", "bert", "gpt", "llm", "cnn", "rnn", "gan",
        "vae", "rl", "optimization", "gradient", "architecture",
        "network", "embedding", "fine-tuning", "finetune", "pretrain",
        "pretraining", "survey", "review", "segmentation", "detection",
        "classification", "regression", "robotics", "locomotion",
        "simulation", "reinforcement", "supervised", "unsupervised",
        "generative", "adversarial", "contrastive", "representation",
        "self-supervised", "federated", "multi-modal", "multimodal",
    }
    words = set(w.strip(".,;:!?()[]{}\"'") for w in text_lower.split())
    if words & _RESEARCH_HINTS:
        logger.info("[SCOPE_GATE] scope=IN_SCOPE, reason=research_hint_fallback")
        return ScopeResult(
            scope=ScopeClass.IN_SCOPE,
            reason="research_hint_fallback",
        )

    # ------------------------------------------------------------------
    # 9. Nothing matched → treat as general out-of-scope
    # ------------------------------------------------------------------
    logger.info("[SCOPE_GATE] scope=OUT_OF_SCOPE_GENERAL, reason=no_matching_signal")
    return ScopeResult(
        scope=ScopeClass.OUT_OF_SCOPE_GENERAL,
        reason="no_matching_signal",
        response=RESPONSE_OUT_OF_SCOPE_GENERAL,
    )
