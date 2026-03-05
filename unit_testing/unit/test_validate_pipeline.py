"""
Validation-only script for the PCA query pipeline fix.

Exercises the full pipeline (parse → query build → arXiv fetch → score → enforce)
across all test cases A–H and prints a structured report.

Usage:
    python _validate_pipeline.py
"""

import json
import os
import sys
import time

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.agent.prompt_controller import (
    PromptParser,
    OutputEnforcer,
    ParsedPrompt,
)
from src.tools.fetch_arxiv import (
    _build_topic_clause,
    fetch_arxiv_papers,
)
from src.tools.score_relevance import score_relevance_and_importance

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

parser = PromptParser()
enforcer = OutputEnforcer()

# Minimal research profile that treats the extracted topic as the interest.
def make_profile(topic: str):
    return {
        "research_topics": [topic],
        "avoid_topics": [],
        "arxiv_categories_include": [],
        "arxiv_categories_exclude": [],
        "preferred_venues": [],
        "my_paper_titles": [],
    }


def build_arxiv_query_string(topic: str) -> str:
    """Reconstruct the arXiv query string that _fetch_real_papers would use."""
    topics = [t.strip() for t in topic.split(" OR ") if t.strip()]
    if len(topics) > 1:
        ti_parts = " OR ".join(f'ti:"{t}"' for t in topics)
        abs_parts = " OR ".join(f'abs:"{t}"' for t in topics)
        return f"({ti_parts} OR {abs_parts})"
    return _build_topic_clause(topics[0])


def run_case(label: str, prompt: str, *, fetch: bool = True):
    """Run a single validation case and return a result dict."""
    parsed = parser.parse(prompt)
    topic = parsed.interests_only
    time_days = parsed.time_days
    arxiv_query = build_arxiv_query_string(topic) if topic else "(no topic)"

    result = {
        "label": label,
        "raw_prompt": prompt,
        "interests_only": topic,
        "time_days": time_days,
        "arxiv_query": arxiv_query,
        "titles": [],
        "relevance_scores": [],
        "warning": None,
        "fallback_triggered": False,
    }

    if not fetch:
        return result

    # Fetch from real arXiv API
    days = time_days if time_days and time_days <= 365 else 30
    try:
        fetch_result = fetch_arxiv_papers(
            query=topic,
            max_results=10,
            days_back=min(days, 30),  # arXiv API max 30
            use_mock=False,
        )
    except Exception as e:
        result["titles"] = [f"FETCH ERROR: {e}"]
        return result

    if not fetch_result.success or not fetch_result.papers:
        result["titles"] = ["(no results)"]
        # Build fake paper list for enforcer test
        papers_dicts = []
    else:
        papers_dicts = []
        profile = make_profile(topic)
        for p in fetch_result.papers[:10]:
            paper_dict = {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "abstract": p.abstract,
                "categories": p.categories,
                "authors": p.authors,
                "publication_date": p.published,
                "link": p.link,
            }
            scoring = score_relevance_and_importance(paper_dict, profile)
            paper_dict["relevance_score"] = scoring.relevance_score
            papers_dicts.append(paper_dict)
            result["titles"].append(p.title)
            result["relevance_scores"].append(scoring.relevance_score)

    # Run OutputEnforcer
    if papers_dicts:
        enforced = enforcer.enforce(
            papers_dicts,
            ParsedPrompt(requested_count=min(len(papers_dicts), 10)),
        )
        result["warning"] = enforced.message
    else:
        result["warning"] = "(no papers to enforce)"

    return result


def print_case(r: dict):
    """Pretty-print one validation result."""
    print(f"\n{'='*72}")
    print(f"  {r['label']}")
    print(f"{'='*72}")
    print(f"  Raw prompt      : {r['raw_prompt']}")
    print(f"  interests_only  : {r['interests_only']}")
    print(f"  time_days       : {r['time_days']}")
    print(f"  arXiv query     : {r['arxiv_query']}")
    if r["titles"]:
        print(f"  Top titles ({len(r['titles'])}):")
        for i, t in enumerate(r["titles"], 1):
            score_str = f"  [rel={r['relevance_scores'][i-1]:.3f}]" if i <= len(r["relevance_scores"]) else ""
            print(f"    {i:>2}. {t}{score_str}")
    else:
        print("  Top titles: (none)")
    print(f"  Warning flag    : {r['warning']}")
    print(f"  Fallback triggered: {r['fallback_triggered']}")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

SECTIONS = {
    "A": {
        "name": "Acronyms and Short Topics (precision)",
        "cases": [
            ("A1", "provide recent research papers on PCA published within the last two weeks"),
            ("A2", "papers on LLM published within the last 10 days"),
            ("A3", "recent papers on GANs from the last 3 days"),
            ("A4", "papers about SVM within the last 14 days"),
            ("A5", "papers on t-SNE from the past 2 weeks"),
            ("A6", "papers regarding UMAP within the last month"),
            ("A7", "papers on CNN from the last two weeks"),
            ("A8", "papers on RNN within the last 30 days"),
            ("A9", "papers on VAE from the past 21 days"),
            ("A10", "papers on RL within the last 3 weeks"),
        ],
    },
    "B": {
        "name": "Full Topic Names",
        "cases": [
            ("B1", "papers on Principal Component Analysis within the last two weeks"),
            ("B2", "papers on Support Vector Machines within the last 14 days"),
            ("B3", "papers on Generative Adversarial Networks within the last 3 weeks"),
            ("B4", "papers on Convolutional Neural Networks within the last 30 days"),
            ("B5", "papers on Recurrent Neural Networks within the last 2 months"),
            ("B6", "papers on Variational Autoencoders within the last 45 days"),
            ("B7", "papers on Reinforcement Learning within the last 3 weeks"),
            ("B8", "papers on Graph Neural Networks within the last month"),
            ("B9", "papers on Diffusion Models within the last 14 days"),
            ("B10", "papers on Transformer Models within the last two weeks"),
        ],
    },
    "C": {
        "name": "Multi-word Topics (phrase vs token AND)",
        "cases": [
            ("C1", "recent papers on deep reinforcement learning from the last 30 days"),
            ("C2", "papers on graph neural networks within the last 3 months"),
            ("C3", "papers on diffusion models from the past 21 days"),
            ("C4", "papers on large language models within the last 2 weeks"),
            ("C5", "papers on variational autoencoders from the last 14 days"),
            ("C6", "papers on probabilistic graphical models within the last month"),
            ("C7", "papers on contrastive learning within the last two weeks"),
            ("C8", "papers on representation learning within the last 30 days"),
        ],
    },
    "D": {
        "name": "Topics containing 'recent' (must NOT be stripped)",
        "cases": [
            ("D1", "papers on Recent Advances in Graph Neural Networks"),
            ("D2", "papers on Recent Trends in Deep Learning"),
            ("D3", "papers on Recent Developments in Reinforcement Learning"),
            ("D4", "papers on recent advances in computer vision"),
            ("D5", "papers on recently proposed attention mechanisms"),
        ],
    },
    "E": {
        "name": "Hard Time Expressions (word numbers and variants)",
        "cases": [
            ("E1", "papers on PCA from the last two weeks"),
            ("E2", "papers on PCA from the last twenty two weeks"),
            ("E3", "papers on PCA from the last forty five days"),
            ("E4", "papers on PCA within the last ninety nine days"),
            ("E5", "papers on PCA within the last 22 weeks"),
            ("E6", "papers on PCA within the past 1 month"),
            ("E7", "papers on PCA within the last 3 months"),
            ("E8", "papers on PCA within the last twelve months"),
            ("E9", "papers on PCA from the past ten days"),
            ("E10", "papers on PCA from the last thirty five days"),
        ],
    },
    "F": {
        "name": "Mixed Prompts (backward compat & regressions)",
        "cases": [
            ("F1", "find 5 papers on Multi Armed Bandits"),
            ("F2", "recent papers on TSNE from the last 3 days"),
            ("F3", "provide 10 papers on reinforcement learning published in the last month"),
            ("F4", "papers on attention mechanisms"),
            ("F5", "provide recent research papers on PCA, TSNE, and UMAP within the last two weeks"),
            ("F6", "find recent papers on support vector machines and kernel methods"),
            ("F7", "top 10 recent papers on large language models"),
        ],
    },
    "G": {
        "name": "Edge Scientific Topics (complex real terms)",
        "cases": [
            ("G1", "papers on Bayesian Inference within the last 3 months"),
            ("G2", "papers on Markov Decision Processes within the last two weeks"),
            ("G3", "papers on Stochastic Gradient Descent within the last 30 days"),
            ("G4", "papers on Maximum Likelihood Estimation within the last month"),
            ("G5", "papers on Monte Carlo Methods within the last 14 days"),
            ("G6", "papers on Dimensionality Reduction within the last 3 weeks"),
            ("G7", "papers on manifold learning within the last 30 days"),
        ],
    },
    "H": {
        "name": "Negative / Stress Cases",
        "cases": [
            ("H1", "papers on qwertyuiop within the last two weeks"),
            ("H2", "papers on blorf dynamics from the last 30 days"),
            ("H3", "papers on X within the last two weeks"),
            ("H4", "papers on unknown algorithm xyz123 from the last 3 weeks"),
        ],
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_results = {}
    issues = []

    for section_key in sorted(SECTIONS.keys()):
        section = SECTIONS[section_key]
        print(f"\n\n{'#'*72}")
        print(f"# SECTION {section_key}: {section['name']}")
        print(f"{'#'*72}")

        section_results = []
        for label, prompt in section["cases"]:
            # Sections D and E only need parsing validation (no arXiv fetch)
            # Sections A-C, F-H do full fetch
            need_fetch = section_key not in ("D", "E")
            r = run_case(label, prompt, fetch=need_fetch)
            print_case(r)
            section_results.append(r)

            # --- Validation checks ---
            # Check for time leakage in interests_only
            leaky_words = ["published", "within", "from the last", "during",
                           "days", "weeks", "months", "past"]
            # Only flag leakage for sections where temporal stripping is expected
            if section_key not in ("D",):
                for lw in leaky_words:
                    if lw in r["interests_only"].lower():
                        # But don't flag it if it's part of the actual topic
                        if section_key == "D":
                            continue
                        msg = f"[{label}] LEAKAGE: '{lw}' found in interests_only='{r['interests_only']}'"
                        issues.append(msg)
                        print(f"  *** {msg}")

            # Check multi-word query construction (sections B, C, G)
            if section_key in ("B", "C", "G"):
                q = r["arxiv_query"]
                if " " in r["interests_only"] and "AND" not in q:
                    msg = f"[{label}] MISSING token AND clause in query"
                    issues.append(msg)
                    print(f"  *** {msg}")
                if " " in r["interests_only"] and 'ti:"' not in q:
                    msg = f"[{label}] MISSING phrase clause in query"
                    issues.append(msg)
                    print(f"  *** {msg}")

            # arXiv rate-limit: 3 seconds between calls
            if need_fetch:
                time.sleep(3.5)

        all_results[section_key] = section_results

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print(f"\n\n{'#'*72}")
    print(f"# FINAL SUMMARY")
    print(f"{'#'*72}")

    # Summarise time parsing (section E)
    print("\n--- Section E: Time Parsing Results ---")
    expected_E = {
        "E1": 14, "E2": 154, "E3": 45, "E4": 99, "E5": 154,
        "E6": 30, "E7": 90, "E8": 360, "E9": 10, "E10": 35,
    }
    for r in all_results.get("E", []):
        exp = expected_E.get(r["label"])
        status = "OK" if r["time_days"] == exp else f"FAIL (got {r['time_days']}, expected {exp})"
        print(f"  {r['label']}: time_days={r['time_days']}  expected={exp}  {status}")
        if r["time_days"] != exp:
            issues.append(f"[{r['label']}] time_days mismatch: got {r['time_days']}, expected {exp}")

    # Summarise D: "recent" preservation
    print("\n--- Section D: 'Recent' Preservation ---")
    for r in all_results.get("D", []):
        has_recent = "recent" in r["interests_only"].lower()
        status = "OK (preserved)" if has_recent else "FAIL (stripped)"
        print(f"  {r['label']}: interests_only='{r['interests_only']}'  {status}")
        if not has_recent:
            issues.append(f"[{r['label']}] 'recent' was stripped from interests_only")

    # Issues
    print(f"\n--- Issues ({len(issues)}) ---")
    if issues:
        for iss in issues:
            print(f"  - {iss}")
    else:
        print("  None! All checks passed.")

    print(f"\nTotal test cases: {sum(len(s['cases']) for s in SECTIONS.values())}")
    print(f"Issues found: {len(issues)}")
    print()


if __name__ == "__main__":
    main()
