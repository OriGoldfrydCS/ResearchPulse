"""Quick verification script for topic matching fixes."""
import sys
sys.path.insert(0, "src")
from tools.decide_delivery import _topics_overlap

# ---- ATTENTION test ----
psych_paper = ["study", "children", "inattention", "behavioral", "deficit", "disorder",
               "sustained", "cognitive", "performance", "psychology"]
ml_paper = ["attention", "mechanism", "self-attention", "transformer", "multi-head",
            "layers", "query", "key", "value", "softmax"]
colleague_attention = ["attention"]

match_psych, t_psych = _topics_overlap(psych_paper, colleague_attention)
match_ml, t_ml = _topics_overlap(ml_paper, colleague_attention)
print("=== ATTENTION ===")
print("Psychology paper matches 'attention':", match_psych, "->", t_psych)
print("ML paper matches 'attention':        ", match_ml, "->", t_ml)
assert not match_psych, "FAIL: psychology 'inattention' should NOT match 'attention'"
assert match_ml, "FAIL: ML attention paper SHOULD match 'attention'"

# ---- TRANSFORMERS test ----
transform_paper = ["data", "transformation", "pipeline", "processing", "feature", "engineering"]
transformer_paper = ["transformer", "model", "architecture", "encoder", "decoder",
                     "self-attention", "training", "language"]
colleague_transformers = ["transformers"]

match_transform, t_transform = _topics_overlap(transform_paper, colleague_transformers)
match_actual, t_actual = _topics_overlap(transformer_paper, colleague_transformers)
print("\n=== TRANSFORMERS ===")
print("'transformation' paper matches 'transformers':", match_transform, "->", t_transform)
print("Actual transformer paper matches 'transformers':", match_actual, "->", t_actual)
assert not match_transform, "FAIL: 'transformation' should NOT match 'transformers'"
assert match_actual, "FAIL: 'transformer' paper SHOULD match 'transformers'"

# ---- RAG test ----
storage_paper = ["storage", "database", "fragmentation", "encourage", "system", "distributed"]
rag_paper = ["retrieval-augmented", "generation", "retrieval", "augmented", "knowledge",
             "rag", "language", "model"]
colleague_rag = ["rag"]

match_storage, t_storage = _topics_overlap(storage_paper, colleague_rag)
match_rag, t_rag = _topics_overlap(rag_paper, colleague_rag)
print("\n=== RAG ===")
print("'storage/fragmentation' paper matches 'rag':", match_storage, "->", t_storage)
print("Actual RAG paper matches 'rag':              ", match_rag, "->", t_rag)
assert not match_storage, "FAIL: 'storage' should NOT match 'rag'"
assert match_rag, "FAIL: actual RAG paper SHOULD match 'rag'"

# ---- RAG expanded form (no literal 'rag') ----
rag_expanded_paper = ["retrieval-augmented", "generation", "knowledge", "base", "documents"]
match_exp, t_exp = _topics_overlap(rag_expanded_paper, colleague_rag)
print("RAG paper (expanded form, no literal 'rag'):", match_exp, "->", t_exp)
assert match_exp, "FAIL: expanded RAG form SHOULD match 'rag' via ML_TERM_EXPANSIONS"

# ---- Score relevance substring ratio ----
from tools.score_relevance import _calculate_topic_overlap, _extract_keywords
paper_kw = _extract_keywords("Data transformation pipeline for feature engineering")
score_transform = _calculate_topic_overlap(paper_kw, ["transformers"])
print("\n=== SCORE RELEVANCE ===")
print("'data transformation' vs topic 'transformers':", score_transform)
assert score_transform == 0.0, f"FAIL: 'transformation' should NOT match 'transformers' (got {score_transform})"

paper_kw2 = _extract_keywords("Efficient transformer model for language understanding")
score_real = _calculate_topic_overlap(paper_kw2, ["transformers"])
print("'transformer model' vs topic 'transformers':  ", score_real)
assert score_real > 0.0, f"FAIL: 'transformer' SHOULD match 'transformers' (got {score_real})"

print("\n=== ALL CHECKS PASSED ===")
