"""
Comprehensive scope-gate coverage tests.

Goal: catch vocabulary gaps where legitimate arXiv research topics are
incorrectly rejected (false negatives) AND ensure off-topic queries are
still blocked (true negatives).

This file is ADDITIVE — it only tests classify_user_request output
and never modifies any production code.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.agent.scope_gate import classify_user_request, ScopeClass


# =============================================================================
# Helper
# =============================================================================

def _assert_in_scope(message: str):
    result = classify_user_request(message)
    assert result.scope == ScopeClass.IN_SCOPE, (
        f"FALSE REJECTION — expected IN_SCOPE for: {message!r}, "
        f"got {result.scope.value} (reason={result.reason})"
    )


def _assert_out_of_scope(message: str):
    result = classify_user_request(message)
    assert result.scope != ScopeClass.IN_SCOPE, (
        f"FALSE ACCEPT — expected OUT_OF_SCOPE for: {message!r}, "
        f"got {result.scope.value} (reason={result.reason})"
    )


# =============================================================================
# Legitimate research topics that MUST be IN_SCOPE
# Each entry is a realistic user query about an arXiv-relevant topic.
# =============================================================================

class TestLegitimateResearchTopics:
    """Queries about real arXiv research areas — all must be IN_SCOPE."""

    # -- Common ML/AI abbreviations and terms ----------------------------------
    @pytest.mark.parametrize("message", [
        # RAG
        "Provide the most recent research papers on RAG",
        "papers on retrieval-augmented generation",
        "latest RAG papers",
        # RLHF / alignment
        "papers on RLHF",
        "recent papers on reinforcement learning from human feedback",
        "papers on AI alignment",
        # DPO / preference optimization
        "papers on DPO",
        "find papers on direct preference optimization",
        # Graph neural networks
        "papers on GNN",
        "latest papers on graph neural networks",
        "recent GNN papers for molecular property prediction",
        # Classic ML abbreviations
        "papers on SVM",
        "papers on support vector machines",
        "papers on KNN",
        "papers on k-nearest neighbors",
        "papers on random forests",
        "papers on XGBoost",
        "papers on gradient boosting",
        # Neural network architectures
        "papers on MLP",
        "papers on multilayer perceptrons",
        "papers on ResNet",
        "papers on EfficientNet",
        "papers on Vision Transformer",
        "papers on ViT",
        "papers on U-Net",
        "papers on YOLO object detection",
        # NLP-specific
        "papers on NER",
        "papers on named entity recognition",
        "papers on NLP tokenization",
        "papers on word embeddings",
        "papers on sentence embeddings",
        "papers on SBERT",
        "papers on text classification",
        "papers on sentiment analysis",
        "papers on topic modeling",
        "papers on information extraction",
        "papers on coreference resolution",
        "papers on dependency parsing",
        # LLM-related
        "papers on prompt engineering",
        "papers on chain of thought reasoning",
        "papers on in-context learning",
        "papers on instruction tuning",
        "papers on PEFT",
        "papers on parameter-efficient fine-tuning",
        "papers on LoRA adapters",
        "papers on mixture of experts",
        "papers on MoE",
        "papers on speculative decoding",
        "papers on LLM quantization",
        "papers on knowledge distillation",
        "papers on model compression",
        "papers on pruning neural networks",
        "papers on sparse attention",
        # Computer vision
        "papers on OCR",
        "papers on optical character recognition",
        "papers on image segmentation",
        "papers on object detection",
        "papers on pose estimation",
        "papers on depth estimation",
        "papers on 3D reconstruction",
        "papers on NeRF",
        "papers on neural radiance fields",
        "papers on SLAM",
        "papers on visual SLAM",
        "papers on image generation",
        "papers on super resolution",
        "papers on style transfer",
        "papers on image inpainting",
        "papers on video understanding",
        "papers on optical flow",
        "papers on point clouds",
        "papers on scene understanding",
        # Multimodal
        "papers on CLIP",
        "papers on vision-language models",
        "papers on multimodal learning",
        "papers on image captioning",
        "papers on visual question answering",
        "papers on text-to-image generation",
        # Reinforcement learning
        "papers on PPO",
        "papers on proximal policy optimization",
        "papers on actor-critic methods",
        "papers on model-based reinforcement learning",
        "papers on multi-agent reinforcement learning",
        "papers on offline reinforcement learning",
        "papers on imitation learning",
        "papers on inverse reinforcement learning",
        # Robotics and control
        "papers on robot manipulation",
        "papers on motion planning",
        "papers on autonomous navigation",
        "papers on drone control",
        "papers on legged locomotion",
        # Systems / infra
        "papers on distributed training",
        "papers on model parallelism",
        "papers on edge computing for AI",
        "papers on federated learning",
        "papers on neural architecture search",
        "papers on NAS",
        "papers on AutoML",
        # Safety / trustworthy AI
        "papers on adversarial robustness",
        "papers on out-of-distribution detection",
        "papers on OOD detection",
        "papers on model interpretability",
        "papers on explainable AI",
        "papers on XAI",
        "papers on AI fairness",
        "papers on LLM hallucination",
        "papers on watermarking language models",
        # Math / theory
        "papers on convex optimization",
        "papers on Bayesian inference",
        "papers on variational inference",
        "papers on information theory",
        "papers on PAC learning",
        "papers on spectral methods",
        "papers on random matrix theory",
        # Physics
        "papers on quantum computing",
        "papers on quantum error correction",
        "papers on lattice QCD",
        "papers on dark matter detection",
        "papers on gravitational waves",
        "papers on topological insulators",
        "papers on Bose-Einstein condensate",
        "papers on high energy physics",
        # Biology / bioinformatics
        "papers on protein folding",
        "papers on AlphaFold",
        "papers on single-cell RNA sequencing",
        "papers on genomic variant calling",
        "papers on CRISPR off-target prediction",
        "papers on drug-target interaction",
        # Finance
        "papers on portfolio optimization",
        "papers on option pricing models",
        "papers on market microstructure",
        # Misc arXiv-relevant
        "papers on differential privacy",
        "papers on homomorphic encryption",
        "papers on zero-knowledge proofs",
        "papers on smart contracts verification",
        "papers on program synthesis",
        "papers on automated theorem proving",
        "papers on SAT solvers",
        "papers on constraint satisfaction",
        "papers on combinatorial optimization",
        "papers on time series forecasting",
        "papers on anomaly detection in time series",
        "papers on causal inference",
        "papers on counterfactual reasoning",
        "papers on active learning",
        "papers on curriculum learning",
        "papers on continual learning",
        "papers on catastrophic forgetting",
        "papers on meta-learning",
        "papers on few-shot learning",
        "papers on contrastive learning",
        "papers on self-supervised learning",
        "papers on data augmentation",
        "papers on label noise",
        "papers on domain adaptation",
        "papers on transfer learning",
        "papers on multi-task learning",
    ])
    def test_legitimate_topic_in_scope(self, message: str):
        _assert_in_scope(message)


# =============================================================================
# Off-topic queries that MUST stay OUT_OF_SCOPE
# =============================================================================

class TestOffTopicStaysBlocked:
    """Queries that are NOT arXiv-related — must NOT be IN_SCOPE."""

    @pytest.mark.parametrize("message", [
        # Everyday life
        "papers on best pizza recipes",
        "find papers on gardening tips",
        "latest papers on celebrity gossip",
        "papers on fashion trends 2026",
        "papers on how to train your dog",
        "papers on home decoration ideas",
        "papers on yoga exercises",
        "papers on travel destinations",
        "papers on wedding planning",
        "papers on makeup tutorials",
        # General requests with "papers" keyword (should still block)
        "recommend papers on fishing techniques",
        "find papers on woodworking projects",
        "search papers on knitting patterns",
        "latest papers on wine tasting",
        "recent papers on interior design",
        # Non-research requests
        "what is the weather in London",
        "tell me a joke",
        "write my essay about Shakespeare",
        "help me code a website",
        "translate this to Spanish",
        "who won the basketball game",
        "best restaurants near me",
        "how to cook pasta",
        "movie recommendations for tonight",
        "fix my Python code",
    ])
    def test_off_topic_stays_blocked(self, message: str):
        _assert_out_of_scope(message)
