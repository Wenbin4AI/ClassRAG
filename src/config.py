from dataclasses import dataclass, field
from src.utils import LLM_Model


@dataclass
class ClassRAGConfig:
    dataset_name: str
    embedding_model: str = "all-mpnet-base-v2"
    llm_model: LLM_Model = None
    chunk_token_size: int = 1000
    chunk_overlap_token_size: int = 100
    spacy_model: str = "en_core_web_trf"
    working_dir: str = "./import"
    batch_size: int = 128
    max_workers: int = 16
    retrieval_top_k: int = 5
    max_iterations: int = 3
    top_k_sentence: int = 1
    passage_ratio: float = 1.5
    passage_node_weight: float = 0.05
    damping: float = 0.5
    iteration_threshold: float = 0.5
    use_vectorized_retrieval: bool = False

    enable_hybrid_attribute_fallback: bool = False
    attribute_keyword_boost: float = 0.25

    entity_class_only: bool = False
    entity_class_overwrite: bool = False
    entity_class_max_classes: int = 4
    entity_class_max_workers: int = 4

    # ------------------------------------------------------------
    # Class / ontology schema enhancement
    # ------------------------------------------------------------
    use_class_schema: bool = False
    entity_classes_path: str = ""
    schema_state_path: str = ""
    query_type_cache_path: str = ""
    enable_llm_query_type_inference: bool = True

    # Class-aware entity propagation
    class_boost_alpha: float = 0.3

    # Class-aware PPR node prior
    ppr_class_prior_gamma: float = 0.3

    # Class compatibility weights
    schema_same_weight: float = 1.0
    schema_descendant_weight: float = 0.9
    schema_ancestor_weight: float = 0.5
    schema_related_weight: float = 0.3

    # ------------------------------------------------------------
    # Query-aware edge transition for PPR
    # ------------------------------------------------------------
    use_query_aware_edge_transition: bool = True

    # General boost for passage-entity edges connected to compatible entities
    edge_class_beta: float = 0.8

    # Extra boost if a passage connects a seed entity and a compatible target entity
    edge_seed_bridge_beta: float = 0.6

    # Extra boost if the compatible entity is activated by BFS propagation
    edge_active_beta: float = 0.4

    # Extra boost for compatible entities within two-hop propagation distance
    edge_two_hop_beta: float = 0.4

    # Minimum compatibility required to reweight an edge
    edge_min_compat: float = 0.01

    # Upper bound for total multiplicative boost
    edge_max_boost: float = 2.5

    attribute_query_keywords: list[str] = field(default_factory=lambda: [
        "born", "birth", "where", "when", "located", "location", "founded", "founder",
        "died", "death", "nationality", "capital", "date", "year"
    ])