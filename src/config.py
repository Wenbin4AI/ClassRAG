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
    use_vectorized_retrieval: bool = False  # True for vectorized matrix computation, False for BFS iteration
    enable_hybrid_attribute_fallback: bool = False
    attribute_keyword_boost: float = 0.25

    entity_class_only: bool = False
    entity_class_overwrite: bool = False
    entity_class_max_classes: int = 4
    entity_class_max_workers: int = 4

    use_class_schema: bool = False
    entity_classes_path: str = ""
    schema_state_path: str = ""
    query_type_cache_path: str = ""
    enable_llm_query_type_inference: bool = True
    class_boost_alpha: float = 0.3
    ppr_class_prior_gamma: float = 0.3
    schema_same_weight: float = 1.0
    schema_descendant_weight: float = 0.9
    schema_ancestor_weight: float = 0.5
    schema_related_weight: float = 0.3

    attribute_query_keywords: list[str] = field(default_factory=lambda: [
        "born", "birth", "where", "when", "located", "location", "founded", "founder",
        "died", "death", "nationality", "capital", "date", "year"
    ])