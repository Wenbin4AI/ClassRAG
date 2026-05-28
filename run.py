import argparse
import json
import os
import re
import warnings
from datetime import datetime

from sentence_transformers import SentenceTransformer

from src.config import ClassRAGConfig
from src.ClassRAG import ClassRAG
from src.evaluate import Evaluator
from src.utils import LLM_Model, setup_logging


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--spacy_model",
        type=str,
        default="en_core_web_trf",
        help="The spaCy model to use."
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        default="model/all-mpnet-base-v2",
        help="The path of embedding model to use."
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="2wikimultihop",
        help="The dataset to use."
    )

    parser.add_argument(
        "--llm_model",
        type=str,
        default="/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
        help="The LLM model to use."
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="The max number of workers to use."
    )

    parser.add_argument(
        "--max_iterations",
        type=int,
        default=3,
        help="The max number of entity propagation iterations."
    )

    parser.add_argument(
        "--iteration_threshold",
        type=float,
        default=0.4,
        help="The threshold for entity propagation."
    )

    parser.add_argument(
        "--passage_ratio",
        type=float,
        default=2.0,
        help="The weight ratio for dense passage score."
    )

    parser.add_argument(
        "--top_k_sentence",
        type=int,
        default=3,
        help="The top-k sentences used for entity propagation."
    )

    parser.add_argument(
        "--use_vectorized_retrieval",
        action="store_true",
        help="Use vectorized matrix-based retrieval instead of BFS iteration."
    )

    # ------------------------------------------------------------
    # Entity class inference options
    # ------------------------------------------------------------
    parser.add_argument(
        "--entity_class_only",
        action="store_true",
        help="Only infer entity classes and save them, then exit before QA and evaluation."
    )

    parser.add_argument(
        "--entity_class_overwrite",
        action="store_false",
        help="Overwrite existing import/{dataset_name}/entity_classes.json."
    )

    parser.add_argument(
        "--entity_class_max_classes",
        type=int,
        default=4,
        help="Maximum number of classes for each entity."
    )

    parser.add_argument(
        "--entity_class_max_workers",
        type=int,
        default=4,
        help="Max workers for entity class inference."
    )

    parser.add_argument(
        "--use_class_schema",
        action="store_false",
        help="Enable ontology class schema enhancement."
    )

    parser.add_argument(
        "--entity_classes_path",
        type=str,
        default="import/2wikimultihop/entity_classes.json",
        help="Path to entity_classes.json."
    )

    parser.add_argument(
        "--schema_state_path",
        type=str,
        default="import/2wikimultihop/schema_state_incremental.json",
        help="Path to schema_state json."
    )

    parser.add_argument(
        "--query_type_cache_path",
        type=str,
        default="import/2wikimultihop/query_types.json",
        help="Path to query type cache json."
    )

    parser.add_argument(
        "--disable_llm_query_type_inference",
        action="store_false",
        help="Disable LLM fallback for query type inference."
    )

    parser.add_argument(
        "--class_boost_alpha",
        type=float,
        default=0.3,
        help="Class-aware boost coefficient for entity propagation."
    )

    parser.add_argument(
        "--ppr_class_prior_gamma",
        type=float,
        default=0.3,
        help="Class-aware prior coefficient for PPR reset."
    )

    return parser.parse_args()


def normalize_passages(chunks):
    """
    Ensure every passage has exactly one leading index.

    If chunks are already:
        0:text
        1:text

    keep them as they are.

    If chunks are plain text:
        text
        text

    convert them to:
        0:text
        1:text
    """
    if not chunks:
        return []

    if all(isinstance(chunk, str) and re.match(r"^\d+:", chunk.strip()) for chunk in chunks):
        return chunks

    return [f"{idx}:{chunk}" for idx, chunk in enumerate(chunks)]


def load_dataset(dataset_name):
    questions_path = f"dataset/{dataset_name}/questions.json"
    chunks_path = f"dataset/{dataset_name}/chunks.json"

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    passages = normalize_passages(chunks)

    return questions, passages


def load_embedding_model(embedding_model_path):
    return SentenceTransformer(embedding_model_path, device="cuda")


def main():
    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    args = parse_arguments()

    result_dir = f"results/{args.dataset_name}/{time_str}"
    os.makedirs(result_dir, exist_ok=True)
    setup_logging(f"{result_dir}/log.txt")

    print(f"Dataset: {args.dataset_name}")
    print(f"Result directory: {result_dir}")

    embedding_model = load_embedding_model(args.embedding_model)
    questions, passages = load_dataset(args.dataset_name)

    print("First 3 passages:")
    for passage in passages[:3]:
        print(passage[:160])

    llm_model = LLM_Model(args.llm_model)

    config = ClassRAGConfig(
        dataset_name=args.dataset_name,
        embedding_model=embedding_model,
        spacy_model=args.spacy_model,
        max_workers=args.max_workers,
        llm_model=llm_model,
        max_iterations=args.max_iterations,
        iteration_threshold=args.iteration_threshold,
        passage_ratio=args.passage_ratio,
        top_k_sentence=args.top_k_sentence,
        use_vectorized_retrieval=args.use_vectorized_retrieval,

        entity_class_only=args.entity_class_only,
        entity_class_overwrite=args.entity_class_overwrite,
        entity_class_max_classes=args.entity_class_max_classes,
        entity_class_max_workers=args.entity_class_max_workers,

        use_class_schema=args.use_class_schema,
        entity_classes_path=args.entity_classes_path,
        schema_state_path=args.schema_state_path,
        query_type_cache_path=args.query_type_cache_path,
        enable_llm_query_type_inference=not args.disable_llm_query_type_inference,
        class_boost_alpha=args.class_boost_alpha,
        ppr_class_prior_gamma=args.ppr_class_prior_gamma,
    )

    rag_model = ClassRAG(global_config=config)
    rag_model.index(passages)

    if args.entity_class_only:
        print("Entity class extraction finished.")
        print(f"Saved to: import/{args.dataset_name}/entity_classes.json")
        print("Exit before QA and evaluation.")
        return

    questions = rag_model.qa(questions)

    predictions_path = f"{result_dir}/predictions.json"
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)

    evaluator = Evaluator(
        llm_model=llm_model,
        predictions_path=predictions_path
    )
    evaluator.evaluate(max_workers=args.max_workers)


if __name__ == "__main__":
    main()