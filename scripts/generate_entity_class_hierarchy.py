#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import argparse
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import LLM_Model, setup_logging


logger = logging.getLogger(__name__)


class EntityClassHierarchyInferer:
    def __init__(
        self,
        llm_model,
        output_path,
        schema_output_path,
        max_workers=8,
        overwrite=False,
        max_paths=4,
        max_path_len=4,
    ):
        self.llm_model = llm_model
        self.output_path = output_path
        self.schema_output_path = schema_output_path
        self.max_workers = max_workers
        self.overwrite = overwrite
        self.max_paths = max_paths
        self.max_path_len = max_path_len

        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        schema_dir = os.path.dirname(self.schema_output_path)
        if schema_dir:
            os.makedirs(schema_dir, exist_ok=True)

        self.cache = self._load_existing_results()

    def _load_existing_results(self):
        if os.path.exists(self.output_path) and not self.overwrite:
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Loaded existing results from: {self.output_path}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load existing results: {e}")
        return {}

    def _save_results(self):
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=4)

    def _save_schema(self):
        schema = self.build_schema(self.cache)
        with open(self.schema_output_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False, indent=4)
        return schema

    def build_prompt(self, entity_text):
        system_prompt = (
            "You are an expert semantic type annotator for named entities. "
            "Your task is to infer one or more semantic class hierarchy paths for an entity. "
            "You must output only one valid JSON object after 'Answer:'. "
            "Do not output explanations, markdown, bullet points, or plain text."
        )

        user_prompt = f"""
Entity: {entity_text}

Task:
Infer semantic class hierarchy paths for this entity.

Output format:
Answer: {{
  "class_paths": [
    ["Entity", "Parent class", "Child class", "Most specific class"]
  ]
}}

Rules:
1. Return at most {self.max_paths} hierarchy paths.
2. Each path must go from broad class to specific class.
3. The first class should be "Entity".
4. The last class must be the most specific useful class for this entity.
5. If the entity belongs to multiple semantic categories, return multiple paths.
6. Each path should contain 2 to {self.max_path_len} classes.
7. Prefer stable, general, reusable classes.
8. Avoid overly rare, historical, regional, dynasty-specific, or noisy labels.
9. Avoid duplicate or near-duplicate paths.
10. Do not include confidence scores or explanations.
11. Do not output a flat class list.

Good outputs:

Entity: Paris
Answer: {{
  "class_paths": [
    ["Entity", "Place", "City", "Capital city"]
  ]
}}

Entity: Apple
Answer: {{
  "class_paths": [
    ["Entity", "Organization", "Company", "Technology company"],
    ["Entity", "Product brand", "Consumer brand"],
    ["Entity", "Biological object", "Fruit"]
  ]
}}

Entity: Albert Einstein
Answer: {{
  "class_paths": [
    ["Entity", "Person", "Scientist", "Physicist"],
    ["Entity", "Person", "Historical figure"]
  ]
}}

Entity: World War II
Answer: {{
  "class_paths": [
    ["Entity", "Event", "Conflict", "War"],
    ["Entity", "Event", "Historical event"]
  ]
}}

Bad output:
Answer: ["Place", "City", "Capital city"]

Reason:
This is bad because it is a flat class list, not hierarchy paths.

Now annotate the given entity.
You must output exactly this format:
Answer: {{
  "class_paths": [
    ["Entity", "Parent class", "Child class", "Most specific class"]
  ]
}}
""".strip()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def build_retry_prompt(self, entity_text):
        system_prompt = (
            "You must return only a valid JSON object. "
            "No explanation. No markdown. No plain text."
        )

        user_prompt = f"""
Entity: {entity_text}

Return at most {self.max_paths} semantic class hierarchy paths.

Correct format:
Answer: {{
  "class_paths": [
    ["Entity", "Person", "Scientist", "Physicist"]
  ]
}}

Output only the correct format.
""".strip()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def parse_class_paths(self, raw_output):
        if raw_output is None:
            return []

        text = str(raw_output).strip()

        text = re.sub(r"^\s*Answer\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^```\s*", "", text).strip()
        text = re.sub(r"\s*```$", "", text).strip()

        parsed = None

        try:
            parsed = json.loads(text)
        except Exception:
            pass

        if parsed is None:
            obj_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if obj_match:
                try:
                    parsed = json.loads(obj_match.group(0))
                except Exception:
                    pass

        if parsed is None:
            array_match = re.search(r"\[.*\]", text, flags=re.DOTALL)
            if array_match:
                try:
                    parsed = json.loads(array_match.group(0))
                except Exception:
                    pass

        if parsed is None:
            return []

        raw_paths = []

        if isinstance(parsed, dict):
            for key in [
                "class_paths",
                "paths",
                "hierarchy_paths",
                "semantic_class_paths",
                "type_paths",
            ]:
                if key in parsed and isinstance(parsed[key], list):
                    raw_paths = parsed[key]
                    break

            if not raw_paths:
                for key in ["classes", "types", "semantic_classes"]:
                    if key in parsed and isinstance(parsed[key], list):
                        raw_paths = [["Entity"] + parsed[key]]
                        break

        elif isinstance(parsed, list):
            if all(isinstance(x, list) for x in parsed):
                raw_paths = parsed
            else:
                raw_paths = [["Entity"] + parsed]

        return self._clean_class_paths(raw_paths)

    def _clean_class_paths(self, raw_paths):
        cleaned_paths = []
        seen_paths = set()

        for path in raw_paths:
            if not isinstance(path, list):
                continue

            cleaned = []
            local_seen = set()

            for c in path:
                if c is None:
                    continue

                c = str(c).strip()
                c = re.sub(r"\s+", " ", c)
                c = c.strip(" .。,:;；，[]{}()\"'")

                if not self._is_valid_class(c):
                    continue

                c = self._normalize_class_capitalization(c)
                key = c.lower()

                if key in local_seen:
                    continue

                local_seen.add(key)
                cleaned.append(c)

            if not cleaned:
                continue

            if cleaned[0].lower() != "entity":
                cleaned = ["Entity"] + cleaned

            cleaned = cleaned[: self.max_path_len]

            if len(cleaned) < 2:
                continue

            path_key = tuple(x.lower() for x in cleaned)
            if path_key in seen_paths:
                continue

            seen_paths.add(path_key)
            cleaned_paths.append(cleaned)

            if len(cleaned_paths) >= self.max_paths:
                break

        return cleaned_paths

    def _is_valid_class(self, c):
        if not c:
            return False

        if len(c) > 50:
            return False

        if len(c.split()) > 4:
            return False

        lowered = c.lower()

        invalid = {
            "unknown",
            "none",
            "null",
            "n/a",
            "answer",
            "class",
            "classes",
            "type",
            "types",
        }

        if lowered in invalid:
            return False

        bad_fragments = [
            "because",
            "reason",
            "output",
            "format",
            "json",
            "confidence",
            "explanation",
        ]

        if any(x in lowered for x in bad_fragments):
            return False

        if any(ch in c for ch in ["{", "}", "[", "]", "`"]):
            return False

        return True

    def _normalize_class_capitalization(self, class_name):
        class_name = str(class_name).strip()
        class_name = re.sub(r"\s+", " ", class_name)

        if not class_name:
            return class_name

        if class_name.lower() == "entity":
            return "Entity"

        if any(ch.isupper() for ch in class_name[1:]):
            return class_name[0].upper() + class_name[1:]

        words = class_name.lower().split()

        if len(words) == 1:
            return words[0].capitalize()

        return words[0].capitalize() + " " + " ".join(words[1:])

    def infer_one_entity(self, entity_hash_id, entity_text):
        try:
            messages = self.build_prompt(entity_text)
            raw_output = self.llm_model.infer_raw(messages)
            class_paths = self.parse_class_paths(raw_output)

            if len(class_paths) == 0:
                logger.warning(f"Empty class paths for [{entity_text}], retrying once.")
                retry_messages = self.build_retry_prompt(entity_text)
                raw_output_retry = self.llm_model.infer_raw(retry_messages)
                class_paths_retry = self.parse_class_paths(raw_output_retry)

                if len(class_paths_retry) > 0:
                    raw_output = raw_output_retry
                    class_paths = class_paths_retry

            leaf_classes = sorted({path[-1] for path in class_paths if len(path) > 0})

            return entity_hash_id, {
                "entity": entity_text,
                "classes": leaf_classes,
                "class_paths": class_paths,
                "raw_output": raw_output,
                "status": "success" if len(class_paths) > 0 else "empty",
            }

        except Exception as e:
            logger.error(f"Failed to infer hierarchy for entity [{entity_text}]: {e}")
            return entity_hash_id, {
                "entity": entity_text,
                "classes": [],
                "class_paths": [],
                "raw_output": "",
                "status": "failed",
                "error": str(e),
            }

    def annotate_entities(self, entity_hash_id_to_text):
        pending_items = []

        for entity_hash_id, entity_text in entity_hash_id_to_text.items():
            if not self.overwrite and entity_hash_id in self.cache:
                old_item = self.cache[entity_hash_id]
                if isinstance(old_item, dict) and old_item.get("class_paths"):
                    continue

            pending_items.append((entity_hash_id, entity_text))

        logger.info(f"Total entities: {len(entity_hash_id_to_text)}")
        logger.info(f"Pending entities: {len(pending_items)}")

        if len(pending_items) == 0:
            logger.info("No pending entities. Use cached results directly.")
            self._save_schema()
            return self.cache

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.infer_one_entity, entity_hash_id, entity_text)
                for entity_hash_id, entity_text in pending_items
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Inferring Entity Class Hierarchies",
            ):
                entity_hash_id, result = future.result()
                self.cache[entity_hash_id] = result

                self._save_results()
                self._save_schema()

        self._save_results()
        self._save_schema()

        logger.info(f"Entity class hierarchy saved to: {self.output_path}")
        logger.info(f"Class hierarchy schema saved to: {self.schema_output_path}")

        return self.cache

    def build_schema(self, entity_results):
        parent_to_children = defaultdict(set)
        child_to_parents = defaultdict(set)
        class_to_entities = defaultdict(set)
        leaf_class_to_entities = defaultdict(set)

        for entity_hash_id, item in entity_results.items():
            if not isinstance(item, dict):
                continue

            class_paths = item.get("class_paths", [])

            for path in class_paths:
                if not isinstance(path, list) or len(path) < 2:
                    continue

                path = [str(x).strip() for x in path if str(x).strip()]

                if len(path) < 2:
                    continue

                for c in path:
                    class_to_entities[c].add(entity_hash_id)

                leaf_class_to_entities[path[-1]].add(entity_hash_id)

                for parent, child in zip(path[:-1], path[1:]):
                    if parent == child:
                        continue
                    parent_to_children[parent].add(child)
                    child_to_parents[child].add(parent)

        return {
            "parent_to_children": {
                k: sorted(v) for k, v in sorted(parent_to_children.items())
            },
            "child_to_parents": {
                k: sorted(v) for k, v in sorted(child_to_parents.items())
            },
            "class_to_entities": {
                k: sorted(v) for k, v in sorted(class_to_entities.items())
            },
            "leaf_class_to_entities": {
                k: sorted(v) for k, v in sorted(leaf_class_to_entities.items())
            },
            "metadata": {
                "num_entities": len(entity_results),
                "num_classes": len(class_to_entities),
                "num_parent_classes": len(parent_to_children),
                "num_child_classes": len(child_to_parents),
            },
        }


def load_entities_from_entity_embedding(entity_embedding_path):
    if not os.path.exists(entity_embedding_path):
        raise FileNotFoundError(f"Entity embedding file not found: {entity_embedding_path}")

    df = pd.read_parquet(entity_embedding_path)

    if "hash_id" not in df.columns or "text" not in df.columns:
        raise ValueError(
            f"Expected columns ['hash_id', 'text'] in {entity_embedding_path}, "
            f"but got columns: {list(df.columns)}"
        )

    entity_hash_id_to_text = {}

    for _, row in df.iterrows():
        entity_hash_id = str(row["hash_id"]).strip()
        entity_text = str(row["text"]).strip()

        if entity_hash_id and entity_text:
            entity_hash_id_to_text[entity_hash_id] = entity_text

    return entity_hash_id_to_text


def build_test_entities(entity_texts):
    entity_hash_id_to_text = {}

    for idx, entity in enumerate(entity_texts):
        entity = entity.strip()
        if not entity:
            continue
        entity_hash_id_to_text[f"test-entity-{idx}"] = entity

    return entity_hash_id_to_text


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--llm_model",
        type=str,
        default="/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="2wikimultihop",
    )

    parser.add_argument(
        "--working_dir",
        type=str,
        default="import",
    )

    parser.add_argument(
        "--entity_embedding_path",
        type=str,
        default="",
        help="Optional. If empty, use import/{dataset_name}/entity_embedding.parquet",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Optional. If empty, use import/{dataset_name}/entity_class_paths.json",
    )

    parser.add_argument(
        "--schema_output_path",
        type=str,
        default="",
        help="Optional. If empty, use import/{dataset_name}/class_hierarchy_schema.json",
    )

    parser.add_argument(
        "--entities",
        type=str,
        default="",
        help="Optional comma-separated entities for quick test. If set, parquet will not be loaded.",
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max_paths",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max_path_len",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    dataset_dir = os.path.join(args.working_dir, args.dataset_name)

    entity_embedding_path = args.entity_embedding_path
    if not entity_embedding_path:
        entity_embedding_path = os.path.join(dataset_dir, "entity_embedding.parquet")

    output_path = args.output_path
    if not output_path:
        output_path = os.path.join(dataset_dir, "entity_class_paths.json")

    schema_output_path = args.schema_output_path
    if not schema_output_path:
        schema_output_path = os.path.join(dataset_dir, "class_hierarchy_schema.json")

    log_path = os.path.join(dataset_dir, "entity_class_hierarchy_log.txt")
    os.makedirs(dataset_dir, exist_ok=True)
    setup_logging(log_path)

    logger.info("Starting entity class hierarchy inference.")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Entity embedding path: {entity_embedding_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Schema output path: {schema_output_path}")

    if args.entities:
        entity_texts = [e.strip() for e in args.entities.split(",") if e.strip()]
        entity_hash_id_to_text = build_test_entities(entity_texts)
        logger.info(f"Using test entities: {entity_texts}")
    else:
        entity_hash_id_to_text = load_entities_from_entity_embedding(entity_embedding_path)
        logger.info(f"Loaded {len(entity_hash_id_to_text)} entities from entity embedding store.")

    llm_model = LLM_Model(args.llm_model)

    inferer = EntityClassHierarchyInferer(
        llm_model=llm_model,
        output_path=output_path,
        schema_output_path=schema_output_path,
        max_workers=args.max_workers,
        overwrite=args.overwrite,
        max_paths=args.max_paths,
        max_path_len=args.max_path_len,
    )

    results = inferer.annotate_entities(entity_hash_id_to_text)
    schema = inferer.build_schema(results)

    print("\n========== Entity Class Hierarchy Results ==========")
    print(f"Total entities: {len(results)}")
    print(f"Total classes: {schema['metadata']['num_classes']}")
    print(f"Saved entity paths to: {output_path}")
    print(f"Saved hierarchy schema to: {schema_output_path}")

    print("\nPreview:")
    for idx, (entity_hash_id, item) in enumerate(results.items()):
        if idx >= 10:
            break
        print(f"\n[{entity_hash_id}] {item.get('entity')}")
        print(f"Classes: {item.get('classes')}")
        print(f"Class paths: {item.get('class_paths')}")
        print(f"Status: {item.get('status')}")

    print("\n====================================================")


if __name__ == "__main__":
    main()