import os
import re
import sys
import json
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ---------------------------------------------------------------------
# Make sure this file can be run directly from project root:
# python src/entity_class_inferer.py
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.utils import LLM_Model, setup_logging


logger = logging.getLogger(__name__)


class EntityClassInferer:
    """
    Use an LLM to infer one or multiple semantic classes for each entity.

    Current design:
    1. Only use entity name as input.
    2. Allow each entity to have multiple intuitive classes.
    3. Prefer broad and human-readable classes.
    4. Save results into an independent JSON file.
    5. Do not modify graph structure.
    """

    def __init__(
        self,
        llm_model,
        output_path,
        max_workers=8,
        overwrite=False,
        max_classes=4,
    ):
        self.llm_model = llm_model
        self.output_path = output_path
        self.max_workers = max_workers
        self.overwrite = overwrite
        self.max_classes = max_classes

        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.entity_class_cache = self._load_existing_results()

        # Used only as a fallback parser for malformed outputs such as:
        # "place city capital city"
        # It is not used as a closed taxonomy for prompting.
        self.common_multiword_classes = {
            "capital city",
            "historical figure",
            "fictional character",
            "literary work",
            "written work",
            "artistic work",
            "work of art",
            "musical work",
            "religious figure",
            "royal figure",
            "sports team",
            "political party",
            "government agency",
            "educational institution",
            "research institution",
            "technology company",
            "media company",
            "product brand",
            "brand name",
            "geographic region",
            "natural feature",
            "body of water",
            "mountain range",
            "historical event",
            "military conflict",
            "scientific concept",
            "cultural concept",
            "legal concept",
            "programming language",
            "operating system",
            "software product",
        }

    def _load_existing_results(self):
        """
        Load previous entity-class results if the output file already exists.
        This avoids repeated LLM calls.
        """
        if os.path.exists(self.output_path) and not self.overwrite:
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Loaded existing entity class results from: {self.output_path}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load existing entity class results: {e}")
                return {}
        return {}

    def _save_results(self):
        """
        Save current annotation results to JSON.
        """
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.entity_class_cache, f, ensure_ascii=False, indent=4)

    def build_prompt(self, entity_text):
        """
        Build a strict prompt for intuitive multi-class entity classification.

        Expected output:
            Answer: ["Place", "City", "Capital city"]

        Important:
        - Each class must be an independent JSON string.
        - Do not output words separated by spaces.
        - Prefer 2-4 intuitive classes.
        """

        system_prompt = (
            "You are an expert semantic type annotator for named entities. "
            "Your task is to assign a small number of intuitive semantic classes to an entity. "
            "You must output only one valid JSON array after 'Answer:'. "
            "Each class must be an independent string inside the array. "
            "Do not output explanations, markdown, bullet points, plain words, or space-separated labels."
        )

        user_prompt = f"""
Entity:
{entity_text}

Task:
Assign the most intuitive semantic classes for this entity.

Rules:
1. Return at most {self.max_classes} classes.
2. Prefer 2 to 4 classes.
3. Each class must be a separate string in a JSON array.
4. Use broad and natural classes that humans can immediately understand.
5. Avoid overly specific classes.
6. Avoid rare historical, regional, dynasty-specific, or overly detailed labels.
7. Avoid redundant labels with nearly the same meaning.
8. Do not output a long chain of related labels.
9. Do not output labels separated only by spaces.
10. Do not include confidence scores or explanations.

Good outputs:

Entity: Paris
Answer: ["Place", "City", "Capital city"]

Entity: Apple
Answer: ["Organization", "Brand", "Fruit"]

Entity: Teutberga
Answer: ["Person", "Monarch", "Historical figure"]

Entity: Albert Einstein
Answer: ["Person", "Scientist", "Historical figure"]

Entity: World War II
Answer: ["Event", "War", "Historical event"]

Bad outputs:

Entity: Apple
Answer: organization brand fruit

Reason: this is bad because the classes are not a JSON array.

Entity: Apple
Answer: ["Organization brand fruit"]

Reason: this is bad because multiple classes are merged into one string.

Entity: Teutberga
Answer: ["Person", "Monarch", "Medieval monarch", "Carolingian ruler", "European monarch", "Ruler", "Royalty", "Historical figure"]

Reason: this is bad because it contains too many overlapping and overly specific labels.

Now annotate the given entity.

You must output exactly this format:
Answer: ["Class 1", "Class 2", "Class 3"]
""".strip()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def build_retry_prompt(self, entity_text):
        """
        A stricter retry prompt used when the first output is not parseable.
        """

        system_prompt = (
            "You must return only a valid JSON array of entity classes. "
            "No explanation. No markdown. No plain text. No space-separated labels."
        )

        user_prompt = f"""
Entity: {entity_text}

Return at most {self.max_classes} intuitive entity classes.

Correct format:
Answer: ["Person", "Monarch", "Historical figure"]

Incorrect format:
Answer: person monarch historical figure

Output only the correct format.
""".strip()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def parse_class_list(self, raw_output):
        """
        Parse LLM output into a clean list of class strings.

        Supported good formats:
            ["Person", "Monarch", "Historical figure"]
            Answer: ["Person", "Monarch", "Historical figure"]
            {"classes": ["Person", "Monarch", "Historical figure"]}
            Person, Monarch, Historical figure

        Also handles short malformed outputs:
            "organization brand fruit" -> ["Organization", "Brand", "Fruit"]
            "place city capital city" -> ["Place", "City", "Capital city"]
        """

        if raw_output is None:
            return []

        text = str(raw_output).strip()

        # Remove possible Answer prefix
        text = re.sub(r"^\s*Answer\s*:\s*", "", text, flags=re.IGNORECASE).strip()

        # Remove possible tags
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()

        # Remove markdown fences
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^```\s*", "", text).strip()
        text = re.sub(r"\s*```$", "", text).strip()

        # 1. Direct JSON parsing
        try:
            parsed = json.loads(text)

            if isinstance(parsed, list):
                return self._clean_classes(parsed)

            if isinstance(parsed, dict):
                for key in ["classes", "entity_classes", "types", "semantic_classes"]:
                    if key in parsed and isinstance(parsed[key], list):
                        return self._clean_classes(parsed[key])

        except Exception:
            pass

        # 2. Extract first JSON array
        array_match = re.search(r"\[.*?\]", text, flags=re.DOTALL)
        if array_match:
            try:
                parsed = json.loads(array_match.group(0))
                if isinstance(parsed, list):
                    return self._clean_classes(parsed)
            except Exception:
                pass

        # 3. Comma / semicolon / newline separated outputs
        if "," in text or ";" in text or "\n" in text:
            candidates = re.split(r"[,;\n]+", text)
            return self._clean_classes(candidates)

        # 4. Short malformed output recovery
        # Examples:
        # "organization brand fruit"
        # "place city capital city"
        recovered = self._recover_short_space_separated_output(text)
        if recovered:
            return self._clean_classes(recovered)

        return []

    def _recover_short_space_separated_output(self, text):
        """
        Recover short malformed outputs where the model returned space-separated labels.

        Example:
            "organization brand fruit"
            -> ["Organization", "Brand", "Fruit"]

            "place city capital city"
            -> ["Place", "City", "Capital city"]

        For long outputs, return [] to force retry instead of accepting noisy labels.
        """

        text = str(text).strip().lower()
        text = text.strip(" .。,:;；，[]{}()\"'")

        if not text:
            return []

        # Remove obvious prefixes
        text = re.sub(r"^(classes|types|entity classes)\s*[:：]\s*", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        tokens = text.split()

        # Too long means it is probably an uncontrolled label chain.
        # Example:
        # "person monarch historical figure medieval monarch carolingian ruler european monarch ruler royalty"
        if len(tokens) > 6:
            logger.warning(f"Rejecting long unstructured class output: {text}")
            return []

        recovered = []
        i = 0

        while i < len(tokens):
            matched = False

            # Try longest phrase first: 3-word, then 2-word
            for n in [3, 2]:
                if i + n <= len(tokens):
                    phrase = " ".join(tokens[i:i + n])
                    if phrase in self.common_multiword_classes:
                        recovered.append(self._to_readable_class_name(phrase))
                        i += n
                        matched = True
                        break

            if matched:
                continue

            # Otherwise treat one token as one broad class
            token = tokens[i]
            if self._is_valid_single_token_class(token):
                recovered.append(self._to_readable_class_name(token))
            i += 1

        return recovered

    def _is_valid_single_token_class(self, token):
        """
        Whether a single word can be treated as a reasonable class.
        This is only used for malformed output recovery.
        """

        if not token:
            return False

        invalid_tokens = {
            "and", "or", "the", "a", "an", "of", "to", "for", "with",
            "entity", "class", "classes", "type", "types", "label", "labels",
            "answer", "none", "unknown"
        }

        if token in invalid_tokens:
            return False

        if not re.match(r"^[a-z][a-z\-]*$", token):
            return False

        return True

    def _to_readable_class_name(self, text):
        """
        Convert a class label to readable capitalization.
        """

        text = str(text).strip()
        text = re.sub(r"\s+", " ", text)

        if not text:
            return text

        words = text.split()
        if len(words) == 1:
            return words[0].capitalize()

        return words[0].capitalize() + " " + " ".join(words[1:])

    def _clean_classes(self, classes):
        """
        Clean and deduplicate class names.

        Rules:
        - each class is a string
        - remove empty values
        - normalize spaces
        - remove duplicated classes
        - remove very long or unnatural class names
        - keep at most self.max_classes classes
        """

        cleaned = []
        seen = set()

        for c in classes:
            if c is None:
                continue

            c = str(c).strip()
            c = re.sub(r"\s+", " ", c)
            c = c.strip(" .。,:;；，[]{}()\"'")

            if not c:
                continue

            # Remove invalid outputs
            if len(c) > 40:
                continue

            # Avoid long sentence-like labels
            if len(c.split()) > 3:
                continue

            # Avoid labels that are actually JSON or explanations
            if any(bad in c.lower() for bad in ["because", "reason", "output", "format"]):
                continue

            # Normalize readable capitalization
            c = self._normalize_class_capitalization(c)

            key = c.lower()
            if key in seen:
                continue

            seen.add(key)
            cleaned.append(c)

            if len(cleaned) >= self.max_classes:
                break

        return cleaned

    def _normalize_class_capitalization(self, class_name):
        """
        Normalize class name capitalization while preserving readable multi-word forms.
        """

        class_name = str(class_name).strip()
        class_name = re.sub(r"\s+", " ", class_name)

        if not class_name:
            return class_name

        # If it already contains uppercase letters in a meaningful way, keep mostly as-is
        # but ensure the first character is uppercase.
        if any(ch.isupper() for ch in class_name[1:]):
            return class_name[0].upper() + class_name[1:]

        words = class_name.lower().split()
        if len(words) == 1:
            return words[0].capitalize()

        return words[0].capitalize() + " " + " ".join(words[1:])

    def infer_one_entity(self, entity_hash_id, entity_text):
        """
        Infer semantic classes for one entity.
        If the first output is invalid, retry once with a stricter prompt.
        """

        try:
            messages = self.build_prompt(entity_text)
            raw_output = self.llm_model.infer_raw(messages)
            classes = self.parse_class_list(raw_output)

            # Retry once if parsing failed
            if len(classes) == 0:
                logger.warning(f"Empty or invalid classes for [{entity_text}], retrying once.")

                retry_messages = self.build_retry_prompt(entity_text)
                raw_output_retry = self.llm_model.infer_raw(retry_messages)
                classes_retry = self.parse_class_list(raw_output_retry)

                if len(classes_retry) > 0:
                    raw_output = raw_output_retry
                    classes = classes_retry

            return entity_hash_id, {
                "entity": entity_text,
                "classes": classes,
                "raw_output": raw_output,
                "status": "success" if len(classes) > 0 else "empty",
            }

        except Exception as e:
            logger.error(f"Failed to infer classes for entity [{entity_text}]: {e}")

            return entity_hash_id, {
                "entity": entity_text,
                "classes": [],
                "raw_output": "",
                "status": "failed",
                "error": str(e),
            }

    def annotate_entities(self, entity_hash_id_to_text):
        """
        Annotate all entities.

        Input:
        {
            "entity-test-0": "Paris",
            "entity-test-1": "Apple"
        }

        Output:
        {
            "entity-test-0": {
                "entity": "Paris",
                "classes": ["Place", "City", "Capital city"],
                "raw_output": "...",
                "status": "success"
            }
        }
        """

        pending_items = []

        for entity_hash_id, entity_text in entity_hash_id_to_text.items():
            if not self.overwrite and entity_hash_id in self.entity_class_cache:
                continue
            pending_items.append((entity_hash_id, entity_text))

        logger.info(f"Total entities: {len(entity_hash_id_to_text)}")
        logger.info(f"Pending entities: {len(pending_items)}")

        if len(pending_items) == 0:
            logger.info("No pending entities. Use cached results directly.")
            return self.entity_class_cache

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.infer_one_entity, entity_hash_id, entity_text)
                for entity_hash_id, entity_text in pending_items
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Inferring Entity Classes"):
                entity_hash_id, result = future.result()
                self.entity_class_cache[entity_hash_id] = result

                # Save progressively to prevent loss if interrupted
                self._save_results()

        self._save_results()
        logger.info(f"Entity class results saved to: {self.output_path}")

        return self.entity_class_cache


def build_test_entities(entity_texts):
    """
    Build fake entity_hash_id_to_text for independent testing.
    This does not depend on EntityEmbeddingStore.
    """

    entity_hash_id_to_text = {}

    for idx, entity in enumerate(entity_texts):
        entity = entity.strip()
        if not entity:
            continue

        entity_hash_id = f"test-entity-{idx}"
        entity_hash_id_to_text[entity_hash_id] = entity

    return entity_hash_id_to_text


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--llm_model",
        type=str,
        default="/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
        help="The LLM model name or path used by your OpenAI-compatible endpoint."
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./import/entity_class_test/entity_classes_test.json",
        help="Where to save test entity class results."
    )

    parser.add_argument(
        "--entities",
        type=str,
        default="Paris,Apple,Teutberga,Lothair II,Amazon,The Odyssey,World War II,Google",
        help="Comma-separated entity names for quick testing."
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of parallel LLM calls."
    )

    parser.add_argument(
        "--max_classes",
        type=int,
        default=4,
        help="Maximum number of classes for each entity."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing cached entity class results."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    setup_logging("./import/entity_class_test/log.txt")

    logger.info("Starting standalone EntityClassInferer test.")
    logger.info(f"LLM model: {args.llm_model}")
    logger.info(f"Output path: {args.output_path}")

    entity_texts = [e.strip() for e in args.entities.split(",") if e.strip()]
    entity_hash_id_to_text = build_test_entities(entity_texts)

    logger.info(f"Test entities: {entity_texts}")

    llm_model = LLM_Model(args.llm_model)

    inferer = EntityClassInferer(
        llm_model=llm_model,
        output_path=args.output_path,
        max_workers=args.max_workers,
        overwrite=args.overwrite,
        max_classes=args.max_classes,
    )

    results = inferer.annotate_entities(entity_hash_id_to_text)

    print("\n========== Entity Class Inference Results ==========")
    for entity_hash_id, item in results.items():
        print(f"\n[{entity_hash_id}] {item.get('entity')}")
        print(f"Classes: {item.get('classes')}")
        print(f"Status: {item.get('status')}")
        print(f"Raw Output: {item.get('raw_output')}")
    print("\n====================================================")
    print(f"Saved to: {args.output_path}")

# python src/entity_class_inferer.py --overwrite
# python run.py   --dataset_name hotpotqa   --entity_class_only   --entity_class_max_classes 4   --entity_class_max_workers 16

if __name__ == "__main__":
    main()