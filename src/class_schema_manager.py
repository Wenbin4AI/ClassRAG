import os
import re
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class OntologySchemaManager:
    """
    Load entity class annotations and induced ontology schema.

    Main functions:
    1. entity_hash_id -> canonical classes
    2. class compatibility computation
    """

    def __init__(
        self,
        entity_classes_path,
        schema_state_path,
        same_weight=1.0,
        descendant_weight=0.9,
        ancestor_weight=0.5,
        related_weight=0.3,
    ):
        self.entity_classes_path = entity_classes_path
        self.schema_state_path = schema_state_path

        self.same_weight = same_weight
        self.descendant_weight = descendant_weight
        self.ancestor_weight = ancestor_weight
        self.related_weight = related_weight

        self.canonical_map = {}
        self.parent_to_children = defaultdict(set)
        self.child_to_parents = defaultdict(set)
        self.related_adj = defaultdict(set)

        self.entity_hash_id_to_classes = {}

        self._load_schema()
        self._load_entity_classes()

    def _normalize_class_name(self, cls):
        if cls is None:
            return ""
        cls = str(cls).strip()
        cls = re.sub(r"\s+", " ", cls)
        cls = cls.strip(" .。,:;；，[]{}()\"'")
        return cls

    def _load_schema(self):
        if not self.schema_state_path or not os.path.exists(self.schema_state_path):
            logger.warning(f"Schema state file not found: {self.schema_state_path}")
            return

        with open(self.schema_state_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.canonical_map = data.get("canonical_map", {}) or {}

        for edge in data.get("subclass_edges", []):
            parent = self.canonicalize_class(edge.get("parent", ""))
            child = self.canonicalize_class(edge.get("child", ""))

            if parent and child and parent != child:
                self.parent_to_children[parent].add(child)
                self.child_to_parents[child].add(parent)

        for edge in data.get("related_edges", []):
            source = self.canonicalize_class(edge.get("source", ""))
            target = self.canonicalize_class(edge.get("target", ""))

            if source and target and source != target:
                self.related_adj[source].add(target)
                self.related_adj[target].add(source)

        logger.info(f"Loaded schema from: {self.schema_state_path}")
        logger.info(f"Canonical map size: {len(self.canonical_map)}")
        logger.info(f"Subclass parents: {len(self.parent_to_children)}")
        logger.info(f"Related nodes: {len(self.related_adj)}")

    def _load_entity_classes(self):
        if not self.entity_classes_path or not os.path.exists(self.entity_classes_path):
            logger.warning(f"Entity classes file not found: {self.entity_classes_path}")
            return

        with open(self.entity_classes_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entity_hash_id, item in data.items():
            classes = item.get("classes", [])

            if not isinstance(classes, list):
                continue

            canonical_classes = self.canonicalize_classes(classes)

            if canonical_classes:
                self.entity_hash_id_to_classes[entity_hash_id] = canonical_classes

        logger.info(f"Loaded entity classes from: {self.entity_classes_path}")
        logger.info(f"Entities with class annotations: {len(self.entity_hash_id_to_classes)}")

    def canonicalize_class(self, cls):
        cls = self._normalize_class_name(cls)
        if not cls:
            return ""

        # Try exact match
        if cls in self.canonical_map:
            return self.canonical_map[cls]

        # Try case-insensitive match
        lower_cls = cls.lower()
        for raw_cls, canonical_cls in self.canonical_map.items():
            if str(raw_cls).lower() == lower_cls:
                return canonical_cls

        return cls

    def canonicalize_classes(self, classes):
        results = []
        seen = set()

        for cls in classes:
            canonical = self.canonicalize_class(cls)

            if not canonical:
                continue

            key = canonical.lower()
            if key not in seen:
                seen.add(key)
                results.append(canonical)

        return results

    def is_ancestor(self, ancestor, descendant):
        ancestor = self.canonicalize_class(ancestor)
        descendant = self.canonicalize_class(descendant)

        if not ancestor or not descendant:
            return False

        if ancestor == descendant:
            return True

        visited = set()
        stack = list(self.parent_to_children.get(ancestor, set()))

        while stack:
            node = stack.pop()

            if node == descendant:
                return True

            if node in visited:
                continue

            visited.add(node)
            stack.extend(self.parent_to_children.get(node, set()))

        return False

    def is_related(self, class_a, class_b):
        class_a = self.canonicalize_class(class_a)
        class_b = self.canonicalize_class(class_b)

        if not class_a or not class_b:
            return False

        return class_b in self.related_adj.get(class_a, set())

    def get_entity_classes(self, entity_hash_id):
        return self.entity_hash_id_to_classes.get(entity_hash_id, [])

    def compatibility_between_classes(self, entity_class, target_class):
        """
        Compute compatibility between one entity class and one query target class.

        Logic:
        1. Same / equivalent canonical class: high score
        2. Entity class is descendant of target class: high score
        3. Entity class is ancestor of target class: medium score
        4. Related relation: weak score
        5. Otherwise: 0
        """
        entity_class = self.canonicalize_class(entity_class)
        target_class = self.canonicalize_class(target_class)

        if not entity_class or not target_class:
            return 0.0

        if entity_class == target_class:
            return self.same_weight

        # target is parent of entity_class
        # Example: entity=City, target=Place
        if self.is_ancestor(target_class, entity_class):
            return self.descendant_weight

        # entity_class is parent of target
        # Example: entity=Place, target=City
        if self.is_ancestor(entity_class, target_class):
            return self.ancestor_weight

        if self.is_related(entity_class, target_class):
            return self.related_weight

        return 0.0

    def compatibility(self, entity_classes, target_classes):
        """
        Max compatibility between any entity class and any target class.
        """
        entity_classes = self.canonicalize_classes(entity_classes)
        target_classes = self.canonicalize_classes(target_classes)

        if not entity_classes or not target_classes:
            return 0.0

        best_score = 0.0

        for entity_class in entity_classes:
            for target_class in target_classes:
                score = self.compatibility_between_classes(entity_class, target_class)
                best_score = max(best_score, score)

        return best_score

    def compatibility_for_entity(self, entity_hash_id, target_classes):
        entity_classes = self.get_entity_classes(entity_hash_id)
        return self.compatibility(entity_classes, target_classes)


class QueryTypeInferer:
    """
    Infer target answer classes for a query.

    First use simple rules.
    If rules fail and LLM inference is enabled, ask LLM.
    """

    def __init__(
        self,
        schema_manager,
        llm_model=None,
        cache_path=None,
        enable_llm=True,
    ):
        self.schema_manager = schema_manager
        self.llm_model = llm_model
        self.cache_path = cache_path
        self.enable_llm = enable_llm

        self.cache = self._load_cache()

    def _load_cache(self):
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load query type cache: {e}")
        return {}

    def _save_cache(self):
        if not self.cache_path:
            return

        output_dir = os.path.dirname(self.cache_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=4)

    def infer(self, question):
        question_key = str(question).strip()

        if question_key in self.cache:
            return self.cache[question_key]

        target_classes = self._rule_based_infer(question)

        if not target_classes and self.enable_llm and self.llm_model is not None:
            target_classes = self._llm_based_infer(question)

        target_classes = self.schema_manager.canonicalize_classes(target_classes)

        self.cache[question_key] = target_classes
        self._save_cache()

        return target_classes

    def _rule_based_infer(self, question):
        q = question.lower()
        tokens = set(re.findall(r"\w+", q))

        # Location / place questions
        if (
            "where" in tokens
            or "located" in tokens
            or "location" in tokens
            or "place" in tokens
            or "capital" in tokens
        ):
            return ["Place", "City", "Country", "Location"]

        # Birth/place pattern
        if "born" in tokens or "birthplace" in tokens:
            return ["Place", "City", "Country", "Location"]

        # Time/date questions
        if (
            "when" in tokens
            or "date" in tokens
            or "year" in tokens
            or "time" in tokens
        ):
            return ["Date", "Time", "Year"]

        # Person questions
        if "who" in tokens or "whom" in tokens:
            return ["Person", "Organization"]

        # Organization/company questions
        if (
            "company" in tokens
            or "organization" in tokens
            or "institution" in tokens
            or "university" in tokens
        ):
            return ["Organization", "Company", "Institution"]

        # Number questions
        if (
            "how" in tokens and "many" in tokens
        ) or "number" in tokens or "count" in tokens or "population" in tokens:
            return ["Number", "Quantity"]

        # Country/city explicit questions
        if "country" in tokens:
            return ["Country", "Place"]

        if "city" in tokens:
            return ["City", "Place"]

        if "language" in tokens:
            return ["Language"]

        return []

    def _call_llm_raw(self, messages):
        if hasattr(self.llm_model, "infer_raw"):
            return self.llm_model.infer_raw(messages)
        return self.llm_model.infer(messages)

    def _llm_based_infer(self, question):
        system_prompt = (
            "You are an expert query answer-type classifier. "
            "Given a question, infer the expected semantic classes of the answer. "
            "Return only a valid JSON list of short class names."
        )

        user_prompt = f"""
Question:
{question}

Task:
Infer the expected answer type classes for this question.

Rules:
1. Return 1 to 4 concise class names.
2. Use intuitive ontology classes such as Person, Place, City, Country, Organization, Date, Number, Work, Event, Concept.
3. Do not explain.
4. Return JSON list only.

Examples:
Question: Where was Albert Einstein born?
Answer: ["Place", "City", "Country"]

Question: Who founded Apple?
Answer: ["Person", "Organization"]

Question: When was Apple founded?
Answer: ["Date", "Year"]

Now return only the JSON list.
""".strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw_output = self._call_llm_raw(messages)
            return self._parse_class_list(raw_output)
        except Exception as e:
            logger.warning(f"LLM query type inference failed: {e}")
            return []

    def _parse_class_list(self, raw_output):
        if raw_output is None:
            return []

        text = str(raw_output).strip()
        text = re.sub(r"^\s*Answer\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^```\s*", "", text).strip()
        text = re.sub(r"\s*```$", "", text).strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

        array_match = re.search(r"\[.*?\]", text, flags=re.DOTALL)
        if array_match:
            try:
                parsed = json.loads(array_match.group(0))
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass

        # fallback: comma split
        if "," in text:
            return [x.strip() for x in text.split(",") if x.strip()]

        return []