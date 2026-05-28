import os
import re
import sys
import json
import argparse
import logging
from itertools import combinations
from collections import Counter, defaultdict
from tqdm import tqdm


# ---------------------------------------------------------------------
# Make sure this file can be run directly from project root:
# python src/schema_state_builder.py
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.utils import LLM_Model, setup_logging


logger = logging.getLogger(__name__)


VALID_RELATIONS = {
    "EQUIVALENT",
    "A_PARENT_B",
    "B_PARENT_A",
    "RELATED",
    "UNRELATED",
    "UNCERTAIN",
}


RELATION_PRIORITY = {
    "EQUIVALENT": 4,
    "A_PARENT_B": 3,
    "B_PARENT_A": 3,
    "RELATED": 2,
    "UNRELATED": 1,
    "UNCERTAIN": 0,
}


class DisjointSetUnion:
    """
    Union-Find structure for equivalent classes.
    """

    def __init__(self, items, class_freq=None):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}
        self.class_freq = class_freq or {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def _choose_representative(self, a, b):
        """
        Choose canonical class name for an equivalent group.

        Priority:
        1. Higher frequency
        2. Shorter name
        3. Lexicographically smaller name
        """
        fa = self.class_freq.get(a, 0)
        fb = self.class_freq.get(b, 0)

        if fa > fb:
            return a, b
        if fb > fa:
            return b, a

        if len(a) < len(b):
            return a, b
        if len(b) < len(a):
            return b, a

        return (a, b) if a.lower() <= b.lower() else (b, a)

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return ra

        keep, merge = self._choose_representative(ra, rb)

        self.parent[merge] = keep

        if self.rank[keep] == self.rank[merge]:
            self.rank[keep] += 1

        return keep

    def groups(self):
        group_map = defaultdict(list)

        for x in list(self.parent.keys()):
            group_map[self.find(x)].append(x)

        return {
            rep: sorted(members)
            for rep, members in group_map.items()
        }

    def canonical_map(self):
        return {
            x: self.find(x)
            for x in list(self.parent.keys())
        }


class SchemaState:
    """
    Core schema state.

    It maintains:
    1. Equivalent classes by DSU
    2. Parent-child relations by DAG
    3. Related relations by undirected graph
    4. Comparison records and caches
    """

    def __init__(self, classes, class_freq):
        self.raw_classes = sorted(classes)
        self.class_freq = dict(class_freq)

        self.dsu = DisjointSetUnion(self.raw_classes, class_freq=self.class_freq)

        self.parent_to_children = defaultdict(set)
        self.child_to_parents = defaultdict(set)
        self.related_edges = set()

        self.comparison_cache = {}
        self.relation_records = []

        self.unrelated_pairs = set()
        self.uncertain_pairs = set()
        self.conflict_records = []

    def find(self, class_name):
        return self.dsu.find(class_name)

    def _pair_key(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        return tuple(sorted([ra, rb]))

    def already_compared(self, a, b):
        key = self._pair_key(a, b)
        return key in self.comparison_cache

    def _edge_key(self, a, b):
        return tuple(sorted([a, b]))

    def is_ancestor(self, ancestor, descendant):
        """
        Whether ancestor -> ... -> descendant exists in DAG.
        """
        ancestor = self.find(ancestor)
        descendant = self.find(descendant)

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

    def would_create_cycle(self, parent, child):
        """
        Adding parent -> child creates cycle if child is already ancestor of parent.
        """
        parent = self.find(parent)
        child = self.find(child)

        if parent == child:
            return True

        return self.is_ancestor(child, parent)

    def canonicalize_graphs(self):
        """
        Re-map all DAG and related graph nodes through DSU representatives.
        Remove self-loops and duplicated edges.
        """

        # Canonicalize DAG
        new_parent_to_children = defaultdict(set)
        new_child_to_parents = defaultdict(set)

        for parent, children in self.parent_to_children.items():
            p = self.find(parent)

            for child in children:
                c = self.find(child)

                if p == c:
                    continue

                # Avoid cycles after equivalence merging
                if self._would_create_cycle_in_maps(p, c, new_parent_to_children):
                    self.conflict_records.append({
                        "type": "cycle_after_canonicalization",
                        "parent": p,
                        "child": c,
                    })
                    continue

                new_parent_to_children[p].add(c)
                new_child_to_parents[c].add(p)

        self.parent_to_children = new_parent_to_children
        self.child_to_parents = new_child_to_parents

        # Canonicalize related graph
        new_related_edges = set()

        for a, b in self.related_edges:
            ra = self.find(a)
            rb = self.find(b)

            if ra == rb:
                continue

            # Do not keep related edge if one is ancestor of the other
            if self.is_ancestor(ra, rb) or self.is_ancestor(rb, ra):
                continue

            new_related_edges.add(self._edge_key(ra, rb))

        self.related_edges = new_related_edges

    def _would_create_cycle_in_maps(self, parent, child, parent_to_children):
        """
        Cycle check on a temporary parent_to_children map.
        """
        if parent == child:
            return True

        visited = set()
        stack = list(parent_to_children.get(child, set()))

        while stack:
            node = stack.pop()
            if node == parent:
                return True
            if node in visited:
                continue
            visited.add(node)
            stack.extend(parent_to_children.get(node, set()))

        return False

    def add_equivalent(self, a, b, record):
        """
        Add equivalent relation into DSU.
        """
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        self.dsu.union(ra, rb)
        self.relation_records.append(record)

        # After DSU union, remap graph structures
        self.canonicalize_graphs()

    def add_subclass_edge(self, parent, child, record):
        """
        Add parent -> child into DAG if it does not create cycle.
        """
        parent = self.find(parent)
        child = self.find(child)

        if parent == child:
            return

        if self.would_create_cycle(parent, child):
            self.conflict_records.append({
                "type": "cycle_conflict",
                "parent": parent,
                "child": child,
                "record": record,
            })
            return

        # If parent is already ancestor of child, this direct edge is redundant
        if self.is_ancestor(parent, child):
            self.relation_records.append({
                **record,
                "status": "skipped_redundant_parent_edge"
            })
            return

        self.parent_to_children[parent].add(child)
        self.child_to_parents[child].add(parent)

        # Remove related edge if it exists, because parent-child is stronger
        edge_key = self._edge_key(parent, child)
        if edge_key in self.related_edges:
            self.related_edges.remove(edge_key)

        self.relation_records.append(record)

        # Remove obvious transitive redundant edges
        self.remove_transitive_redundant_edges()

    def add_related_edge(self, a, b, record):
        """
        Add undirected related edge if no stronger relation already exists.
        """
        a = self.find(a)
        b = self.find(b)

        if a == b:
            return

        # If there is parent-child or ancestor-descendant relation, do not add related
        if self.is_ancestor(a, b) or self.is_ancestor(b, a):
            self.relation_records.append({
                **record,
                "status": "skipped_due_to_parent_child_path"
            })
            return

        self.related_edges.add(self._edge_key(a, b))
        self.relation_records.append(record)

    def add_unrelated(self, a, b, record):
        a = self.find(a)
        b = self.find(b)

        if a != b:
            self.unrelated_pairs.add(self._edge_key(a, b))

        self.relation_records.append(record)

    def add_uncertain(self, a, b, record):
        a = self.find(a)
        b = self.find(b)

        if a != b:
            self.uncertain_pairs.add(self._edge_key(a, b))

        self.relation_records.append(record)

    def remove_transitive_redundant_edges(self):
        """
        Remove direct edge A -> C if there is another path A -> ... -> C.

        This keeps the subclass DAG cleaner.
        """
        edges = []

        for parent, children in self.parent_to_children.items():
            for child in children:
                edges.append((parent, child))

        for parent, child in edges:
            # Temporarily remove direct edge
            self.parent_to_children[parent].discard(child)
            self.child_to_parents[child].discard(parent)

            # If path still exists, direct edge is redundant
            if not self.is_ancestor(parent, child):
                self.parent_to_children[parent].add(child)
                self.child_to_parents[child].add(parent)

    def should_skip_pair(self, a, b):
        """
        Pruning before calling LLM.
        """
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return True, "same_equivalence_group"

        key = self._pair_key(ra, rb)
        if key in self.comparison_cache:
            return True, "already_compared"

        if self.is_ancestor(ra, rb) or self.is_ancestor(rb, ra):
            return True, "already_parent_child_path"

        return False, ""

    def apply_relation(self, class_a, class_b, relation, confidence, raw_output, reason=""):
        """
        Apply LLM relation judgment to SchemaState.
        """

        relation = str(relation).strip().upper()

        if relation not in VALID_RELATIONS:
            relation = "UNCERTAIN"

        ra_before = self.find(class_a)
        rb_before = self.find(class_b)
        pair_key = self._pair_key(class_a, class_b)

        record = {
            "class_a": class_a,
            "class_b": class_b,
            "canonical_a_before": ra_before,
            "canonical_b_before": rb_before,
            "relation": relation,
            "confidence": confidence,
            "reason": reason,
            "raw_output": raw_output,
        }

        self.comparison_cache[pair_key] = record

        if relation == "EQUIVALENT":
            self.add_equivalent(class_a, class_b, record)

        elif relation == "A_PARENT_B":
            self.add_subclass_edge(parent=class_a, child=class_b, record=record)

        elif relation == "B_PARENT_A":
            self.add_subclass_edge(parent=class_b, child=class_a, record=record)

        elif relation == "RELATED":
            self.add_related_edge(class_a, class_b, record)

        elif relation == "UNRELATED":
            self.add_unrelated(class_a, class_b, record)

        else:
            self.add_uncertain(class_a, class_b, record)

    def export(self):
        """
        Export schema state as serializable dict.
        """
        self.canonicalize_graphs()

        groups = self.dsu.groups()
        canonical_map = self.dsu.canonical_map()

        subclass_edges = []
        for parent, children in self.parent_to_children.items():
            for child in sorted(children):
                subclass_edges.append({
                    "parent": parent,
                    "child": child,
                })

        related_edges = [
            {
                "source": a,
                "target": b,
            }
            for a, b in sorted(self.related_edges)
        ]

        unrelated_pairs = [
            [a, b]
            for a, b in sorted(self.unrelated_pairs)
        ]

        uncertain_pairs = [
            [a, b]
            for a, b in sorted(self.uncertain_pairs)
        ]

        canonical_classes = sorted(set(canonical_map.values()))

        return {
            "class_statistics": {
                "num_raw_classes": len(self.raw_classes),
                "num_canonical_classes": len(canonical_classes),
                "num_equivalence_groups": sum(1 for _, members in groups.items() if len(members) > 1),
                "num_subclass_edges": len(subclass_edges),
                "num_related_edges": len(related_edges),
                "num_unrelated_pairs": len(unrelated_pairs),
                "num_uncertain_pairs": len(uncertain_pairs),
            },
            "raw_classes": self.raw_classes,
            "class_frequency": self.class_freq,
            "canonical_classes": canonical_classes,
            "canonical_map": canonical_map,
            "equivalence_groups": [
                members
                for _, members in sorted(groups.items())
                if len(members) > 1
            ],
            "subclass_edges": subclass_edges,
            "related_edges": related_edges,
            "unrelated_pairs": unrelated_pairs,
            "uncertain_pairs": uncertain_pairs,
            "conflict_records": self.conflict_records,
            "llm_relation_records": self.relation_records,
        }


class LLMClassRelationJudge:
    """
    Use LLM to judge relation between two class names.
    """

    def __init__(self, llm_model, max_retries=2):
        self.llm_model = llm_model
        self.max_retries = max_retries

    def build_prompt(self, class_a, class_b):
        system_prompt = (
            "You are an expert ontology schema relation judge. "
            "Given two entity class names, determine their semantic relation. "
            "Use only the meanings of the class names. "
            "Do not assume access to entity instances. "
            "Do not invent examples. "
            "Do not force a relation when the relation is weak or unclear. "
            "Return only valid JSON."
        )

        user_prompt = f"""
Class A:
{class_a}

Class B:
{class_b}

Task:
Judge the semantic relation between Class A and Class B.

Relation options:
1. EQUIVALENT
2. A_PARENT_B
3. B_PARENT_A
4. RELATED
5. UNRELATED
6. UNCERTAIN

Definitions:
- EQUIVALENT:
  Class A and Class B express the same semantic category or nearly the same category.
  Conceptually, they would cover the same or almost the same set of entities.
  Example: Place and Location.

- A_PARENT_B:
  Class A is broader and more general than Class B.
  Class B is a subtype or subclass of Class A.
  Conceptually, entities in B should also belong to A.
  Example: A=Person, B=Politician.

- B_PARENT_A:
  Class B is broader and more general than Class A.
  Class A is a subtype or subclass of Class B.
  Conceptually, entities in A should also belong to B.
  Example: A=City, B=Place.

- RELATED:
  Class A and Class B are semantically connected or partially overlapping,
  but they are not equivalent and neither is a clear parent of the other.
  Conceptually, they may share some entities or frequently appear in related contexts.
  Example: Politician and Government.

- UNRELATED:
  Class A and Class B have no stable semantic equivalence, containment, or clear relatedness.
  Conceptually, their entity sets are mutually exclusive or semantically far apart.
  Example: Fruit and War.

- UNCERTAIN:
  The relation cannot be reliably determined from the class names alone.

Important rules:
1. Prefer UNRELATED or UNCERTAIN if the relation is weak.
2. Do not over-connect classes.
3. Parent-child relation must be a clear semantic general-specific relation.
4. Related relation must be meaningful, not just vaguely associated.
5. Output exactly one relation.

Return JSON only in this format:
{{
  "relation": "A_PARENT_B",
  "confidence": 0.95,
  "reason": "Class A is a broader category and Class B is a more specific subtype."
}}
""".strip()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _call_llm_raw(self, messages):
        """
        Prefer infer_raw so JSON punctuation is preserved.
        """
        if hasattr(self.llm_model, "infer_raw"):
            return self.llm_model.infer_raw(messages)

        logger.warning(
            "LLM_Model has no infer_raw(). Falling back to infer(), "
            "but JSON parsing may fail if infer() normalizes punctuation."
        )
        return self.llm_model.infer(messages)

    def parse_relation_output(self, raw_output):
        """
        Parse LLM relation output.
        """

        if raw_output is None:
            return {
                "relation": "UNCERTAIN",
                "confidence": 0.0,
                "reason": "",
                "raw_output": "",
            }

        text = str(raw_output).strip()

        text = re.sub(r"^\s*Answer\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()

        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^```\s*", "", text).strip()
        text = re.sub(r"\s*```$", "", text).strip()

        # Try direct JSON parsing
        try:
            parsed = json.loads(text)
            relation = str(parsed.get("relation", "UNCERTAIN")).strip().upper()
            confidence = float(parsed.get("confidence", 0.0))
            reason = str(parsed.get("reason", "")).strip()

            if relation not in VALID_RELATIONS:
                relation = "UNCERTAIN"

            return {
                "relation": relation,
                "confidence": confidence,
                "reason": reason,
                "raw_output": raw_output,
            }

        except Exception:
            pass

        # Try extracting JSON object
        object_match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if object_match:
            try:
                parsed = json.loads(object_match.group(0))
                relation = str(parsed.get("relation", "UNCERTAIN")).strip().upper()
                confidence = float(parsed.get("confidence", 0.0))
                reason = str(parsed.get("reason", "")).strip()

                if relation not in VALID_RELATIONS:
                    relation = "UNCERTAIN"

                return {
                    "relation": relation,
                    "confidence": confidence,
                    "reason": reason,
                    "raw_output": raw_output,
                }

            except Exception:
                pass

        # Fallback text matching
        upper_text = text.upper()

        for relation in sorted(VALID_RELATIONS, key=len, reverse=True):
            if relation in upper_text:
                return {
                    "relation": relation,
                    "confidence": 0.5,
                    "reason": "Parsed by fallback text matching.",
                    "raw_output": raw_output,
                }

        return {
            "relation": "UNCERTAIN",
            "confidence": 0.0,
            "reason": "Failed to parse relation output.",
            "raw_output": raw_output,
        }

    def judge(self, class_a, class_b):
        """
        Judge relation between two class names.
        """
        messages = self.build_prompt(class_a, class_b)
        raw_output = self._call_llm_raw(messages)
        parsed = self.parse_relation_output(raw_output)

        if parsed["relation"] != "UNCERTAIN":
            return parsed

        for _ in range(self.max_retries):
            raw_output = self._call_llm_raw(messages)
            parsed = self.parse_relation_output(raw_output)

            if parsed["relation"] != "UNCERTAIN":
                return parsed

        return parsed


def load_entity_classes(entity_classes_path):
    """
    Load entity_classes.json and count class frequencies.
    """

    with open(entity_classes_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    class_counter = Counter()

    for _, item in data.items():
        classes = item.get("classes", [])

        if not isinstance(classes, list):
            continue

        for cls in classes:
            if cls is None:
                continue

            cls = str(cls).strip()
            cls = re.sub(r"\s+", " ", cls)
            cls = cls.strip(" .。,:;；，[]{}()\"'")

            if not cls:
                continue

            class_counter[cls] += 1

    return class_counter


def save_json(obj, path):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def build_ordered_classes(class_counter, min_freq=1, max_classes=0):
    """
    Sort classes by frequency descending.

    max_classes=0 means use all classes.
    """

    items = [
        (cls, freq)
        for cls, freq in class_counter.items()
        if freq >= min_freq
    ]

    items.sort(key=lambda x: (-x[1], x[0].lower()))

    if max_classes and max_classes > 0:
        items = items[:max_classes]

    classes = [cls for cls, _ in items]
    filtered_counter = Counter({cls: freq for cls, freq in items})

    return classes, filtered_counter


def build_schema(
    entity_classes_path,
    output_path,
    llm_model,
    min_freq=1,
    max_classes=0,
    max_pairs=0,
    confidence_threshold=0.0,
    checkpoint_every=20,
):
    """
    Main schema induction process.
    """

    logger.info(f"Loading entity classes from: {entity_classes_path}")
    class_counter = load_entity_classes(entity_classes_path)

    logger.info(f"Total unique raw classes before filtering: {len(class_counter)}")

    classes, filtered_counter = build_ordered_classes(
        class_counter,
        min_freq=min_freq,
        max_classes=max_classes,
    )

    logger.info(f"Total classes after filtering: {len(classes)}")

    total_pairs = len(classes) * (len(classes) - 1) // 2
    logger.info(f"Total possible pairwise comparisons: {total_pairs}")

    if max_pairs and max_pairs > 0:
        logger.info(f"Max pairs limit enabled: {max_pairs}")

    state = SchemaState(classes=classes, class_freq=filtered_counter)
    judge = LLMClassRelationJudge(llm_model=llm_model)

    processed_pairs = 0
    llm_calls = 0
    skipped_pairs = 0

    pair_iter = combinations(classes, 2)

    for class_a, class_b in tqdm(pair_iter, total=total_pairs, desc="Building SchemaState"):
        if max_pairs and processed_pairs >= max_pairs:
            logger.info(f"Reached max_pairs={max_pairs}. Stop early.")
            break

        processed_pairs += 1

        should_skip, skip_reason = state.should_skip_pair(class_a, class_b)
        if should_skip:
            skipped_pairs += 1
            continue

        result = judge.judge(class_a, class_b)

        relation = result["relation"]
        confidence = result.get("confidence", 0.0)

        if confidence < confidence_threshold:
            relation = "UNCERTAIN"

        state.apply_relation(
            class_a=class_a,
            class_b=class_b,
            relation=relation,
            confidence=confidence,
            raw_output=result.get("raw_output", ""),
            reason=result.get("reason", ""),
        )

        llm_calls += 1

        if checkpoint_every and llm_calls % checkpoint_every == 0:
            exported = state.export()
            exported["runtime_statistics"] = {
                "processed_pairs": processed_pairs,
                "llm_calls": llm_calls,
                "skipped_pairs": skipped_pairs,
            }
            save_json(exported, output_path)
            logger.info(f"Checkpoint saved to: {output_path}")

    exported = state.export()
    exported["runtime_statistics"] = {
        "processed_pairs": processed_pairs,
        "llm_calls": llm_calls,
        "skipped_pairs": skipped_pairs,
    }

    save_json(exported, output_path)

    logger.info(f"Schema construction finished.")
    logger.info(f"Processed pairs: {processed_pairs}")
    logger.info(f"LLM calls: {llm_calls}")
    logger.info(f"Skipped pairs: {skipped_pairs}")
    logger.info(f"Schema saved to: {output_path}")

    return exported


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--entity_classes_path",
        type=str,
        default="./import/2wikimultihop/entity_classes.json",
        help="Path to entity_classes.json."
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./import/2wikimultihop/schema_state.json",
        help="Where to save schema state."
    )

    parser.add_argument(
        "--llm_model",
        type=str,
        default="/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
        help="LLM model name or path used by your OpenAI-compatible endpoint."
    )

    parser.add_argument(
        "--min_freq",
        type=int,
        default=1,
        help="Only keep classes with frequency >= min_freq."
    )

    parser.add_argument(
        "--max_classes",
        type=int,
        default=100,
        help="Use top-N frequent classes. Set 0 to use all classes."
    )

    parser.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="Maximum number of pairwise comparisons. Set 0 for no limit."
    )

    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.0,
        help="Relations below this confidence are treated as UNCERTAIN."
    )

    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=20,
        help="Save checkpoint after every N LLM calls."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    log_path = os.path.join(os.path.dirname(args.output_path), "schema_state_builder.log")
    setup_logging(log_path)

    logger.info("Starting SchemaState construction.")
    logger.info(f"Entity class path: {args.entity_classes_path}")
    logger.info(f"Output path: {args.output_path}")

    llm_model = LLM_Model(args.llm_model)

    build_schema(
        entity_classes_path=args.entity_classes_path,
        output_path=args.output_path,
        llm_model=llm_model,
        min_freq=args.min_freq,
        max_classes=args.max_classes,
        max_pairs=args.max_pairs,
        confidence_threshold=args.confidence_threshold,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()