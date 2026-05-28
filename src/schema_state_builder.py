import os
import re
import sys
import json
import argparse
import logging
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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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


def invert_relation(relation):
    """
    Invert relation direction when cached pair is queried in reverse order.
    """
    if relation == "A_PARENT_B":
        return "B_PARENT_A"
    if relation == "B_PARENT_A":
        return "A_PARENT_B"
    return relation


class DisjointSetUnion:
    """
    Union-Find structure for equivalent class merging.
    """

    def __init__(self, items=None, class_freq=None):
        self.parent = {}
        self.rank = {}
        self.class_freq = class_freq or {}

        if items:
            for item in items:
                self.add(item)

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        self.add(x)

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def _choose_representative(self, a, b):
        """
        Choose canonical name for an equivalent group.

        Priority:
        1. Higher class frequency
        2. Shorter class name
        3. Lexicographically smaller class name
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

        for item in list(self.parent.keys()):
            group_map[self.find(item)].append(item)

        return {
            rep: sorted(members)
            for rep, members in group_map.items()
        }

    def canonical_map(self):
        return {
            item: self.find(item)
            for item in list(self.parent.keys())
        }


class LLMClassRelationJudge:
    """
    Use LLM to judge the semantic relation between two class names.
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
  Conceptually, they would contain the same or almost the same kind of entities.
  Example: Place and Location.

- A_PARENT_B:
  Class A is broader and more general than Class B.
  Class B is a subtype or subclass of Class A.
  Conceptually, entities of B should also belong to A.
  Example: A=Person, B=Politician.

- B_PARENT_A:
  Class B is broader and more general than Class A.
  Class A is a subtype or subclass of Class B.
  Conceptually, entities of A should also belong to B.
  Example: A=City, B=Place.

- RELATED:
  Class A and Class B are semantically connected or partially overlapping,
  but they are not equivalent and neither is a clear parent of the other.
  Conceptually, their covered entities may overlap or the categories are strongly associated.
  Example: Politician and Government.

- UNRELATED:
  Class A and Class B have no stable semantic equivalence, containment, or clear relatedness.
  Conceptually, their covered entities are mutually exclusive or semantically far apart.
  Example: Fruit and War.

- UNCERTAIN:
  The relation cannot be reliably determined from the class names alone.

Important rules:
1. Prefer UNRELATED or UNCERTAIN if the relation is weak.
2. Do not over-connect classes.
3. Parent-child relation must be a clear semantic general-specific relation.
4. Related relation must be meaningful, not just vaguely associated.
5. Output exactly one relation.
6. Return JSON only.

Output format:
{{
  "relation": "A_PARENT_B",
  "confidence": 0.95,
  "reason": "Class A is broader and Class B is a more specific subtype."
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
            "LLM_Model has no infer_raw(). Falling back to infer(). "
            "If infer() normalizes punctuation, JSON parsing may be degraded."
        )
        return self.llm_model.infer(messages)

    def parse_relation_output(self, raw_output):
        """
        Parse LLM output into relation / confidence / reason.
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

        # Try extracting first JSON object
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


class SchemaState:
    """
    Incremental schema state.

    It maintains:
    1. Equivalent relation by DSU
    2. Parent-child relation by DAG
    3. Related relation by undirected graph
    4. Comparison cache
    5. Insertion records
    """

    def __init__(
        self,
        class_freq,
        confidence_threshold=0.0,
        unrelated_skip_confidence=0.0,
    ):
        self.class_freq = dict(class_freq)

        self.dsu = DisjointSetUnion(items=list(class_freq.keys()), class_freq=self.class_freq)

        self.parent_to_children = defaultdict(set)
        self.child_to_parents = defaultdict(set)
        self.related_edges = set()

        self.inserted_classes = set()

        self.comparison_cache = {}
        self.llm_relation_records = []
        self.insertion_records = []

        self.unrelated_pairs = set()
        self.uncertain_pairs = set()
        self.conflict_records = []

        self.confidence_threshold = confidence_threshold
        self.unrelated_skip_confidence = unrelated_skip_confidence

        self.num_llm_calls = 0
        self.num_cache_hits = 0
        self.num_pruned_pairs = 0

    def find(self, cls):
        return self.dsu.find(cls)

    def _edge_key(self, a, b):
        a = self.find(a)
        b = self.find(b)
        return tuple(sorted([a, b]))

    def _pair_key(self, a, b):
        a = self.find(a)
        b = self.find(b)
        return tuple(sorted([a, b]))

    def _relation_for_orientation(self, record, query_a, query_b):
        """
        Return relation in the direction of query_a/query_b.
        """
        relation = record["relation"]

        stored_a = record["class_a"]
        stored_b = record["class_b"]

        if stored_a == query_a and stored_b == query_b:
            return relation

        if stored_a == query_b and stored_b == query_a:
            return invert_relation(relation)

        return relation

    def get_roots(self):
        """
        Current canonical roots among inserted classes.
        """
        self.canonicalize_graphs()

        inserted_reps = {self.find(x) for x in self.inserted_classes}

        roots = []
        for cls in sorted(inserted_reps, key=lambda x: (-self.class_freq.get(x, 0), x.lower())):
            if len(self.child_to_parents.get(cls, set())) == 0:
                roots.append(cls)

        return roots

    def is_ancestor(self, ancestor, descendant):
        """
        Whether ancestor -> ... -> descendant exists.
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
        Adding parent -> child creates a cycle if child is already ancestor of parent.
        """
        parent = self.find(parent)
        child = self.find(child)

        if parent == child:
            return True

        return self.is_ancestor(child, parent)

    def canonicalize_graphs(self):
        """
        Map all graph nodes to DSU representatives.
        Remove self-loops, duplicate edges, and related edges that are already parent-child.
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

                new_parent_to_children[p].add(c)
                new_child_to_parents[c].add(p)

        self.parent_to_children = new_parent_to_children
        self.child_to_parents = new_child_to_parents

        # Canonicalize related edges
        new_related_edges = set()

        for a, b in self.related_edges:
            ra = self.find(a)
            rb = self.find(b)

            if ra == rb:
                continue

            if self.is_ancestor(ra, rb) or self.is_ancestor(rb, ra):
                continue

            new_related_edges.add(tuple(sorted([ra, rb])))

        self.related_edges = new_related_edges

        # Canonicalize inserted classes
        self.inserted_classes = {self.find(x) for x in self.inserted_classes}

    def remove_subclass_edge(self, parent, child):
        parent = self.find(parent)
        child = self.find(child)

        self.parent_to_children[parent].discard(child)
        self.child_to_parents[child].discard(parent)

    def add_equivalent(self, a, b, record=None):
        a = self.find(a)
        b = self.find(b)

        if a == b:
            return self.find(a)

        rep = self.dsu.union(a, b)
        self.canonicalize_graphs()

        return rep

    def add_subclass_edge(self, parent, child, record=None):
        """
        Add parent -> child into DAG if it does not create cycle.
        """
        parent = self.find(parent)
        child = self.find(child)

        if parent == child:
            return False

        if self.would_create_cycle(parent, child):
            self.conflict_records.append({
                "type": "cycle_conflict",
                "parent": parent,
                "child": child,
                "record": record,
            })
            return False

        if self.is_ancestor(parent, child):
            return True

        self.parent_to_children[parent].add(child)
        self.child_to_parents[child].add(parent)

        edge_key = self._edge_key(parent, child)
        if edge_key in self.related_edges:
            self.related_edges.remove(edge_key)

        self.remove_transitive_redundant_edges()

        return True

    def add_related_edge(self, a, b, record=None):
        a = self.find(a)
        b = self.find(b)

        if a == b:
            return False

        if self.is_ancestor(a, b) or self.is_ancestor(b, a):
            return False

        self.related_edges.add(self._edge_key(a, b))
        return True

    def add_unrelated_pair(self, a, b):
        a = self.find(a)
        b = self.find(b)

        if a != b:
            self.unrelated_pairs.add(self._edge_key(a, b))

    def add_uncertain_pair(self, a, b):
        a = self.find(a)
        b = self.find(b)

        if a != b:
            self.uncertain_pairs.add(self._edge_key(a, b))

    def remove_transitive_redundant_edges(self):
        """
        Remove direct edge A -> C if another path A -> ... -> C exists.
        """
        edges = []

        for parent, children in self.parent_to_children.items():
            for child in children:
                edges.append((parent, child))

        for parent, child in edges:
            parent = self.find(parent)
            child = self.find(child)

            if child not in self.parent_to_children.get(parent, set()):
                continue

            self.parent_to_children[parent].discard(child)
            self.child_to_parents[child].discard(parent)

            if not self.is_ancestor(parent, child):
                self.parent_to_children[parent].add(child)
                self.child_to_parents[child].add(parent)

    def compare(self, class_a, class_b, judge):
        """
        Compare two canonical classes with pruning and cache.
        Relation is returned in the orientation class_a -> class_b.
        """
        class_a = self.find(class_a)
        class_b = self.find(class_b)

        if class_a == class_b:
            return {
                "relation": "EQUIVALENT",
                "confidence": 1.0,
                "reason": "Same equivalence group.",
                "raw_output": "",
                "from_cache": True,
            }

        # Existing parent-child path can be used directly
        if self.is_ancestor(class_a, class_b):
            self.num_pruned_pairs += 1
            return {
                "relation": "A_PARENT_B",
                "confidence": 1.0,
                "reason": "Existing ancestor path.",
                "raw_output": "",
                "from_cache": True,
            }

        if self.is_ancestor(class_b, class_a):
            self.num_pruned_pairs += 1
            return {
                "relation": "B_PARENT_A",
                "confidence": 1.0,
                "reason": "Existing ancestor path.",
                "raw_output": "",
                "from_cache": True,
            }

        pair_key = self._pair_key(class_a, class_b)

        if pair_key in self.comparison_cache:
            self.num_cache_hits += 1
            cached = self.comparison_cache[pair_key]
            relation = self._relation_for_orientation(cached, class_a, class_b)

            return {
                "relation": relation,
                "confidence": cached.get("confidence", 0.0),
                "reason": cached.get("reason", ""),
                "raw_output": cached.get("raw_output", ""),
                "from_cache": True,
            }

        result = judge.judge(class_a, class_b)
        self.num_llm_calls += 1

        relation = result.get("relation", "UNCERTAIN")
        confidence = float(result.get("confidence", 0.0))

        if confidence < self.confidence_threshold:
            relation = "UNCERTAIN"

        if relation not in VALID_RELATIONS:
            relation = "UNCERTAIN"

        record = {
            "class_a": class_a,
            "class_b": class_b,
            "relation": relation,
            "confidence": confidence,
            "reason": result.get("reason", ""),
            "raw_output": result.get("raw_output", ""),
        }

        self.comparison_cache[pair_key] = record
        self.llm_relation_records.append(record)

        return {
            **record,
            "from_cache": False,
        }

    def apply_relation(self, class_a, class_b, result):
        """
        Apply a relation result to the corresponding structure.
        """
        class_a = self.find(class_a)
        class_b = self.find(class_b)

        relation = result.get("relation", "UNCERTAIN")
        confidence = float(result.get("confidence", 0.0))

        if relation == "EQUIVALENT":
            self.add_equivalent(class_a, class_b, result)

        elif relation == "A_PARENT_B":
            self.add_subclass_edge(parent=class_a, child=class_b, record=result)

        elif relation == "B_PARENT_A":
            self.add_subclass_edge(parent=class_b, child=class_a, record=result)

        elif relation == "RELATED":
            self.add_related_edge(class_a, class_b, record=result)

        elif relation == "UNRELATED":
            self.add_unrelated_pair(class_a, class_b)

        else:
            self.add_uncertain_pair(class_a, class_b)

    def insert_class(self, new_class, judge):
        """
        Insert one new class into existing schema forest.

        It compares the new class with each existing root.
        If root is parent of new class, search its direct children recursively.
        If new class is parent of root, place new class above the root.
        If related, only add related edge to the root and stop searching that tree.
        If unrelated or uncertain, skip that tree.
        """
        original_new_class = new_class
        new_class = self.find(new_class)

        record = {
            "class": original_new_class,
            "canonical_class_before": new_class,
            "actions": [],
        }

        if len(self.inserted_classes) == 0:
            self.inserted_classes.add(new_class)
            record["actions"].append({
                "action": "insert_as_first_root",
                "class": new_class,
            })
            self.insertion_records.append(record)
            return

        roots = self.get_roots()
        structurally_inserted = False
        equivalent_merged = False

        for root in list(roots):
            new_class = self.find(new_class)
            root = self.find(root)

            if new_class == root:
                structurally_inserted = True
                equivalent_merged = True
                break

            result = self.compare(new_class, root, judge)
            relation = result["relation"]
            confidence = float(result.get("confidence", 0.0))

            if relation == "EQUIVALENT":
                rep = self.add_equivalent(new_class, root, result)
                self.inserted_classes.add(rep)
                structurally_inserted = True
                equivalent_merged = True

                record["actions"].append({
                    "action": "equivalent_to_root",
                    "new_class": original_new_class,
                    "root": root,
                    "representative": rep,
                    "result": result,
                })
                break

            elif relation == "A_PARENT_B":
                # new_class is parent of root
                added = self.add_subclass_edge(parent=new_class, child=root, record=result)
                if added:
                    structurally_inserted = True
                    record["actions"].append({
                        "action": "new_class_parent_of_root",
                        "new_class": new_class,
                        "root": root,
                        "result": result,
                    })

            elif relation == "B_PARENT_A":
                # root is parent of new_class; search under root
                inserted_under_root = self._insert_under_parent(
                    new_class=new_class,
                    parent=root,
                    judge=judge,
                    record=record,
                )
                if inserted_under_root:
                    structurally_inserted = True

            elif relation == "RELATED":
                self.add_related_edge(new_class, root, record=result)
                record["actions"].append({
                    "action": "related_to_root_skip_subtree",
                    "new_class": new_class,
                    "root": root,
                    "result": result,
                })

            elif relation == "UNRELATED":
                self.add_unrelated_pair(new_class, root)
                record["actions"].append({
                    "action": "unrelated_to_root_skip_subtree",
                    "new_class": new_class,
                    "root": root,
                    "result": result,
                })

            else:
                self.add_uncertain_pair(new_class, root)
                record["actions"].append({
                    "action": "uncertain_with_root_skip_subtree",
                    "new_class": new_class,
                    "root": root,
                    "result": result,
                })

        if not structurally_inserted and not equivalent_merged:
            # No parent/equivalent/root-parent relation found.
            # The class becomes an independent root.
            new_class = self.find(new_class)
            record["actions"].append({
                "action": "insert_as_independent_root",
                "class": new_class,
            })

        self.inserted_classes.add(self.find(new_class))
        self.canonicalize_graphs()
        self.insertion_records.append(record)

    def _insert_under_parent(self, new_class, parent, judge, record):
        """
        Insert new_class somewhere under parent.

        Assumption:
            parent is already judged as a parent of new_class.

        Strategy:
            Compare new_class with direct children of parent.
            - If equivalent to a child: union and stop.
            - If child is parent of new_class: recurse into child.
            - If new_class is parent of a child: reparent that child under new_class.
            - If related to a child: add related edge and skip that subtree.
            - If unrelated/uncertain: skip that child subtree.

            If no better child position is found, add parent -> new_class.
        """
        new_class = self.find(new_class)
        parent = self.find(parent)

        children = sorted(
            list(self.parent_to_children.get(parent, set())),
            key=lambda x: (-self.class_freq.get(x, 0), x.lower())
        )

        if not children:
            added = self.add_subclass_edge(parent=parent, child=new_class)
            if added:
                record["actions"].append({
                    "action": "insert_as_child_of_leaf_parent",
                    "parent": parent,
                    "child": new_class,
                })
            return added

        placed_deeper = False
        equivalent_merged = False
        children_to_reparent = []

        for child in children:
            new_class = self.find(new_class)
            child = self.find(child)

            if new_class == child:
                placed_deeper = True
                equivalent_merged = True
                break

            result = self.compare(new_class, child, judge)
            relation = result["relation"]

            if relation == "EQUIVALENT":
                rep = self.add_equivalent(new_class, child, result)
                self.inserted_classes.add(rep)
                placed_deeper = True
                equivalent_merged = True

                record["actions"].append({
                    "action": "equivalent_to_child",
                    "new_class": new_class,
                    "child": child,
                    "representative": rep,
                    "result": result,
                })
                break

            elif relation == "A_PARENT_B":
                # new_class is parent of child
                children_to_reparent.append(child)
                record["actions"].append({
                    "action": "new_class_parent_of_existing_child",
                    "new_class": new_class,
                    "child": child,
                    "result": result,
                })

            elif relation == "B_PARENT_A":
                # child is parent of new_class; go deeper
                record["actions"].append({
                    "action": "descend_into_child_subtree",
                    "new_class": new_class,
                    "child_parent": child,
                    "result": result,
                })

                child_inserted = self._insert_under_parent(
                    new_class=new_class,
                    parent=child,
                    judge=judge,
                    record=record,
                )
                if child_inserted:
                    placed_deeper = True

            elif relation == "RELATED":
                self.add_related_edge(new_class, child, record=result)
                record["actions"].append({
                    "action": "related_to_child_skip_subtree",
                    "new_class": new_class,
                    "child": child,
                    "result": result,
                })

            elif relation == "UNRELATED":
                self.add_unrelated_pair(new_class, child)
                record["actions"].append({
                    "action": "unrelated_to_child_skip_subtree",
                    "new_class": new_class,
                    "child": child,
                    "result": result,
                })

            else:
                self.add_uncertain_pair(new_class, child)
                record["actions"].append({
                    "action": "uncertain_with_child_skip_subtree",
                    "new_class": new_class,
                    "child": child,
                    "result": result,
                })

        if equivalent_merged:
            return True

        new_class = self.find(new_class)
        parent = self.find(parent)

        # If new_class is parent of one or more existing children, insert it between parent and those children.
        if children_to_reparent:
            if not placed_deeper:
                self.add_subclass_edge(parent=parent, child=new_class)

            for child in children_to_reparent:
                child = self.find(child)

                if child == new_class:
                    continue

                # Reparent: parent -> child becomes parent -> new_class -> child
                self.remove_subclass_edge(parent, child)
                self.add_subclass_edge(parent=new_class, child=child)

            record["actions"].append({
                "action": "insert_as_intermediate_node",
                "parent": parent,
                "new_class": new_class,
                "reparented_children": sorted([self.find(c) for c in children_to_reparent]),
            })

            return True

        # If new_class was placed under a deeper child, no direct parent -> new_class edge is needed.
        if placed_deeper:
            return True

        # Otherwise, parent is the best known parent, insert directly.
        added = self.add_subclass_edge(parent=parent, child=new_class)
        if added:
            record["actions"].append({
                "action": "insert_as_direct_child",
                "parent": parent,
                "child": new_class,
            })

        return added

    def export(self):
        """
        Export schema state as serializable dict.
        """
        self.canonicalize_graphs()

        canonical_map = self.dsu.canonical_map()
        groups = self.dsu.groups()

        canonical_classes = sorted({self.find(x) for x in self.inserted_classes})

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

        roots = self.get_roots()

        return {
            "class_statistics": {
                "num_raw_classes": len(self.class_freq),
                "num_inserted_classes": len(self.inserted_classes),
                "num_canonical_inserted_classes": len(canonical_classes),
                "num_roots": len(roots),
                "num_equivalence_groups": sum(1 for _, members in groups.items() if len(members) > 1),
                "num_subclass_edges": len(subclass_edges),
                "num_related_edges": len(related_edges),
                "num_unrelated_pairs": len(unrelated_pairs),
                "num_uncertain_pairs": len(uncertain_pairs),
            },
            "runtime_statistics": {
                "num_llm_calls": self.num_llm_calls,
                "num_cache_hits": self.num_cache_hits,
                "num_pruned_pairs": self.num_pruned_pairs,
            },
            "class_frequency": self.class_freq,
            "roots": roots,
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
            "llm_relation_records": self.llm_relation_records,
            "insertion_records": self.insertion_records,
        }


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


def save_json(obj, path):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def build_schema_incrementally(
    entity_classes_path,
    output_path,
    llm_model,
    min_freq=1,
    max_classes=0,
    max_insertions=0,
    confidence_threshold=0.0,
    unrelated_skip_confidence=0.0,
    checkpoint_every=10,
):
    logger.info(f"Loading entity classes from: {entity_classes_path}")

    class_counter = load_entity_classes(entity_classes_path)
    logger.info(f"Total unique raw classes before filtering: {len(class_counter)}")

    classes, filtered_counter = build_ordered_classes(
        class_counter,
        min_freq=min_freq,
        max_classes=max_classes,
    )

    logger.info(f"Total classes after filtering: {len(classes)}")

    if max_insertions and max_insertions > 0:
        classes = classes[:max_insertions]
        filtered_counter = Counter({cls: filtered_counter[cls] for cls in classes})
        logger.info(f"Max insertions enabled. Classes to insert: {len(classes)}")

    state = SchemaState(
        class_freq=filtered_counter,
        confidence_threshold=confidence_threshold,
        unrelated_skip_confidence=unrelated_skip_confidence,
    )

    judge = LLMClassRelationJudge(llm_model=llm_model)

    for idx, cls in enumerate(tqdm(classes, desc="Incrementally Building SchemaState"), start=1):
        logger.info(f"Inserting class {idx}/{len(classes)}: {cls}")
        state.insert_class(cls, judge)

        if checkpoint_every and idx % checkpoint_every == 0:
            exported = state.export()
            exported["progress"] = {
                "inserted": idx,
                "total": len(classes),
                "latest_class": cls,
            }
            save_json(exported, output_path)
            logger.info(f"Checkpoint saved to: {output_path}")

    exported = state.export()
    exported["progress"] = {
        "inserted": len(classes),
        "total": len(classes),
        "latest_class": classes[-1] if classes else "",
    }

    save_json(exported, output_path)

    logger.info("Schema construction finished.")
    logger.info(f"Schema saved to: {output_path}")
    logger.info(f"LLM calls: {state.num_llm_calls}")
    logger.info(f"Cache hits: {state.num_cache_hits}")
    logger.info(f"Pruned pairs: {state.num_pruned_pairs}")

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
        default="./import/2wikimultihop/schema_state_incremental.json",
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
        default=0,
        help="Use top-N frequent classes. Set 0 to use all classes."
    )

    parser.add_argument(
        "--max_insertions",
        type=int,
        default=0,
        help="Maximum number of classes to insert after filtering. Set 0 for no limit."
    )

    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Relations below this confidence are treated as UNCERTAIN."
    )

    parser.add_argument(
        "--unrelated_skip_confidence",
        type=float,
        default=0.0,
        help="Reserved for stricter unrelated subtree pruning."
    )

    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=10,
        help="Save checkpoint after every N inserted classes."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    log_path = os.path.join(os.path.dirname(args.output_path), "schema_state_builder.log")
    setup_logging(log_path)

    logger.info("Starting incremental SchemaState construction.")
    logger.info(f"Entity class path: {args.entity_classes_path}")
    logger.info(f"Output path: {args.output_path}")

    llm_model = LLM_Model(args.llm_model)

    build_schema_incrementally(
        entity_classes_path=args.entity_classes_path,
        output_path=args.output_path,
        llm_model=llm_model,
        min_freq=args.min_freq,
        max_classes=args.max_classes,
        max_insertions=args.max_insertions,
        confidence_threshold=args.confidence_threshold,
        unrelated_skip_confidence=args.unrelated_skip_confidence,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()