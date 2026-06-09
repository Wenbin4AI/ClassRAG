#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from collections import defaultdict, deque


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def normalize_class_name(x):
    return str(x).strip()


def clean_path(path):
    """
    修复单条 path 内部的自环/重复。
    例如:
    Entity -> Person -> Scientist -> Person
    会截断为:
    Entity -> Person -> Scientist
    """
    new_path = []
    seen = set()

    for c in path:
        c = normalize_class_name(c)
        if not c:
            continue

        key = c.lower()
        if key in seen:
            break

        seen.add(key)
        new_path.append(c)

    return new_path


def build_candidate_edges(entity_results):
    edges = []
    path_records = []

    for entity_id, item in entity_results.items():
        entity_text = item.get("entity", "")

        for raw_path in item.get("class_paths", []):
            if not isinstance(raw_path, list):
                continue

            path = clean_path(raw_path)

            if len(path) < 2:
                continue

            path_records.append({
                "entity_id": entity_id,
                "entity": entity_text,
                "raw_path": raw_path,
                "cleaned_path": path,
            })

            for parent, child in zip(path[:-1], path[1:]):
                if parent == child:
                    continue

                edges.append({
                    "parent": parent,
                    "child": child,
                    "entity_id": entity_id,
                    "entity": entity_text,
                    "path": path,
                })

    return edges, path_records


def has_path(graph, start, target):
    """
    判断 start 是否已经可以到达 target。
    如果 child 已经可以到达 parent，那么再加入 parent -> child 就会形成环。
    """
    if start == target:
        return True

    stack = [start]
    visited = set()

    while stack:
        node = stack.pop()

        if node == target:
            return True

        if node in visited:
            continue

        visited.add(node)

        for nxt in graph.get(node, set()):
            if nxt not in visited:
                stack.append(nxt)

    return False


def edge_priority(edge, edge_support):
    """
    边的保留优先级。
    支持次数越多，越应该保留。
    越靠近 Entity 的泛化边通常更稳定，也可以保留。
    """
    parent = edge["parent"]
    child = edge["child"]
    support = edge_support[(parent, child)]

    path = edge.get("path", [])
    try:
        depth = path.index(parent)
    except Exception:
        depth = 999

    return (-support, depth, parent, child)


def build_dag_schema(entity_results):
    candidate_edges, path_records = build_candidate_edges(entity_results)

    edge_support = defaultdict(int)
    edge_examples = defaultdict(list)

    for e in candidate_edges:
        key = (e["parent"], e["child"])
        edge_support[key] += 1

        if len(edge_examples[key]) < 5:
            edge_examples[key].append({
                "entity_id": e["entity_id"],
                "entity": e["entity"],
                "path": e["path"],
            })

    unique_edges = {}
    for e in candidate_edges:
        key = (e["parent"], e["child"])
        if key not in unique_edges:
            unique_edges[key] = e

    sorted_edges = sorted(
        unique_edges.values(),
        key=lambda e: edge_priority(e, edge_support)
    )

    parent_to_children = defaultdict(set)
    child_to_parents = defaultdict(set)

    kept_edges = []
    removed_edges = []

    for e in sorted_edges:
        parent = e["parent"]
        child = e["child"]

        if parent == child:
            removed_edges.append({
                "parent": parent,
                "child": child,
                "reason": "self_loop",
                "support": edge_support[(parent, child)],
                "examples": edge_examples[(parent, child)],
            })
            continue

        if has_path(parent_to_children, child, parent):
            removed_edges.append({
                "parent": parent,
                "child": child,
                "reason": "cycle",
                "support": edge_support[(parent, child)],
                "examples": edge_examples[(parent, child)],
            })
            continue

        parent_to_children[parent].add(child)
        child_to_parents[child].add(parent)

        kept_edges.append({
            "parent": parent,
            "child": child,
            "support": edge_support[(parent, child)],
            "examples": edge_examples[(parent, child)],
        })

    class_to_entities = defaultdict(set)
    leaf_class_to_entities = defaultdict(set)

    for entity_id, item in entity_results.items():
        for raw_path in item.get("class_paths", []):
            path = clean_path(raw_path)
            if len(path) < 2:
                continue

            for c in path:
                class_to_entities[c].add(entity_id)

            leaf_class_to_entities[path[-1]].add(entity_id)

    schema = {
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
            "num_candidate_edges": len(unique_edges),
            "num_kept_edges": len(kept_edges),
            "num_removed_edges": len(removed_edges),
            "num_removed_cycle_edges": sum(
                1 for x in removed_edges if x["reason"] == "cycle"
            ),
            "num_removed_self_loop_edges": sum(
                1 for x in removed_edges if x["reason"] == "self_loop"
            ),
        }
    }

    report = {
        "metadata": schema["metadata"],
        "removed_edges": removed_edges,
        "kept_edges": kept_edges,
    }

    return schema, report


def parse_args():
    parser = argparse.ArgumentParser()

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
        "--input_path",
        type=str,
        default="",
    )

    parser.add_argument(
        "--output_schema_path",
        type=str,
        default="",
    )

    parser.add_argument(
        "--cycle_report_path",
        type=str,
        default="",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset_dir = os.path.join(args.working_dir, args.dataset_name)

    input_path = args.input_path or os.path.join(
        dataset_dir,
        "entity_class_paths.json"
    )

    output_schema_path = args.output_schema_path or os.path.join(
        dataset_dir,
        "class_hierarchy_schema_dag.json"
    )

    cycle_report_path = args.cycle_report_path or os.path.join(
        dataset_dir,
        "class_hierarchy_cycle_report.json"
    )

    print(f"Loading entity class paths from: {input_path}")
    entity_results = load_json(input_path)

    schema, report = build_dag_schema(entity_results)

    save_json(schema, output_schema_path)
    save_json(report, cycle_report_path)

    print("\n========== Cycle Check Finished ==========")
    print(f"Entities: {schema['metadata']['num_entities']}")
    print(f"Classes: {schema['metadata']['num_classes']}")
    print(f"Candidate edges: {schema['metadata']['num_candidate_edges']}")
    print(f"Kept edges: {schema['metadata']['num_kept_edges']}")
    print(f"Removed edges: {schema['metadata']['num_removed_edges']}")
    print(f"Removed cycle edges: {schema['metadata']['num_removed_cycle_edges']}")
    print(f"Removed self-loop edges: {schema['metadata']['num_removed_self_loop_edges']}")
    print(f"Saved DAG schema to: {output_schema_path}")
    print(f"Saved cycle report to: {cycle_report_path}")
    print("==========================================\n")


if __name__ == "__main__":
    main()

# python scripts/check_and_fix_class_hierarchy_cycles.py \
#     --dataset_name 2wikimultihop \
#     --working_dir import