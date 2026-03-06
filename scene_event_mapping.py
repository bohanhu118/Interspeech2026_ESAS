import json
import pandas as pd


# Load ontology
with open("model/beats/ontology.json", "r") as f:
    ontology = json.load(f)

# lookup by id and by name
id_to_item = {item["id"]: item for item in ontology}
name_to_id = {item["name"]: item["id"] for item in ontology}

# Load class label indices
csv_path = "model/beats/class_labels_indices.csv"
df = pd.read_csv(csv_path)
mid_to_index = dict(zip(df["mid"], df["index"]))


def get_disent_event_indices(parent_names):
    """
    Return all unique indices for descendants (children, grandchildren, ...)
    of given parent names. Optionally include the parents themselves.
    """
    result_mids = set()
    stack = [name_to_id[name] for name in parent_names if name in name_to_id]

    visited = set()
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)

        node = id_to_item.get(node_id)
        if not node:
            continue

        if node_id not in [name_to_id[name] for name in parent_names]:
            # add every descendant (not just leaves)
            result_mids.add(node_id)

        child_ids = node.get("child_ids", [])
        stack.extend(child_ids)

    for n in parent_names:
        if n in ["Animal", "Music"]:
            result_mids.add(name_to_id[n])

    # Convert mids to indices if present in class_label_indices
    result_indices = [mid_to_index[mid] for mid in result_mids if mid in mid_to_index]
    return sorted(result_indices)