#!/usr/bin/env python3
"""Add the training metadata the three baselines read to a batched-replay LeRobot v3 dataset (in place, two files).

    add_train_meta.py <dataset root> [--prompts prompts.json] [--write-draft-prompts out.json] [--check]

1. isg_meta/goal_groups.json  (isg-goal-groups/v1) -- episode -> goal group id, from isg_meta/episodes.jsonl
   `instance.goal_group_id` (the eval_production collector's group: same goal within admission's
   goal_identity.equivalent_goal tolerance, one id across train / standard-eval / sibling splits). openpi's
   --shuffle-goal-image draws a training sample's goal image among the training episodes of its group.

       {"format": "isg-goal-groups/v1", "key": "instance.goal_group_id", "dataset": {...},
        "episodes": {"0": "alignment-axial_01__0d4016ae1378b2d4", ...},
        "groups": {"alignment-axial_01__0d4016ae1378b2d4": {"task": "alignment-axial_01", "episodes": [0, 1]}, ...}}

2. meta/tasks.jsonl -- one line per task (alias) of meta/tasks.parquet: {"task_index", "task_name", "task"}, `task`
   being the alias's natural-language prompt. This is the BEHAVIOR challenge demos' own file for task
   descriptions; openpi (--data.prompt-source task_description, which --text-only selects) and ACT
   (--prompt-source task_description) already read it and check it against meta/tasks.parquet, so a policy
   needs no new loader code and the checkpoint records the exact prompts it trained with.

Prompts come from --prompts (a JSON {alias: prompt}, every alias of the dataset required) or, without it, from the
draft templates below (one sentence per alias from its family and variant: target, partner, support). The drafts
are a starting point for the owner to edit: --write-draft-prompts writes them to a file without touching the dataset.
Neither file changes LeRobot's own metadata or any data/video file.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

GOAL_GROUPS_FORMAT = "isg-goal-groups/v1"

# variant: <family>-<subtask kind>-<target>[-<partner>]-<support>; BEHAVIOR model ids are dropped: `_abcdef_0`
# always, a bare trailing `_abcdef` only when it is not an English word (bar_soap_utgixp, not colored_pencil)
_MODEL_ID_INDEXED = re.compile(r"_[a-z]{6}_\d+$")
_MODEL_ID = re.compile(r"_([a-z]{6})$")
try:
    _WORDS = set(Path("/usr/share/dict/words").read_text().lower().split())
except OSError:
    _WORDS = None  # without a dictionary, bare six-letter suffixes are kept


def _strip_id(token: str) -> str:
    token = _MODEL_ID_INDEXED.sub("", token)
    m = _MODEL_ID.search(token)
    if m and _WORDS is not None and m.group(1) not in _WORDS:
        token = token[: m.start()]
    return token


def _name(token: str) -> str:
    return _strip_id(token).replace("_", " ").strip()


def _support(token: str) -> str:
    token = _strip_id(token)
    return {"walls": "wall", "floors": "floor", "floor_pitched": "floor", "bed_iorsl": "bed",
            "bookcase_inside": "bookcase", "breakfast_table_board": "breakfast table"}.get(token, token.replace("_", " "))


def draft_prompt(alias: str, variant: str) -> str:
    """One instruction per alias. Text alone cannot state the instance's exact goal (angle, pose, distance); these
    name the task, the objects and the relation, as the text-only baseline's conditioning."""
    parts = variant.split("-")
    family = parts[0]
    kind = parts[1] if len(parts) > 1 else ""
    if family == "camera_relocalization":
        return "Drive and adjust your torso until your head camera sees the goal view."
    if family == "object_scaling":
        return f"Move the {_name(parts[1])} in your hand toward or away from your head camera to the goal distance."
    if family == "mental_rotation":
        return f"Rotate the {_name(parts[1])} in your hand to the goal orientation."
    if family == "alignment":
        target, support = _name(parts[2]), _support(parts[3])
        if kind == "world_vertical":
            return f"Push the {target} hanging on the {support} until it is upright."
        return f"Push the {target} on the {support} until it is aligned with the {support} edges, facing you."
    if family == "disembedding":
        target, support = _name(parts[1]), _support(parts[-1])
        return f"Find the {target} among the objects on the source tray and place it on the empty tray on the {support}."
    if family == "configuration_matching":
        if kind == "block_design":
            return f"Arrange the four blocks on the {_support(parts[-1])} into the goal 2 x 2 pattern."
        if kind == "attachment":
            return f"Attach the {_name(parts[2])} to the {_name(parts[3])} on the {_support(parts[4])}."
        target = _name(parts[2])
        where = f" on the {_support(parts[3])}" if len(parts) > 3 else ""
        if kind.startswith("articulation_open"):
            return f"Open the {target}{where} to the goal angle."
        if kind.startswith("articulation_close"):
            return f"Close the {target}{where}."
    if family == "inter_object_relation_satisfaction":
        relation = kind.split("_")[0]
        target, partner = _name(parts[2]), _name(parts[3])
        support = _support(parts[4]) if len(parts) > 4 else ""
        if relation == "inside":
            return f"Put the {target} inside the {partner}."
        same = target == partner  # two objects of one category: "one ... the other"
        if relation == "nextto":
            return (f"Move one {target} next to the other {partner}." if same
                    else f"Move the {target} next to the {partner}.")
        if relation == "ontop":
            return f"Put the {target} on top of the {partner}."
        if relation == "touching":
            if same:
                return f"Push one {target} against the other so that they touch, without moving the other {partner}."
            return f"Push the {target} against the {partner} so that they touch, without moving the {partner}."
        if relation == "under":
            if kind.endswith("large_scale"):
                return f"Put the {target} on the floor under the {partner}."
            # small-scale under: the target is set down on the partner (argument order: partner under target)
            return f"Set the {target} down on the {partner}, so that the {partner} is under the {target}."
    raise ValueError(f"no prompt template for alias {alias!r} (variant {variant!r})")


def load_episodes(root: Path) -> list[dict]:
    path = root / "isg_meta/episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if [r["episode_index"] for r in rows] != list(range(len(rows))):
        raise ValueError(f"{path}: episode_index must be 0..N-1 in order")
    return rows


def tasks_by_index(root: Path) -> dict[int, str]:
    table = pq.read_table(root / "meta/tasks.parquet").to_pandas()
    return {int(i): str(t) for t, i in zip(table.index, table["task_index"])}


def goal_groups(root: Path, episodes: list[dict]) -> dict:
    by_ep, groups = {}, collections.OrderedDict()
    for e in episodes:
        gid = (e.get("instance") or {}).get("goal_group_id")
        if not gid:
            raise ValueError(f"episode {e['episode_index']} has no instance.goal_group_id")
        by_ep[str(e["episode_index"])] = gid
        g = groups.setdefault(gid, {"task": e["task"], "episodes": []})
        if g["task"] != e["task"]:
            raise ValueError(f"goal group {gid} spans tasks {g['task']} and {e['task']}")
        g["episodes"].append(e["episode_index"])
    sizes = collections.Counter(len(g["episodes"]) for g in groups.values())
    info = json.loads((root / "meta/info.json").read_text())
    return {"format": GOAL_GROUPS_FORMAT,
            "description": "episode -> goal group id (isg_meta/episodes.jsonl instance.goal_group_id: the eval_production "
                           "collector's grouping, episodes whose goals are equivalent within admission's goal_identity "
                           "tolerance; the same id across the train / standard-eval / sibling splits). Episodes of one "
                           "group share their goal, so each one's goal image shows the others' goal (openpi "
                           "--shuffle-goal-image).",
            "key": "instance.goal_group_id",
            "dataset": {"total_episodes": info["total_episodes"], "total_frames": info["total_frames"],
                        "groups": len(groups), "episodes_per_group": {str(k): v for k, v in sorted(sizes.items())}},
            "episodes": by_ep, "groups": groups}


def prompts_for(root: Path, episodes: list[dict], explicit: dict | None) -> dict[str, str]:
    variants = {}
    for e in episodes:
        variants.setdefault(e["task"], (e.get("instance") or {}).get("variant") or e["task"])
    if explicit is not None:
        missing = sorted(set(variants) - set(explicit))
        if missing:
            raise ValueError(f"--prompts lacks {len(missing)} aliases of this dataset: {missing[:5]}")
        return {alias: explicit[alias] for alias in variants}
    return {alias: draft_prompt(alias, variant) for alias, variant in variants.items()}


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--prompts", type=Path, help="JSON {alias: prompt}; default: the draft templates")
    ap.add_argument("--write-draft-prompts", type=Path, help="write the draft prompts here and exit")
    ap.add_argument("--check", action="store_true", help="only print what would be written")
    a = ap.parse_args(argv)
    root = a.root
    episodes = load_episodes(root)
    names = tasks_by_index(root)
    explicit = json.loads(a.prompts.read_text()) if a.prompts else None
    prompts = prompts_for(root, episodes, explicit)
    if a.write_draft_prompts:
        a.write_draft_prompts.write_text(json.dumps(prompts, indent=1) + "\n")
        print(f"wrote {len(prompts)} draft prompts to {a.write_draft_prompts}")
        return 0
    if unknown := sorted(set(prompts) - set(names.values())):
        raise ValueError(f"episodes.jsonl tasks {unknown[:5]} are not in meta/tasks.parquet")
    groups = goal_groups(root, episodes)
    lines = [json.dumps({"task_index": i, "task_name": name, "task": prompts[name]})
             for i, name in sorted(names.items()) if name in prompts]
    print(f"{root}: {len(episodes)} episodes, {groups['dataset']['groups']} goal groups "
          f"(episodes per group {groups['dataset']['episodes_per_group']}), {len(lines)} task prompts, e.g.")
    for line in lines[:3]:
        print("  ", line)
    if a.check:
        return 0
    write_atomic(root / "isg_meta/goal_groups.json", json.dumps(groups, indent=1) + "\n")
    write_atomic(root / "meta/tasks.jsonl", "\n".join(lines) + "\n")
    print(f"wrote {root / 'isg_meta/goal_groups.json'} and {root / 'meta/tasks.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
