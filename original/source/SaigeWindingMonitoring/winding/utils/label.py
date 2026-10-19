import json
from bisect import bisect_right
from typing import TypeAlias


Sections: TypeAlias = dict[int, dict | None]
Label: TypeAlias = dict | None


def _build_sections(labels: list[dict]):
    sections: Sections = {}
    prev_end_idx = -1

    for label in sorted(labels, key=lambda x: int(x["start_idx"])):
        if "start_idx" not in label or "end_idx" not in label:
            raise ValueError(f"Label must have start_idx and end_idx: {label}")

        start_idx = int(label["start_idx"])
        end_idx = int(label["end_idx"])

        if start_idx >= end_idx:
            raise ValueError(f"Start index must be less than end index: {start_idx} < {end_idx}")

        if prev_end_idx >= start_idx:
            raise ValueError(
                f"Start index must be greater than previous end index: {prev_end_idx} >= {start_idx}"
            )

        if prev_end_idx + 1 < start_idx:
            sections[prev_end_idx + 1] = None

        sections[start_idx] = {k: v for k, v in label.items() if k not in ["start_idx", "end_idx"]}
        prev_end_idx = end_idx

    sections[prev_end_idx + 1] = None
    keys = list(sections.keys())

    return keys, sections


class LabelStore:
    def __init__(self, label_path: str):
        self._label_path = label_path

        with open(self._label_path, "r") as f:
            data = json.load(f)

        self._direction_keys, self._direction_sections = _build_sections(data.get("direction", []))
        self._protrusion_keys, self._protrusion_sections = _build_sections(data.get("protrusion", []))
        self._state_keys, self._state_sections = _build_sections(data.get("state", []))

    @staticmethod
    def _get_label(index: int, keys: list[int], sections: Sections) -> Label:
        if index < 0 or index >= keys[-1]:
            return None

        idx = bisect_right(keys, index) - 1
        key = keys[idx]
        label = sections[key]
        return label

    @staticmethod
    def _get_section(index: int, keys: list[int], sections: Sections) -> tuple[int, int, Label]:
        if index < 0:
            return -1, -1, None
        elif index >= keys[-1]:
            return keys[-1], keys[-1], None

        idx = bisect_right(keys, index) - 1
        start_key = keys[idx]
        end_key = keys[idx + 1] - 1
        label = sections[start_key]

        return start_key, end_key, label

    @staticmethod
    def _prev_section(index: int, keys: list[int], sections: Sections) -> tuple[int, int, Label]:
        if index <= 0:
            return -1, -1, None

        idx = bisect_right(keys, index - 1) - 1
        start_key = keys[idx]
        end_key = keys[idx + 1] - 1 if idx + 1 < len(keys) else keys[-1]
        label = sections[start_key]

        if label is not None:
            return start_key, end_key, label

        if idx == 0:
            return -1, -1, None

        prev_start_key = keys[idx - 1]
        prev_end_key = keys[idx] - 1
        prev_label = sections[prev_start_key]

        return prev_start_key, prev_end_key, prev_label

    @staticmethod
    def _next_section(index: int, keys: list[int], sections: Sections) -> tuple[int, int, Label]:
        if index >= keys[-1]:
            return keys[-1], keys[-1], None

        idx = bisect_right(keys, index + 1) - 1
        start_key = keys[idx]
        end_key = keys[idx + 1] - 1 if idx + 1 < len(keys) else keys[-1]
        label = sections[start_key]

        if label is not None:
            return start_key, end_key, label

        if idx + 1 == len(keys):
            return keys[-1], keys[-1], None

        next_start_key = keys[idx + 1]
        next_end_key = keys[idx + 2] - 1 if idx + 2 < len(keys) else keys[-1]
        next_label = sections[next_start_key]

        return next_start_key, next_end_key, next_label

    @property
    def direction(self) -> Sections:
        return self._direction_sections

    @property
    def protrusion(self) -> Sections:
        return self._protrusion_sections

    @property
    def state(self) -> Sections:
        return self._state_sections

    def get_label(self, index: int) -> dict[str, Label]:
        return {
            "direction": self._get_label(index, self._direction_keys, self._direction_sections),
            "protrusion": self._get_label(index, self._protrusion_keys, self._protrusion_sections),
            "state": self._get_label(index, self._state_keys, self._state_sections),
        }

    def get_section(self, index: int) -> dict[str, tuple[int, int, Label]]:
        return {
            "direction": self._get_section(index, self._direction_keys, self._direction_sections),
            "protrusion": self._get_section(index, self._protrusion_keys, self._protrusion_sections),
            "state": self._get_section(index, self._state_keys, self._state_sections),
        }

    def __getitem__(self, index: int) -> dict[str, Label]:
        return self.get_label(index)

    def prev_direction(self, index: int):
        return self._prev_section(index, self._direction_keys, self._direction_sections)

    def current_direction(self, index: int):
        return self._get_section(index, self._direction_keys, self._direction_sections)

    def next_direction(self, index: int):
        return self._next_section(index, self._direction_keys, self._direction_sections)

    def prev_protrusion(self, index: int):
        return self._prev_section(index, self._protrusion_keys, self._protrusion_sections)

    def current_protrusion(self, index: int):
        return self._get_section(index, self._protrusion_keys, self._protrusion_sections)

    def next_protrusion(self, index: int):
        return self._next_section(index, self._protrusion_keys, self._protrusion_sections)

    def prev_state(self, index: int):
        return self._prev_section(index, self._state_keys, self._state_sections)

    def current_state(self, index: int):
        return self._get_section(index, self._state_keys, self._state_sections)

    def next_state(self, index: int):
        return self._next_section(index, self._state_keys, self._state_sections)
