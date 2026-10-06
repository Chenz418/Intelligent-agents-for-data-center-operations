"""Strict provider tool-argument JSON parsing."""

import json


def strict_json_loads(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"non-JSON constant: {value}")

    return json.loads(text, object_pairs_hook=unique, parse_constant=reject)
