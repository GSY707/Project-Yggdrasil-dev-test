"""Dump memory tree from test database"""
import json
from yggdrasil.database import Database
from yggdrasil.memory_engine import MemoryEngine

db = Database("yggdrasil/yggdrasil.db")
engine = MemoryEngine(db)

def dump_tree(name, depth=0):
    try:
        ctx = engine.retrieve_context(name)
    except Exception as e:
        print("  " * depth + "[ERROR] " + name + ": " + str(e))
        return
    node = ctx["node"]
    content_preview = node["content"][:150].replace("\n", " ")
    prefix = "  " * depth
    print(prefix + node["name"] + " (k=" + str(node["k_value"]) + ", w=" + str(round(node["current_weight"], 3)) + ")")
    print(prefix + "  -> " + content_preview)
    for edge in ctx["outgoing_edges"]:
        print(prefix + "  [edge] --" + edge["label"] + "--> " + edge["target"])
    for child in ctx["children"]:
        dump_tree(child["name"], depth + 1)

dump_tree("root")
