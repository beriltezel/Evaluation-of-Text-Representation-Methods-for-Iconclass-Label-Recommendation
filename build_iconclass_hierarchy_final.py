import os
import sqlite3
from iconclass import init


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "iconclass_hierarchy.db")
USED_NOTATION_KEYS_PATH = os.path.join(BASE_DIR, "used_notation_keys.txt")

LANGS = ["en"]


def load_used_notation_keys(path=USED_NOTATION_KEYS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def keep_notation(notation, used_notation_keys):
    if notation.find("(+") > 0 and notation not in used_notation_keys:
        return False
    return True


def notation_of(node):
    return str(node)


def extract_variant_notations(ic):
    variant_notations = []

    for notation, obj in ic.source._D.items():
        if notation is None:
            continue

        key = obj.get("k")
        if not key:
            continue

        for suffix in key.get("s", []):
            variant_notations.append(f"{notation}(+{suffix})")

    return variant_notations


def build_db():
    ic = init()
    used_notation_keys = load_used_notation_keys(USED_NOTATION_KEYS_PATH)

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("DROP TABLE IF EXISTS iconclass;")

    cur.execute("""
        CREATE TABLE iconclass (
            notation TEXT NOT NULL,
            lang TEXT NOT NULL,
            label TEXT NOT NULL,
            parent TEXT,
            depth INTEGER NOT NULL,
            PRIMARY KEY (notation, lang)
        );
    """)

    cur.execute("CREATE INDEX idx_iconclass_parent ON iconclass(parent);")
    cur.execute("CREATE INDEX idx_iconclass_depth ON iconclass(depth);")
    cur.execute("CREATE INDEX idx_iconclass_notation ON iconclass(notation);")

    structure = {}
    inserted = 0
    inserted_variants = 0
    skipped_empty = 0
    skipped_filtered = 0

    base_notations = [x for x in ic.source._D.keys() if x is not None]
    variant_notations = extract_variant_notations(ic)

    all_notations = sorted(set(base_notations + variant_notations))

    for notation in all_notations:
        if not keep_notation(notation, used_notation_keys):
            skipped_filtered += 1
            continue

        try:
            node = ic[notation]
        except Exception:
            continue

        path_nodes = list(node.path())
        depth = len(path_nodes)
        
        for i, path_node in enumerate(path_nodes):
            path_notation = notation_of(path_node)

            if path_notation not in structure:
                path_parent = notation_of(path_nodes[i - 1]) if i > 0 else None
                structure[path_notation] = (path_parent, i + 1)

        parent = None
        if len(path_nodes) >= 2:
            parent = notation_of(path_nodes[-2])

        for lang in LANGS:
            try:
                label = node(lang) or ""
            except Exception:
                label = ""

            label = label.strip()

            if not label:
                skipped_empty += 1
                continue

            cur.execute(
                """
                INSERT OR REPLACE INTO iconclass(notation, lang, label, parent, depth)
                VALUES (?, ?, ?, ?, ?);
                """,
                (notation, lang, label, parent, depth)
            )

            inserted += 1

            if "(+" in notation:
                inserted_variants += 1

            if inserted % 100000 == 0:
                con.commit()
                print(f"{inserted} entries...")

    cur.execute("DROP TABLE IF EXISTS hierarchy;")

    cur.execute("""
        CREATE TABLE hierarchy (
            notation TEXT PRIMARY KEY,
            parent TEXT,
            depth INTEGER NOT NULL
        );
    """)

    cur.execute("CREATE INDEX idx_hierarchy_parent ON hierarchy(parent);")

    cur.executemany(
        "INSERT OR REPLACE INTO hierarchy(notation, parent, depth) VALUES (?, ?, ?);",
        [(n, p, d) for n, (p, d) in structure.items()]
    )

    print(f"Hierarchy rows: {len(structure)}")


    con.commit()
    con.close()

    print(f"Database built: {DB_PATH}")
    print(f"Total source notations: {len(all_notations)}")
    print(f"Inserted entries: {inserted}")
    print(f"Inserted variants: {inserted_variants}")
    print(f"Skipped by filtering rule: {skipped_filtered}")
    print(f"Skipped empty labels: {skipped_empty}")


if __name__ == "__main__":
    build_db()