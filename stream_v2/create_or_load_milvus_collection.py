"""
milvus_setup.py

Convenience script to (re-)create a Milvus collection compatible with IdentityManager
and optionally preview its contents.

Usage (interactive):
    $ python milvus_setup.py
"""

from pymilvus import (
    connections,
    CollectionSchema,
    FieldSchema,
    DataType,
    Collection,
    utility,
)

# ─────────────────────────── Config ────────────────────────────
COLLECTION_NAME = "test_collection"
VECTOR_DIM = 512
MILVUS_URL = "http://localhost:19530"

# ────────────────────────── Functions ──────────────────────────
def connect_milvus() -> None:
    """Connect to the default Milvus server."""
    connections.connect(alias="default", uri=MILVUS_URL)


def collection_exists(name: str) -> bool:
    """Return True if a collection with the given name exists."""
    return utility.has_collection(name)


def drop_collection(name: str) -> None:
    """Drop the collection if it exists."""
    if collection_exists(name):
        utility.drop_collection(name)
        print(f"Collection '{name}' dropped.")


def create_collection(name: str) -> Collection:
    """Create a collection schema compatible with IdentityManager."""
    fields = [
        FieldSchema(
            name="vector_id",
            dtype=DataType.VARCHAR,
            max_length=100,
            is_primary=True,  # primary key
        ),
        FieldSchema(name="identity_id", dtype=DataType.INT64),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
    ]
    schema = CollectionSchema(
        fields=fields,
        description="Identity embeddings (compatible with IdentityManager)",
    )
    collection = Collection(name=name, schema=schema)
    print(f"Collection '{name}' created.")
    return collection


def create_index_if_needed(collection: Collection) -> None:
    """Create an HNSW index on the `vector` field if none exists."""
    if collection.indexes:
        print("Index already exists.")
        return

    print("No index found. Creating HNSW index on 'vector' field ...")
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 32, "efConstruction": 200},
    }
    collection.create_index(field_name="vector", index_params=index_params)
    collection.flush()
    print("Index created and flushed.")


def show_preview(collection: Collection) -> None:
    """Load the collection and print record count + up to 10 sample rows."""
    try:
        collection.load()
    except Exception as exc:
        print(f"Could not load collection: {exc}")
        return

    num = collection.num_entities
    print(f"Total records: {num}")

    if num == 0:
        return

    try:
        rows = collection.query(
            expr=None,
            output_fields=["vector_id", "identity_id"],
            limit=min(10, num),
        )
    except Exception as exc:
        print(f"Query failed: {exc}")
        return

    print("Sample rows (≤10):")
    for r in rows:
        print(f"identity_id={r['identity_id']}, vector_id={r['vector_id']}")


# ───────────────────────────── Main ────────────────────────────
if __name__ == "__main__":
    connect_milvus()

    if collection_exists(COLLECTION_NAME):
        answer = input(
            f"Collection '{COLLECTION_NAME}' already exists. "
            "Delete and recreate? [y/N]: "
        ).strip().lower()
        if answer == "y":
            drop_collection(COLLECTION_NAME)
            collection = create_collection(COLLECTION_NAME)
            create_index_if_needed(collection)
        else:
            collection = Collection(COLLECTION_NAME)
            print("Keeping existing collection.")
    else:
        collection = create_collection(COLLECTION_NAME)
        create_index_if_needed(collection)

    preview_answer = input("Show record count and sample rows? [y/N]: ").strip().lower()
    if preview_answer == "y":
        show_preview(collection)