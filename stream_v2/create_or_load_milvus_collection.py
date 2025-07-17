from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility

COLLECTION_NAME = "test_collection"
VECTOR_DIM = 512
MILVUS_URL = "http://localhost:19530"

def connect_milvus():
    connections.connect("default", uri=MILVUS_URL)

def collection_exists(name):
    return utility.has_collection(name)

def create_collection(name):
    fields = [
        FieldSchema(name="vector_id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="identity_id", dtype=DataType.INT64),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
    ]
    schema = CollectionSchema(fields=fields, description="Test collection for identity embeddings")
    collection = Collection(name=name, schema=schema)
    print(f"Collection '{name}' created.")
    return collection

def create_index_if_needed(collection: Collection):
    # Check if index exists
    existing_indexes = collection.indexes
    if not existing_indexes:
        print("No index found. Creating index on 'vector' field...")
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 32, "efConstruction": 200}
        }
        collection.create_index(field_name="vector", index_params=index_params)
        collection.flush()
        print("Index created and flushed.")
    else:
        print("Index already exists.")

def load_and_show_preview(collection: Collection):
    try:
        collection.load()
        print(f"Collection '{collection.name}' loaded.")
    except Exception as e:
        print(f"Could not load collection '{collection.name}': {e}")
        return

    try:
        num = collection.num_entities
        print(f"Total records: {num}")

        if num > 0:
            res = collection.query(
                output_fields=["vector_id", "identity_id"],
                limit=min(10, num)
            )
            print("Sample data (up to 10 rows):")
            for r in res:
                print(f"identity_id: {r['identity_id']}, vector_id: {r['vector_id']}")
        else:
            print("No records found.")
    except Exception as e:
        print(f"Query failed: {e}")

if __name__ == "__main__":
    connect_milvus()

    is_created = False
    if collection_exists(COLLECTION_NAME):
        collection = Collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    else:
        collection = create_collection(COLLECTION_NAME)
        is_created = True

    create_index_if_needed(collection)

    if is_created:
        print("Collection created and ready for insert or testing.")

    load_and_show_preview(collection)