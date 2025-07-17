from typing import List
import uuid
import time

class IdentityManager:
    def __init__(self, milvus_client, collection_name):
        self.client = milvus_client
        self.collection_name = collection_name

    async def identity_exists(self, identity_id: int) -> bool:
        assert isinstance(identity_id, int), f"identity_id must be int, got {type(identity_id)}"

        # توصیه: قبلش flush مطمئن برای دیتای تازه
        await self.client.flush([self.collection_name])

        expr = f"identity_id == {identity_id}"
        results = await self.client.query(
            collection_name=self.collection_name,
            expr=expr,
            output_fields=["identity_id"]
        )

        return len(results) > 0

    async def get_next_identity_id(self) -> int:
        # تولید عدد یونیک بر اساس timestamp
        return int(time.time() * 1000)

    async def insert_identity_vectors(self, identity_id: int, vectors: List, url: str = "N/A"):
        """
        Insert vectors into Milvus with correct data types.
        Each vector gets a unique UUID as vector_id.
        """
        # آماده‌سازی داده به صورت row-based
        data = [
            {
                "vector_id": str(uuid.uuid4()),      # رشته با max_length=100
                "identity_id": identity_id,          # عدد صحیح int64
                #"url": url,                          # رشته با max_length=500
                "vector": vector.tolist()            # لیست 512 تایی از float
            }
            for vector in vectors
        ]

        # درج در Milvus
        await self.client.insert(collection_name=self.collection_name, data=data)