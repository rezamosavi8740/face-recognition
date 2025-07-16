from typing import List
from pymilvus import AsyncMilvusClient


class IdentityManager:
    def __init__(self, milvus_client: AsyncMilvusClient, collection_name: str):
        self.client = milvus_client
        self.collection_name = collection_name

    async def identity_exists(self, identity_id: str) -> bool:
        expr = f'identity_id == "{identity_id}"'
        res = await self.client.query(
            collection_name=self.collection_name,
            expr=expr,
            output_fields=["identity_id"]
        )
        return len(res) > 0

    async def get_next_identity_id(self) -> int:
        res = await self.client.query(
            collection_name=self.collection_name,
            expr=None,
            output_fields=["identity_id"],
            limit=1,
            order_by=[("identity_id", "desc")]
        )
        if res:
            return int(res[0]["identity_id"]) + 1
        return 1

    async def insert_identity_vectors(self, identity_id: int, vectors: List):
        data = {
            "identity_id": [identity_id] * len(vectors),
            "vector": [v.tolist() for v in vectors],
        }
        await self.client.insert(collection_name=self.collection_name, data=data)