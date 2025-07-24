from typing import Sequence, Union
import uuid
import secrets
import numbers
import numpy as np

class IdentityManager:
    """Utility for querying and inserting identity vectors in Milvus."""

    def __init__(self, milvus_client, collection_name: str, auto_flush: bool = False) -> None:
        self.client = milvus_client
        self.collection_name = collection_name
        self.auto_flush = auto_flush  # flush collection automatically after insert if True

    # ──────────────────────────────────────────────────────────────────────────────

    async def identity_exists(self, identity_id) -> bool:
        if isinstance(identity_id, np.integer):
            identity_id = int(identity_id)
        if isinstance(identity_id, str) and identity_id.isdecimal():
            identity_id = int(identity_id)

        if not isinstance(identity_id, numbers.Integral):
            self.logger.warning(f"❌ Unexpected identity_id type: {type(identity_id)} → {identity_id}")
            return False

        rows = await self.client.query(
            self.collection_name,  # ← collection_name
            f"identity_id == {identity_id}",  # ← expr (positional)
            output_fields=["identity_id"],  # ← keyword
        )
        return bool(rows)
    # ──────────────────────────────────────────────────────────────────────────────
    async def get_next_identity_id(self) -> int:
        # 12-digit random integer (0–999,999,999,999) – collision-safe for practical usage
        return secrets.randbelow(10**12)

    # ──────────────────────────────────────────────────────────────────────────────
    async def insert_identity_vectors(self, identity_id, vectors, url: str = "N/A"):
        """Insert a list of embeddings; auto-fix or regenerate identity_id if needed."""
        # force identity_id → int64, otherwise make a new one
        if isinstance(identity_id, str) and identity_id.isdecimal():
            identity_id = int(identity_id)
        elif isinstance(identity_id, np.integer):
            identity_id = int(identity_id)
        elif not isinstance(identity_id, numbers.Integral):
            identity_id = await self.get_next_identity_id()

        rows = [
            {
                "vector_id": str(uuid.uuid4()),
                "identity_id": identity_id,
                "url": url,
                "vector": v.tolist(),
            }
            for v in vectors
        ]

        await self.client.insert(self.collection_name, rows)

        """
        # decide whether to flush
        if flush is None:
            flush = self.auto_flush
        if flush:
            await self.client.flush([self.collection_name])
        """