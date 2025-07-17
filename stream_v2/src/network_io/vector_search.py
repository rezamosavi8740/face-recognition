from src.config import CONFIG
from pymilvus import AsyncMilvusClient

class MilvousSearch:
    def __init__(self):
        self.MILVUS_URL = CONFIG.vector_search.url
        self.COLLECTION_NAME = CONFIG.vector_search.collection_name
        self.VECTOR_FIELD    = "vector"
        self.TOP_K           = 5
        self.has_loaded = False

    async def init_client(self):
        if not hasattr(self, "async_client"):
            self.async_client = AsyncMilvusClient(uri=self.MILVUS_URL, token="root:Milvus")
            await self.load_my_collection()


    async def load_my_collection(self,):
        await self.async_client.load_collection(self.COLLECTION_NAME)
        self.has_loaded = True

    def _display_search_results(self, result):
        # results is a list of lists (one sublist per query vector)
        all_results = []
        for rank, hit in enumerate(result, start=1):
            ent = hit.entity
            all_results.append({

                "url": ent.get('url'),
                "id": str(ent.get('identity_id')),
                "name" : str(ent.get('identity_id')),
                "score": (1 + hit.distance)/2
            })
        return all_results
       
    async def do_search(self, embeddings, logger=None):
        if self.has_loaded == False:
            self.async_client = AsyncMilvusClient(
                                                    uri=self.MILVUS_URL,
                                                    token="root:Milvus"
                                                )
            await self.load_my_collection()

        try:
            results = await self.async_client.search(
                collection_name=self.COLLECTION_NAME,
                data=embeddings,
                anns_field=self.VECTOR_FIELD,
                # param={"metric_type": "COSINE", "params": {"ef": 128}},
                limit= self.TOP_K,
                # expr=None,
                #output_fields=["identity_id", "url"]
                output_fields=["identity_id"]
            )
            # out = {count:self._display_search_results(result) for count, result in enumerate(results)}
            out = [self._display_search_results(result) for  result in results]

            return out
        except Exception:
            logger.error("Milvus search failed:")
            return None