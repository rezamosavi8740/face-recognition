from datetime import datetime, timezone
from elasticsearch import AsyncElasticsearch, helpers
from src.config import CONFIG


class ElasticClient:
    def __init__(self, logger=None):
        self.logger = logger
        self.index = CONFIG.database.elasticsearch.index
        self.client = AsyncElasticsearch(
            CONFIG.database.elasticsearch.url,
            verify_certs=False,
            ssl_show_warn=False,
            timeout=1000,
        )

    async def upload_profiles(self, results: list, stream_id):
        bulk_docs = [self.create_es_document(result, stream_id, single=False) for result in results]

        try:
            response = await helpers.async_bulk(self.client, bulk_docs)
            if self.logger:
                self.logger.info(f"Elasticsearch bulk insert success: {response}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"[Elasticsearch] Bulk insert error: {e}")

    def create_es_document(self, profile: dict, stream_id, single=False) -> dict:
        identity_data = []
        for item in profile.get("looklike", []):
            identity_data.append({
                "rokn_abadi": {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "face_urls": [{"conf":item.get("score"), "url":item.get("url")}],
                }
            })

        doc_body = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "is_stream": True,
            "track_id": profile.get("track_id"),
            "video_id": stream_id,
            "frame_number": profile.get("frame_count"),
            "frame_shape": profile["frame_shape"],
            "frame_url": profile.get("frame_link"),
            "body_url": profile.get("body_link"),
            "face_url": profile.get("face_link"),
            "best_match": {
                "conf": profile.get("looklike", [{}])[0].get("score", 0.0),
                "url": profile.get("looklike", [{}])[0].get("url", " "),
                "identity_id": profile.get("looklike", [{}])[0].get("id", " "),
            },
            "face_box": list(profile.get("face_bbox", [])),
            "face_conf": profile.get("face_score"),
            "body_box": list(profile.get("bbox", [])),
            "body_conf": profile.get("hijab_score", 0.0),
            "gender_class": profile.get("gender", " "),
            "hijab_class": profile.get("hijab", " "),
            "identity": identity_data,
            "gender2_class": profile.get("gender", " "),
            "hijab2_class": profile.get("hijab", " "),
            "gender2_conf": profile.get("gender_score", 0.0),
            "hijab2_conf": profile.get("hijab2_score", 0.0),
            "temp_unique_id": profile.get("unique_id", " "),
            "temp_unique_cosine": profile.get("unique_cosine", 0.0),
            "temp_similar_to": profile.get("similar_to", " "),
        }

        return doc_body if single else {"_index": self.index, "_source": doc_body}
