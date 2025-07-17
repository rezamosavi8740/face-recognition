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

        looklike = profile.get("looklike") or [{}]
        top_match = looklike[0]

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
                "conf": top_match.get("score", 0.0),
                "url": top_match.get("url", " "),
                "identity_id": top_match.get("id", " "),
            },
            "face_box": list(profile.get("face_bbox", [])),
            "face_conf": profile.get("face_score"),
            "body_box": list(profile.get("bbox", [])),
            # for method on body, we have 2 thing: conf and class, the gender_conf and hijab_conf both is body_conf: body_conf -> result["heijab_gender2_score"], the gender_class and hijab_class are different : are in result["hijab2"], result["gender2"]
            "body_conf": profile.get("heijab_gender2_score"),
            "gender_class": profile.get("gender2"),
            "hijab_class": profile.get("hijab2"),
            "identity": identity_data,
            "gender2_class": profile.get("gender"),
            "hijab2_class": profile.get("hijab"),
            "gender2_conf": profile.get("gender_score"),
            "hijab2_conf": profile.get("hijab_score"),
            "temp_unique_id": profile.get("unique_id", " "),
            "temp_unique_cosine": profile.get("unique_cosine", 0.0),
            "temp_similar_to": profile.get("similar_to", " "),
        }

        return doc_body if single else {"_index": self.index, "_source": doc_body}
