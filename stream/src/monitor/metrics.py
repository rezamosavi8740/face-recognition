from prometheus_client import Counter, Gauge, Histogram



prometheus_inference_latency = Histogram('bina_inference_request_latency_seconds', 'Inference roundtrip time',['stream_id', 'model'])
prometheus_queue_size = Gauge('bina_model_queue_size', 'Number of frames in capture queue',['stream_id', 'model'])
prometheus_frame_count = Counter('bina_frame_count_total', 'reading frames from stream',['stream_id'])
prometheus_stream_up = Gauge('bina_stream_up', 'stream is up',['stream_id'])
prometheus_model_count = Counter('bina_model_sample_count_total', 'flow samples in model',['stream_id', 'model'])


