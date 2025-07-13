# moniton triton
insice the prometheus.yml file: 
```
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: 'triton'
    static_configs:
      - targets: ['192.168.130.206:8002']  # use 'localhost:8002' if not in Docker

```
then tell prometheus to use these metrics : 
```

docker run -d \
  --name=prometheus \
  -p 9095:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```
now can check the triton_metrics here: 
http://192.168.160.206:9095/targets



docker run -d \
  --name=grafana \
  -p 3055:3000 \
  grafana/grafana

now check in graphana: 
http://192.168.130.206:3055 

datasources -> choose promotheus, then -> put http://192.168.130.206:9095 
in creating dashboard use this json file: 
```json
{
  "annotations": {
    "list": []
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "id": null,
  "iteration": 1620846016810,
  "panels": [
    {
      "datasource": "${DS_PROMETHEUS}",
      "fieldConfig": {
        "defaults": {},
        "overrides": []
      },
      "gridPos": {
        "h": 9,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "id": 1,
      "options": {
        "legend": {
          "displayMode": "table",
          "placement": "bottom"
        }
      },
      "targets": [
        {
          "expr": "rate(nv_inference_request_success[1m])",
          "interval": "",
          "legendFormat": "{{model}}",
          "refId": "A"
        }
      ],
      "title": "Inference Requests per Second",
      "type": "timeseries"
    },
    {
      "datasource": "${DS_PROMETHEUS}",
      "fieldConfig": {
        "defaults": {},
        "overrides": []
      },
      "gridPos": {
        "h": 9,
        "w": 12,
        "x": 12,
        "y": 0
      },
      "id": 2,
      "options": {},
      "targets": [
        {
          "expr": "rate(nv_inference_compute_infer_duration_us[1m]) / rate(nv_inference_count[1m])",
          "legendFormat": "{{model}}",
          "refId": "A"
        }
      ],
      "title": "Average Inference Duration (us)",
      "type": "timeseries"
    },
    {
      "datasource": "${DS_PROMETHEUS}",
      "gridPos": {
        "h": 9,
        "w": 12,
        "x": 0,
        "y": 9
      },
      "id": 3,
      "targets": [
        {
          "expr": "nv_gpu_utilization",
          "legendFormat": "{{gpu_uuid}}",
          "refId": "A"
        }
      ],
      "title": "GPU Utilization",
      "type": "timeseries"
    },
    {
      "datasource": "${DS_PROMETHEUS}",
      "gridPos": {
        "h": 9,
        "w": 12,
        "x": 12,
        "y": 9
      },
      "id": 4,
      "targets": [
        {
          "expr": "nv_gpu_memory_used_bytes",
          "legendFormat": "{{gpu_uuid}}",
          "refId": "A"
        }
      ],
      "title": "GPU Memory Used (bytes)",
      "type": "timeseries"
    }
  ],
  "refresh": "10s",
  "schemaVersion": 30,
  "style": "dark",
  "tags": [
    "triton",
    "inference",
    "gpu",
    "monitoring"
  ],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-30m",
    "to": "now"
  },
  "timepicker": {
    "refresh_intervals": [
      "5s",
      "10s",
      "30s",
      "1m",
      "5m"
    ]
  },
  "timezone": "",
  "title": "Triton Inference Monitoring (Minimal)",
  "uid": "triton-basic",
  "version": 1
}

```

docker run -it --gpus all --name perf_analyse -v $(pwd):/models --network host --memory="100g" --memory-swap="100g"  tensorrtsdk:25.01  

perf_analyzer   -m pose_detection   -u localhost:8091   -i grpc   -b 1   --concurrency-range 30:80:5  --measurement-mode count_windows  --measurement-request-count 150  --percentile=95   --profile-export-file perf_profile.json   -f perf_report.csv --async

so go to perf_analyse:  
<!-- perf_analyzer   -m pose_detection   -u localhost:8090   -i http   -b 1   --concurrency-range 1:100:5  --measurement-mode count_windows  --measurement-request-count 90  --percentile=95   --async -->

















# choose pose estimation smaller:
shape is (288, 480)
[07/10/2025-15:01:36] [I] Trace averages of 10 runs:
[07/10/2025-15:01:36] [I] Average on 10 runs - GPU latency: 2.24279 ms - Host latency: 6.50306 ms (enqueue 6.43328 ms)
[07/10/2025-15:01:36] [I] Average on 10 runs - GPU latency: 2.24112 ms - Host latency: 6.50109 ms (enqueue 6.43088 ms)
[07/10/2025-15:01:36] [I] Average on 10 runs - GPU latency: 2.24287 ms - Host latency: 6.50269 ms (enqueue 6.43198 ms)
[07/10/2025-15:01:36] [I] Average on 10 runs - GPU latency: 2.24279 ms - Host latency: 6.50328 ms (enqueue 6.43046 ms)
[07/10/2025-15:01:36] [I] Average on 10 runs - GPU latency: 2.24264 ms - Host latency: 6.50129 ms (enqueue 6.42915 ms)
[07/10/2025-15:01:36] [I] Average on 10 runs - GPU latency: 2.24332 ms - Host latency: 6.50244 ms (enqueue 6.43248 ms)
[07/10/2025-15:01:36] [I] Average on 10 runs - GPU latency: 2.24099 ms - Host latency: 6.50077 ms (enqueue 6.43013 ms)
[07/10/2025-15:01:36] [I] Average on 10 runs - GPU latency: 2.24188 ms - Host latency: 6.50104 ms (enqueue 6.42877 ms)
[07/10/2025-15:01:36] [I] Average on 10 runs - GPU latency: 2.24253 ms - Host latency: 6.50195 ms (enqueue 6.4321 ms)
[07/10/2025-15:01:36] [I] Average on 10 runs - GPU latency: 2.24159 ms - Host latency: 6.501 ms (enqueue 6.43115 ms)
[07/10/2025-15:01:36] [I] Average on 10 runs - GPU latency: 2.24166 ms - Host latency: 6.50071 ms (enqueue 6.42922 ms)
[07/10/2025-15:01:36] [I] Average on 10 runs - GPU latency: 2.24233 ms - Host latency: 6.50187 ms (enqueue 6.43107 ms)
[07/10/2025-15:01:36] [I] Average on 10 runs - GPU latency: 2.24255 ms - Host latency: 6.5014 ms (enqueue 6.4307 ms)
[07/10/2025-15:01:36] [I] Average on 10 runs - GPU latency: 2.24159 ms - Host latency: 6.50139 ms (enqueue 6.43231 ms)
[07/10/2025-15:01:36] [I] Average on 10 runs - GPU latency: 2.24171 ms - Host latency: 6.50068 ms (enqueue 6.42989 ms)
[07/10/2025-15:01:36] [I] Average on 10 runs - GPU latency: 2.24465 ms - Host latency: 6.50275 ms (enqueue 6.43315 ms)
[07/10/2025-15:01:36] [I] Average on 10 runs - GPU latency: 2.24526 ms - Host latency: 6.50421 ms (enqueue 6.4333 ms)


shape is (480, 800)
[07/10/2025-15:09:19] [I] Average on 10 runs - GPU latency: 4.8903 ms - Host latency: 16.557 ms (enqueue 16.4854 ms)
[07/10/2025-15:09:19] [I] Average on 10 runs - GPU latency: 4.89291 ms - Host latency: 16.5594 ms (enqueue 16.4878 ms)
[07/10/2025-15:09:19] [I] Average on 10 runs - GPU latency: 4.89183 ms - Host latency: 16.5582 ms (enqueue 16.4891 ms)
[07/10/2025-15:09:19] [I] Average on 10 runs - GPU latency: 4.89322 ms - Host latency: 16.5604 ms (enqueue 16.4887 ms)
[07/10/2025-15:09:19] [I] Average on 10 runs - GPU latency: 4.89257 ms - Host latency: 16.5589 ms (enqueue 16.4874 ms)
[07/10/2025-15:09:19] [I] Average on 10 runs - GPU latency: 4.88932 ms - Host latency: 16.5554 ms (enqueue 16.484 ms)
[07/10/2025-15:09:19] [I] Average on 10 runs - GPU latency: 4.88994 ms - Host latency: 16.5563 ms (enqueue 16.4849 ms)
[07/10/2025-15:09:19] [I] Average on 10 runs - GPU latency: 4.88831 ms - Host latency: 16.5547 ms (enqueue 16.4828 ms)
[07/10/2025-15:09:19] [I] Average on 10 runs - GPU latency: 4.89308 ms - Host latency: 16.5598 ms (enqueue 16.4878 ms)
[07/10/2025-15:09:19] [I] Average on 10 runs - GPU latency: 4.89099 ms - Host latency: 16.5579 ms (enqueue 16.4855 ms)
[07/10/2025-15:09:19] [I] Average on 10 runs - GPU latency: 4.90148 ms - Host latency: 16.568 ms (enqueue 16.4971 ms)
[07/10/2025-15:09:19] [I] Average on 10 runs - GPU latency: 4.89996 ms - Host latency: 16.5658 ms (enqueue 16.4943 ms)
[07/10/2025-15:09:19] [I] Average on 10 runs - GPU latency: 4.898 ms - Host latency: 16.5644 ms (enqueue 16.4924 ms)
[07/10/2025-15:09:19] [I] Average on 10 runs - GPU latency: 4.89939 ms - Host latency: 16.5657 ms (enqueue 16.4943 ms)
[07/10/2025-15:09:19] [I] Average on 10 runs - GPU latency: 4.90427 ms - Host latency: 16.5708 ms (enqueue 16.4992 ms)
[07/10/2025-15:09:19] [I] Average on 10 runs - GPU latency: 4.90041 ms - Host latency: 16.5669 ms (enqueue 16.4933 ms)
[07/10/2025-15:09:19] [I] Average on 10 runs - GPU latency: 4.8998 ms - Host latency: 16.566 ms (enqueue 16.4938 ms)
[07/10/2025-15:09:19] [I] Average on 10 runs - GPU latency: 4.89695 ms - Host latency: 16.5632 ms (enqueue 16.4938 ms)








# test with perf on 4090 small pose (50 instance triton for pose): 

root@ai-vms:/workspace# 

perf_analyzer   -m pose_detection   -u localhost:8090   -i http   -b 1   --concurrency-range 1:80:5  --measurement-mode count_windows  --measurement-request-count 200  --percentile=95   --profile-export-file perf_profile.json   -f perf_report.csv --async

*** Measurement Settings ***
  Batch size: 1
  Service Kind: TRITON
  Using "count_windows" mode for stabilization
  Stabilizing using p95throughput
  Minimum number of samples in each window: 200
  Latency limit: 0 msec
  Concurrency limit: 80 concurrent requests
  Using asynchronous calls for inference

Request concurrency: 1
  Client: 
    Request count: 673
    Throughput: 72.3764 infer/sec
    p50 latency: 12901 usec
    p90 latency: 13662 usec
    p95 latency: 13957 usec
    p99 latency: 14800 usec
    Avg HTTP time: 12906 usec (send/recv 974 usec + response wait 11932 usec)
  Server: 
    Inference count: 674
    Execution count: 674
    Successful request count: 674
    Avg request latency: 9635 usec (overhead 220 usec + queue 474 usec + compute 8941 usec)

  Composing models: 
  pose_core, version: 1
      Inference count: 674
      Execution count: 674
      Successful request count: 674
      Avg request latency: 7946 usec (overhead 183 usec + queue 244 usec + compute input 118 usec + compute infer 7148 usec + compute output 252 usec)

  preprocess, version: 1
      Inference count: 674
      Execution count: 674
      Successful request count: 674
      Avg request latency: 1654 usec (overhead 2 usec + queue 230 usec + compute input 14 usec + compute infer 1231 usec + compute output 176 usec)

Request concurrency: 6
  Client: 
    Request count: 2275
    Throughput: 524.721 infer/sec
    p50 latency: 10137 usec
    p90 latency: 13069 usec
    p95 latency: 14068 usec
    p99 latency: 18467 usec
    Avg HTTP time: 10614 usec (send/recv 941 usec + response wait 9673 usec)
  Server: 
    Inference count: 2277
    Execution count: 2277
    Successful request count: 2277
    Avg request latency: 6989 usec (overhead 354 usec + queue 1808 usec + compute 4827 usec)

  Composing models: 
  pose_core, version: 1
      Inference count: 2277
      Execution count: 1207
      Successful request count: 2277
      Avg request latency: 5557 usec (overhead 322 usec + queue 1652 usec + compute input 46 usec + compute infer 3340 usec + compute output 196 usec)

  preprocess, version: 1
      Inference count: 2275
      Execution count: 2189
      Successful request count: 2275
      Avg request latency: 1402 usec (overhead 2 usec + queue 156 usec + compute input 13 usec + compute infer 1051 usec + compute output 178 usec)

Request concurrency: 11
  Client: 
    Request count: 3430
    Throughput: 724.822 infer/sec
    p50 latency: 14247 usec
    p90 latency: 19174 usec
    p95 latency: 20227 usec
    p99 latency: 23411 usec
    Avg HTTP time: 14361 usec (send/recv 989 usec + response wait 13372 usec)
  Server: 
    Inference count: 3430
    Execution count: 3430
    Successful request count: 3430
    Avg request latency: 8382 usec (overhead 564 usec + queue 2784 usec + compute 5034 usec)

  Composing models: 
  pose_core, version: 1
      Inference count: 3430
      Execution count: 1228
      Successful request count: 3430
      Avg request latency: 6922 usec (overhead 537 usec + queue 2640 usec + compute input 61 usec + compute infer 3479 usec + compute output 203 usec)

  preprocess, version: 1
      Inference count: 3435
      Execution count: 3108
      Successful request count: 3435
      Avg request latency: 1436 usec (overhead 3 usec + queue 144 usec + compute input 14 usec + compute infer 1075 usec + compute output 199 usec)

Request concurrency: 16
  Client: 
    Request count: 4216
    Throughput: 839.906 infer/sec
    p50 latency: 18294 usec
    p90 latency: 24985 usec
    p95 latency: 26722 usec
    p99 latency: 32513 usec
    Avg HTTP time: 18240 usec (send/recv 1051 usec + response wait 17189 usec)
  Server: 
    Inference count: 4216
    Execution count: 4216
    Successful request count: 4216
    Avg request latency: 9840 usec (overhead 705 usec + queue 3659 usec + compute 5476 usec)

  Composing models: 
  pose_core, version: 1
      Inference count: 4216
      Execution count: 1136
      Successful request count: 4216
      Avg request latency: 8354 usec (overhead 681 usec + queue 3531 usec + compute input 74 usec + compute infer 3848 usec + compute output 219 usec)

  preprocess, version: 1
      Inference count: 4215
      Execution count: 3712
      Successful request count: 4215
      Avg request latency: 1466 usec (overhead 4 usec + queue 128 usec + compute input 15 usec + compute infer 1111 usec + compute output 208 usec)

Request concurrency: 21
  Client: 
    Request count: 3863
    Throughput: 899.947 infer/sec
    p50 latency: 25221 usec
    p90 latency: 30736 usec
    p95 latency: 32433 usec
    p99 latency: 36310 usec
    Avg HTTP time: 22554 usec (send/recv 1060 usec + response wait 21494 usec)
  Server: 
    Inference count: 3870
    Execution count: 3870
    Successful request count: 3870
    Avg request latency: 11409 usec (overhead 1287 usec + queue 3671 usec + compute 6451 usec)

  Composing models: 
  pose_core, version: 1
      Inference count: 3870
      Execution count: 945
      Successful request count: 3870
      Avg request latency: 9710 usec (overhead 1265 usec + queue 3545 usec + compute input 109 usec + compute infer 4540 usec + compute output 250 usec)

  preprocess, version: 1
      Inference count: 3855
      Execution count: 3167
      Successful request count: 3855
      Avg request latency: 1681 usec (overhead 4 usec + queue 126 usec + compute input 15 usec + compute infer 1307 usec + compute output 227 usec)

Request concurrency: 26
  Client: 
    Request count: 5328
    Throughput: 1000.66 infer/sec
    p50 latency: 28261 usec
    p90 latency: 34295 usec
    p95 latency: 35340 usec
    p99 latency: 40266 usec
    Avg HTTP time: 25080 usec (send/recv 1107 usec + response wait 23973 usec)
  Server: 
    Inference count: 5327
    Execution count: 5327
    Successful request count: 5327
    Avg request latency: 12039 usec (overhead 1301 usec + queue 4403 usec + compute 6335 usec)

  Composing models: 
  pose_core, version: 1
      Inference count: 5327
      Execution count: 1146
      Successful request count: 5327
      Avg request latency: 10391 usec (overhead 1279 usec + queue 4286 usec + compute input 113 usec + compute infer 4467 usec + compute output 245 usec)

  preprocess, version: 1
      Inference count: 5325
      Execution count: 4444
      Successful request count: 5325
      Avg request latency: 1629 usec (overhead 3 usec + queue 117 usec + compute input 15 usec + compute infer 1273 usec + compute output 220 usec)

Request concurrency: 31
  Client: 
    Request count: 5557
    Throughput: 1157.08 infer/sec
    p50 latency: 26244 usec
    p90 latency: 36487 usec
    p95 latency: 38342 usec
    p99 latency: 43930 usec
    Avg HTTP time: 25936 usec (send/recv 1125 usec + response wait 24811 usec)
  Server: 
    Inference count: 5559
    Execution count: 5559
    Successful request count: 5559
    Avg request latency: 13564 usec (overhead 1261 usec + queue 5861 usec + compute 6442 usec)

  Composing models: 
  pose_core, version: 1
      Inference count: 5559
      Execution count: 982
      Successful request count: 5559
      Avg request latency: 11990 usec (overhead 1238 usec + queue 5748 usec + compute input 110 usec + compute infer 4639 usec + compute output 254 usec)

  preprocess, version: 1
      Inference count: 5565
      Execution count: 4683
      Successful request count: 5565
      Avg request latency: 1553 usec (overhead 2 usec + queue 113 usec + compute input 15 usec + compute infer 1206 usec + compute output 215 usec)

Request concurrency: 36
  Client: 
    Request count: 5649
    Throughput: 1253.49 infer/sec
    p50 latency: 28747 usec
    p90 latency: 40007 usec
    p95 latency: 42454 usec
    p99 latency: 46820 usec
    Avg HTTP time: 27914 usec (send/recv 1133 usec + response wait 26781 usec)
  Server: 
    Inference count: 5651
    Execution count: 5651
    Successful request count: 5651
    Avg request latency: 14012 usec (overhead 1525 usec + queue 5626 usec + compute 6861 usec)

  Composing models: 
  pose_core, version: 1
      Inference count: 5651
      Execution count: 875
      Successful request count: 5651
      Avg request latency: 12359 usec (overhead 1502 usec + queue 5516 usec + compute input 120 usec + compute infer 4959 usec + compute output 261 usec)

  preprocess, version: 1
      Inference count: 5625
      Execution count: 4652
      Successful request count: 5625
      Avg request latency: 1633 usec (overhead 3 usec + queue 110 usec + compute input 15 usec + compute infer 1285 usec + compute output 219 usec)

Request concurrency: 41
  Client: 
    Request count: 6365
    Throughput: 1370.65 infer/sec
    p50 latency: 28291 usec
    p90 latency: 41599 usec
    p95 latency: 43841 usec
    p99 latency: 48889 usec
    Avg HTTP time: 29120 usec (send/recv 1147 usec + response wait 27973 usec)
  Server: 
    Inference count: 6363
    Execution count: 6363
    Successful request count: 6363
    Avg request latency: 15119 usec (overhead 1671 usec + queue 6144 usec + compute 7304 usec)

  Composing models: 
  pose_core, version: 1
      Inference count: 6363
      Execution count: 833
      Successful request count: 6363
      Avg request latency: 13442 usec (overhead 1646 usec + queue 6036 usec + compute input 177 usec + compute infer 5308 usec + compute output 274 usec)

  preprocess, version: 1
      Inference count: 6359
      Execution count: 5314
      Successful request count: 6359
      Avg request latency: 1655 usec (overhead 3 usec + queue 108 usec + compute input 15 usec + compute infer 1298 usec + compute output 230 usec)

Request concurrency: 46
  Client: 
    Request count: 6692
    Throughput: 1369.54 infer/sec
    p50 latency: 32760 usec
    p90 latency: 45041 usec
    p95 latency: 47419 usec
    p99 latency: 52382 usec
    Avg HTTP time: 32726 usec (send/recv 1161 usec + response wait 31565 usec)
  Server: 
    Inference count: 6691
    Execution count: 6691
    Successful request count: 6691
    Avg request latency: 17502 usec (overhead 1672 usec + queue 8119 usec + compute 7711 usec)

  Composing models: 
  pose_core, version: 1
      Inference count: 6691
      Execution count: 849
      Successful request count: 6691
      Avg request latency: 15639 usec (overhead 1648 usec + queue 8012 usec + compute input 478 usec + compute infer 5226 usec + compute output 274 usec)

  preprocess, version: 1
      Inference count: 6693
      Execution count: 5533
      Successful request count: 6693
      Avg request latency: 1842 usec (overhead 3 usec + queue 107 usec + compute input 15 usec + compute infer 1369 usec + compute output 346 usec)

Request concurrency: 51
  Client: 
    Request count: 6080
    Throughput: 1388.12 infer/sec
    p50 latency: 35200 usec
    p90 latency: 48665 usec
    p95 latency: 51295 usec
    p99 latency: 57522 usec
    Avg HTTP time: 35795 usec (send/recv 1169 usec + response wait 34626 usec)
  Server: 
    Inference count: 6093
    Execution count: 6093
    Successful request count: 6093
    Avg request latency: 21677 usec (overhead 2149 usec + queue 9867 usec + compute 9661 usec)

  Composing models: 
  pose_core, version: 1
      Inference count: 6094
      Execution count: 591
      Successful request count: 6091
      Avg request latency: 19522 usec (overhead 2124 usec + queue 9759 usec + compute input 1210 usec + compute infer 6126 usec + compute output 302 usec)

  preprocess, version: 1
      Inference count: 6084
      Execution count: 4978
      Successful request count: 6084
      Avg request latency: 2134 usec (overhead 4 usec + queue 108 usec + compute input 16 usec + compute infer 1466 usec + compute output 539 usec)

Request concurrency: 56
  Client: 
    Request count: 11563
    Throughput: 1352.98 infer/sec
    p50 latency: 40519 usec
    p90 latency: 53658 usec
    p95 latency: 57434 usec
    p99 latency: 63996 usec
    Avg HTTP time: 40363 usec (send/recv 1210 usec + response wait 39153 usec)
  Server: 
    Inference count: 11563
    Execution count: 11563
    Successful request count: 11563
    Avg request latency: 25745 usec (overhead 2324 usec + queue 12212 usec + compute 11209 usec)

  Composing models: 
  pose_core, version: 1
      Inference count: 11563
      Execution count: 1017
      Successful request count: 11563
      Avg request latency: 23115 usec (overhead 2300 usec + queue 12103 usec + compute input 1700 usec + compute infer 6696 usec + compute output 316 usec)

  preprocess, version: 1
      Inference count: 11567
      Execution count: 9326
      Successful request count: 11567
      Avg request latency: 2609 usec (overhead 3 usec + queue 109 usec + compute input 16 usec + compute infer 1691 usec + compute output 789 usec)

Request concurrency: 61
  Client: 
    Request count: 6239
    Throughput: 1400.63 infer/sec
    p50 latency: 43479 usec
    p90 latency: 54915 usec
    p95 latency: 58147 usec
    p99 latency: 63936 usec
    Avg HTTP time: 42534 usec (send/recv 1189 usec + response wait 41345 usec)
  Server: 
    Inference count: 6229
    Execution count: 6229
    Successful request count: 6229
    Avg request latency: 27911 usec (overhead 2529 usec + queue 13610 usec + compute 11772 usec)

  Composing models: 
  pose_core, version: 1
      Inference count: 6229
      Execution count: 487
      Successful request count: 6229
      Avg request latency: 25151 usec (overhead 2511 usec + queue 13503 usec + compute input 2277 usec + compute infer 6554 usec + compute output 305 usec)

  preprocess, version: 1
      Inference count: 6243
      Execution count: 5153
      Successful request count: 6243
      Avg request latency: 2744 usec (overhead 2 usec + queue 107 usec + compute input 16 usec + compute infer 1683 usec + compute output 935 usec)

Request concurrency: 66
  Client: 
    Request count: 5357
    Throughput: 1350.18 infer/sec
    p50 latency: 48339 usec
    p90 latency: 60298 usec
    p95 latency: 63628 usec
    p99 latency: 72447 usec
    Avg HTTP time: 47940 usec (send/recv 1172 usec + response wait 46768 usec)
  Server: 
    Inference count: 5357
    Execution count: 5357
    Successful request count: 5357
    Avg request latency: 35814 usec (overhead 3214 usec + queue 18389 usec + compute 14211 usec)

  Composing models: 
  pose_core, version: 1
      Inference count: 5357
      Execution count: 359
      Successful request count: 5357
      Avg request latency: 32545 usec (overhead 3157 usec + queue 18277 usec + compute input 2882 usec + compute infer 7882 usec + compute output 347 usec)

  preprocess, version: 1
      Inference count: 5365
      Execution count: 4438
      Successful request count: 5365
      Avg request latency: 3215 usec (overhead 3 usec + queue 112 usec + compute input 16 usec + compute infer 2004 usec + compute output 1079 usec)

Request concurrency: 71
  Client: 
    Request count: 5771
    Throughput: 1322.14 infer/sec
    p50 latency: 52593 usec
    p90 latency: 65660 usec
    p95 latency: 69719 usec
    p99 latency: 77772 usec
    Avg HTTP time: 52698 usec (send/recv 1174 usec + response wait 51524 usec)
  Server: 
    Inference count: 5765
    Execution count: 5765
    Successful request count: 5765
    Avg request latency: 41736 usec (overhead 3124 usec + queue 23569 usec + compute 15043 usec)

  Composing models: 
  pose_core, version: 1
      Inference count: 5765
      Execution count: 377
      Successful request count: 5765
      Avg request latency: 38111 usec (overhead 3084 usec + queue 23458 usec + compute input 3188 usec + compute infer 8025 usec + compute output 355 usec)

  preprocess, version: 1
      Inference count: 5778
      Execution count: 4870
      Successful request count: 5778
      Avg request latency: 3588 usec (overhead 3 usec + queue 111 usec + compute input 16 usec + compute infer 2229 usec + compute output 1228 usec)

Request concurrency: 76
  Client: 
    Request count: 6030
    Throughput: 1278.12 infer/sec
    p50 latency: 57694 usec
    p90 latency: 72180 usec
    p95 latency: 76108 usec
    p99 latency: 84867 usec
    Avg HTTP time: 58462 usec (send/recv 1159 usec + response wait 57303 usec)
  Server: 
    Inference count: 6032
    Execution count: 6032
    Successful request count: 6032
    Avg request latency: 47052 usec (overhead 2737 usec + queue 27595 usec + compute 16720 usec)

  Composing models: 
  pose_core, version: 1
      Inference count: 6032
      Execution count: 383
      Successful request count: 6032
      Avg request latency: 42590 usec (overhead 2714 usec + queue 27479 usec + compute input 3446 usec + compute infer 8595 usec + compute output 355 usec)

  preprocess, version: 1
      Inference count: 6011
      Execution count: 4883
      Successful request count: 6011
      Avg request latency: 4442 usec (overhead 3 usec + queue 116 usec + compute input 16 usec + compute infer 2849 usec + compute output 1457 usec)

Inferences/Second vs. Client p95 Batch Latency
Concurrency: 1, throughput: 72.3764 infer/sec, latency 13957 usec
Concurrency: 6, throughput: 524.721 infer/sec, latency 14068 usec
Concurrency: 11, throughput: 724.822 infer/sec, latency 20227 usec
Concurrency: 16, throughput: 839.906 infer/sec, latency 26722 usec
Concurrency: 21, throughput: 899.947 infer/sec, latency 32433 usec
Concurrency: 26, throughput: 1000.66 infer/sec, latency 35340 usec
Concurrency: 31, throughput: 1157.08 infer/sec, latency 38342 usec
Concurrency: 36, throughput: 1253.49 infer/sec, latency 42454 usec
Concurrency: 41, throughput: 1370.65 infer/sec, latency 43841 usec
Concurrency: 46, throughput: 1369.54 infer/sec, latency 47419 usec
Concurrency: 51, throughput: 1388.12 infer/sec, latency 51295 usec
Concurrency: 56, throughput: 1352.98 infer/sec, latency 57434 usec
Concurrency: 61, throughput: 1400.63 infer/sec, latency 58147 usec
Concurrency: 66, throughput: 1350.18 infer/sec, latency 63628 usec
Concurrency: 71, throughput: 1322.14 infer/sec, latency 69719 usec
Concurrency: 76, throughput: 1278.12 infer/sec, latency 76108 usec