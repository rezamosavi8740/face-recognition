import asyncio
from aiortc.contrib.media import MediaPlayer
from src.config import CONFIG, setup_logger, format_with_stars
from src.triton.client import TritonModelManager, TritonModel
from src.mv_utils.on_head import  face_box_extract, face_warp, face_padd, summarize_keypoints 
from src.triton.funcs import FaceAllignment, FaceEmbeding, PoseDetection, HeadGender, HeadHijab
import cv2
import numpy as np
from src.tracker.bytetracker import BYTETracker, Args
from src.network_io.vector_search import MilvousSearch
from src.network_io.media_upload import MinioImageUploader
from src.network_io.profile_upload import ElasticClient
from src.mv_utils.vector_cluster import FaceBank
from src.monitor import metrics 
from src.mv_utils.on_tracks import FaceCandidTrace
# import tracemalloc #plot_landmark,
import av
import sys


class Bina:
    def __init__(self, use_load="aiortc"):
        """
        use_load:
            "cv2"    → use OpenCV VideoCapture
            "av"     → use PyAV (FFmpeg)
            "aiortc" → use self.video.recv() (default)
        """
        self.use_load = use_load
        self.cap = None           # for cv2
        self.av_container = None # for PyAV
        self.av_stream = None
        self.player = None
        self.video = None
        self.triton_client_manager = None
        self.tracker = BYTETracker(Args())
        self.stream_id = CONFIG.stream_id
        self.stream_url = CONFIG.stream_url
        self.fr_landmark_th = CONFIG.fr.landmark
        self.fr_looklike_th = CONFIG.fr.looklike
        self.prometheus = {
            'frame_count': metrics.prometheus_frame_count,
            'queue_size': metrics.prometheus_queue_size,
            'inference_latency': metrics.prometheus_inference_latency,
            'model_count': metrics.prometheus_model_count,
            'stream_up': metrics.prometheus_stream_up
        }
        self.prometheus["stream_up"].labels(stream_id=self.stream_id).set(0)
        self.BATCH_TIMEOUT = 1  # seconds
        self.BATCH_MAX_SIZE = 20   # maximum items in one batch

        self.logger = setup_logger(log_file=CONFIG.dynamic_paths.log, 
                                   name=self.stream_id)
        self.logger.info(format_with_stars("< Welcome to Realtime BinaSystem >", width=80, sign="&"))
        self.logger.info(format_with_stars(self.stream_url, width=80, sign=" "))
        self.logger.info(80*"&")


        
        
        
    @classmethod
    async def create(cls, use_load="aiortc"):
        self = cls(use_load=use_load)  
        

        url = CONFIG.STREAM_URL
        if url.startswith("rtsp://"):
            self.logger.info("Using RTSP stream input")
            self.player = MediaPlayer(url, format="rtsp", options={"rtsp_transport": "tcp"})
        elif url.startswith("rtmp://"):
            self.logger.info("Using RTMP stream input")
            self.player = MediaPlayer(url, format="flv")
        elif url.endswith((".mp4", ".avi", ".mkv")):
            self.logger.info("Using video file input")
            self.player = MediaPlayer(url)
        else:
            self.logger.error(f"Unsupported stream URL format: {url}")
            raise ValueError(f"Unsupported stream URL format: {url}")


        self.video = self.player.video
        self.triton_client_manager = await TritonModelManager.get_instance()

        self.face_allignment = FaceAllignment()
        self.register_model(self.face_allignment)

        self.face_embeding = FaceEmbeding()
        self.register_model(self.face_embeding)

        self.posedetection = PoseDetection()
        self.register_model(self.posedetection)

        self.headgender = HeadGender()
        self.register_model(self.headgender)

        self.headhijab = HeadHijab()
        self.register_model(self.headhijab)

        
        self.vs = MilvousSearch()
        self.minio = MinioImageUploader()
        self.elastic = ElasticClient(logger=self.logger)
        self.candid_trace = FaceCandidTrace(logger=self.logger)
        self.face_bank = FaceBank(max_size=20, similarity_threshold=0.28)
        

        
        
        self.frame_count = 0
        self.head_queue = asyncio.Queue(maxsize=100)
        self.netio_queue = asyncio.Queue(maxsize=100)
        

        return self
    
    
    def register_model(self, model_obj):
        self.triton_client_manager.register_model(TritonModel(
                                                                name=model_obj.name,
                                                                input_name=model_obj.info()["input"][0]["name"],
                                                                output_names= [o["name"] for o in model_obj.info()["output"]],
                                                                pre_process=model_obj.preprocess,
                                                                post_process=model_obj.postprocess,
                                                            ))
                                                            

    def setup_cv2_capture(self):
        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            self.logger.error(f"[cv2] Failed to open stream: {self.stream_url}")

    def setup_av(self):
        try:
            self.av_container = av.open(self.stream_url)
            self.av_stream = self.av_container.streams.video[0]
            self.logger.info("[av] Stream opened successfully.")
        except Exception as e:
            self.logger.error(f"[av] Failed to open stream: {e}")
            self.av_container = None
            self.av_stream = None


    async def read_image(self):
        """Reads image based on selected method (cv2, av, aiortc)"""
        if self.use_load == "cv2":
            if self.cap is None:
                self.setup_cv2_capture()
            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning("[cv2] Failed to read frame")
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        elif self.use_load == "av":
            if self.av_container is None or self.av_stream is None:
                self.setup_av()
                if self.av_container is None or self.av_stream is None:
                    return None  # Abort if still failed

            try:
                for packet in self.av_container.demux(self.av_stream):
                    for frame in packet.decode():
                        return frame.to_ndarray(format="rgb24")
            except Exception as e:
                self.logger.error(f"[av] Error while reading: {e}")
                # Attempt to re-open once
                self.av_container = None
                self.av_stream = None
                return None

        else:  # "aiortc" (default)
            try:
                frame = await self.video.recv()
                return frame.to_ndarray(format="rgb24")
            except Exception as e:
                self.logger.error(f"[aiortc] Failed to read frame: {e}")
                await asyncio.sleep(0.5)
                return None

    
    
    
    async def inference(self):
        tasks = [
            asyncio.create_task(self.pose_worker()),
            asyncio.create_task(self.head_worker()),
            asyncio.create_task(self.netio_worker()),
        ]
    

        await asyncio.gather(*tasks) 
    
    async def pose_worker(self,):
        while True:
            image = await self.read_image()
            if image is None:
                continue
            # image  = cv2.imread('/home/ansarifard/projects/Total_async_bina/2.jpg')
            self.prometheus['frame_count'].labels(self.stream_id).inc()
            self.frame_count +=1 
            self.prometheus["stream_up"].labels(stream_id=self.stream_id).set(1)
            
           
            # if self.frame_count % 100 == 0:
            #     snapshot = tracemalloc.take_snapshot()
            #     top_stats = snapshot.statistics("traceback")
            #     if top_stats:
            #         top_stat = top_stats[0]
            #         size_mb = top_stat.size / (1024 * 1024)
            #         bold = "\033[1m"
            #         reset = "\033[0m"
            #         print(f"\n\n{bold}[WARNING] High memory usage: {size_mb:.2f} MB{reset}\n\n")
            #         for line in top_stat.traceback.format():
            #             print(line)



                        


            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            with self.prometheus['inference_latency'].labels(self.stream_id, "face_detection").time():
                try:
                    results = await asyncio.wait_for(self.triton_client_manager.infer(model_name=self.posedetection.name, raw_input=image), timeout=3*self.BATCH_TIMEOUT)
                except asyncio.TimeoutError:
                    self.logger.error(f"Pose_detection Triton timed out after {3*self.BATCH_TIMEOUT:.2f} seconds")
                    continue
                # results = await self.triton_client_manager.infer(model_name=self.posedetection.name, raw_input=image)
            if not results:
                continue
            results = self.tracker.update(results, image.shape[:2], image.shape[:2])
            self.prometheus['model_count'].labels(self.stream_id, "face_detection").inc()


            # for det in results:
            #     print(f"ID {det['track_id']} at bbox {det['bbox']} with score {det['score']:.2f}")
            # for each id_track must specify wether it has face? better than its historic data? so if yes so send to recognition(head) queue
            
            for result in results:
                face_bbox, face_score = face_box_extract(result["keypoints"], result["bbox"])
                if face_bbox is None:
                    continue
                
                result["face_bbox"] = face_bbox
                result["frame_count"] = self.frame_count
                result["temp_face_score"] = face_score
                result["frame_shape"] = list(image.shape[:2])
                if self.candid_trace.add(result["track_id"], result):
                    # now we can push on other queue
                    await self.head_queue.put((image, result))


            
    
    async def head_worker(self,):
        while True:
            if self.head_queue.full():
                self.logger.error("head_queue is full! Exiting.")
                await self.shutdown()
                sys.exit(1)
            image, result = await self.head_queue.get()
            keypoint_summary = summarize_keypoints(result["keypoints"], min_conf=0.5)
            self.logger.info(f"new instance in head_pip_line: bbox={result['bbox']} {keypoint_summary} track_id={result['track_id']} @ {result['frame_count']}")
            self.prometheus["queue_size"].labels(stream_id=self.stream_id, model='gender_detection').set(self.head_queue.qsize())
            
            x1, y1, x2, y2 = result["face_bbox"]
            face_crop = image[y1:y2, x1:x2, ...]
            with self.prometheus['inference_latency'].labels(self.stream_id, "face_allignment").time():
                try:
                    out_align = await asyncio.wait_for(self.triton_client_manager.infer(model_name=self.face_allignment.name, raw_input=face_crop), timeout=3*self.BATCH_TIMEOUT)
                except asyncio.TimeoutError:
                    self.logger.error(f"face_allignment Triton timed out after {3*self.BATCH_TIMEOUT:.2f} seconds")
                    continue
                # out_align = await self.triton_client_manager.infer(model_name=self.face_allignment.name, raw_input=face_crop)
            self.prometheus['model_count'].labels(self.stream_id, "face_allignment").inc()

            # plot_img = plot_landmark(face_crop, out_align["bbox"], out_align["keypoints"])
            
            if out_align["score"] < self.fr_landmark_th:
                self.logger.info(f"ignore instance, low allignment th {out_align['score']:.2f}: track_id={result['track_id']} @ {result['frame_count']}")
                continue
            x1_cr, y1_cr, x2_cr, y2_cr = out_align["bbox"]
            result["face_score"] = out_align["score"]
            result["face_bbox"] = [x1+x1_cr, y1+y1_cr, x1+x2_cr, y1+y2_cr]
            result["face_keypoints"] = [{"x":x1+item["x"], "y":y1+item["y"]} for item in out_align["keypoints"]]
            # plot_img2 = plot_landmark(image, result["face_bbox"], result["face_keypoints"])
            #just work with result!
            face_crop_padd, face_crop_padd_bbox, face_crop_padd_keypoints = face_padd(image, result["face_bbox"], result["face_keypoints"], padding=0.5) 
            # plot_img3 = plot_landmark(face_crop_padd, face_crop_padd_bbox, face_crop_padd_keypoints)
            # do warp 
            img_face_warped = face_warp(face_crop_padd, face_crop_padd_keypoints, scale_factor=1.3)
            # embed the warp face
            try:
                embedding = await asyncio.wait_for(self.triton_client_manager.infer(model_name=self.face_embeding.name, raw_input=img_face_warped), timeout=3*self.BATCH_TIMEOUT)
            except asyncio.TimeoutError:
                self.logger.error(f"hijab_detection Triton timed out after {3*self.BATCH_TIMEOUT:.2f} seconds")
                continue
            # embedding = await self.triton_client_manager.infer(model_name=self.face_embeding.name, raw_input=img_face_warped)
            # vector search
            result["face_embeding"] = embedding
            result["unique_id"], result["unique_cosine"], result["similar_to"] = self.face_bank.query(embedding)
            
            #gender and hijab on head: 
            face_crop_padd, face_crop_padd_bbox, face_crop_padd_keypoints = face_padd(image, result["face_bbox"], result["face_keypoints"], padding=0.1, square=True)
            with self.prometheus['inference_latency'].labels(self.stream_id, "gender_detection").time():
                try:
                    gender_result = await asyncio.wait_for(self.triton_client_manager.infer(model_name=self.headgender.name, raw_input=face_crop_padd), timeout=3*self.BATCH_TIMEOUT)
                except asyncio.TimeoutError:
                    self.logger.error(f"gender_detection Triton timed out after {3*self.BATCH_TIMEOUT:.2f} seconds")
                    continue
                # gender_result = await self.triton_client_manager.infer(model_name=self.headgender.name, raw_input=face_crop_padd)
            self.prometheus['model_count'].labels(self.stream_id, "gender_detection").inc()
            
            result["gender"] = gender_result["farsi_label"]
            result["gender_score"] = gender_result["score"]
            if gender_result["label"] == 'female':
                try:
                    hijab_result = await asyncio.wait_for(self.triton_client_manager.infer(model_name=self.headhijab.name, raw_input=face_crop_padd), timeout=3*self.BATCH_TIMEOUT)
                except asyncio.TimeoutError:
                    self.logger.error(f"hijab_detection Triton timed out after {3*self.BATCH_TIMEOUT:.2f} seconds")
                    continue
                # hijab_result = await self.triton_client_manager.infer(model_name=self.headhijab.name, raw_input=face_crop_padd)
                result["hijab"] = hijab_result["farsi_label"]
                result["hijab_score"] = hijab_result["score"]
                
            
            #push to network io, now resulst contains: bbox, keypoints, face_bbox, face_keypoints face_embeding
            await self.netio_queue.put((image, result))
                    

    
    async def netio_worker(self):
        while True:
            if self.netio_queue.full():
                self.logger.error("netio_queue is full! Exiting.")
                await self.shutdown()
                sys.exit(1)

            batch= []
            try:
                # Wait for the first item (blocks)
                item = await asyncio.wait_for(self.netio_queue.get(), timeout=self.BATCH_TIMEOUT)
                self.prometheus["queue_size"].labels(stream_id=self.stream_id, model='minio_upload').set(self.netio_queue.qsize())
                
                batch.append(item)
            except asyncio.TimeoutError:
                continue  # skip iteration if queue is empty during initial wait

            # Try to gather more items within the timeout
            start_time = asyncio.get_event_loop().time()
            while len(batch) < self.BATCH_MAX_SIZE:
                remaining_time = self.BATCH_TIMEOUT - (asyncio.get_event_loop().time() - start_time)
                if remaining_time <= 0:
                    break

                try:
                    item = await asyncio.wait_for(self.netio_queue.get(), timeout=remaining_time)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break  # no more items in time

            # Unpack batch
            images, results = zip(*batch)
            self.logger.info(f"new batch in netio_pip_line: track_ids: {[result['track_id'] for result in results]}")

            # === 1. Vector search (batch) and ignore low looklikes ===
            embeddings = [r["face_embeding"] for r in results]
            vectors = np.stack(embeddings)
            
            
            # vector search
            with self.prometheus['inference_latency'].labels(self.stream_id, "vector_search").time():
                try:
                    search_results = await asyncio.wait_for(self.vs.do_search(vectors), timeout=3*self.BATCH_TIMEOUT)
                except asyncio.TimeoutError:
                    self.logger.error(f"Search timed out after {3*self.BATCH_TIMEOUT:.2f} seconds")
                    continue
                # search_results = await self.vs.do_search(vectors)
            self.prometheus['model_count'].labels(self.stream_id, "vector_search").inc()

            
            filtered_pairs = []
            for image, result, look_like in zip(images, results, search_results):
                if look_like[0]["score"] < self.fr_looklike_th:
                    self.logger.info(f"ignore instance, low looklike th {look_like[0]['score']:.4f} : track_id={result['track_id']} @ {result['frame_count']}")
                    continue
                result["looklike"] = look_like
                filtered_pairs.append((image, result))

            
            if filtered_pairs:
                images, results = zip(*filtered_pairs)
            else:
                continue
            
                
            # === 2. MinIO uploads: assign links directly ===
            frame_uploaded = set()
            upload_tasks = []

            for image, result in zip(images, results):
                fc = result["frame_count"]

                # --- Upload full frame once per frame count ---
                if fc not in frame_uploaded:
                    async def upload_full_frame(img=image, frame_num=fc, res=result):
                        link = await self.minio.upload_image(image=img, frame_num=frame_num, dir_name="image")
                        res["frame_link"] = link

                    upload_tasks.append(asyncio.create_task(upload_full_frame()))
                    frame_uploaded.add(fc)

                # --- Upload face crop ---
                x1, y1, x2, y2 = result["face_bbox"]
                face_crop = image[y1:y2, x1:x2, ...]
                
                async def upload_face(img=face_crop, frame_num=fc, res=result):
                    res["face_link"] = await self.minio.upload_image(image=img, frame_num=frame_num, dir_name="face")

                upload_tasks.append(asyncio.create_task(upload_face()))

                # --- Upload body crop ---
                x1b, y1b, x2b, y2b = result["bbox"]
                body_crop = image[y1b:y2b, x1b:x2b, ...]
                if body_crop.size == 0:
                    pass


                async def upload_body(img=body_crop, frame_num=fc, res=result):
                    res["body_link"] = await self.minio.upload_image(image=img, frame_num=frame_num, dir_name="body")
                    # if not res["body_link"]:
                    #     pass

                upload_tasks.append(asyncio.create_task(upload_body()))

            # === Run all upload tasks concurrently and safely assign links ===
            
            with self.prometheus['inference_latency'].labels(self.stream_id, "minio_upload").time():
                try:
                    search_results = await asyncio.wait_for(asyncio.gather(*upload_tasks), timeout=3*self.BATCH_TIMEOUT)
                except asyncio.TimeoutError:
                    self.logger.error(f"minio timed out after {3*self.BATCH_TIMEOUT:.2f} seconds")
                    continue
                # await asyncio.gather(*upload_tasks)
            self.prometheus['model_count'].labels(self.stream_id, "minio_upload").inc()
            
            
            
            # now upload profiles here
            with self.prometheus['inference_latency'].labels(self.stream_id, "profile_upload").time():
                try:
                    search_results = await asyncio.wait_for(self.elastic.upload_profiles(results, self.stream_id), timeout=3*self.BATCH_TIMEOUT)
                except asyncio.TimeoutError:
                    self.logger.error(f"profile upload timed out after {3*self.BATCH_TIMEOUT:.2f} seconds")
                    continue
                # await self.elastic.upload_profiles(results, self.stream_id)
            self.prometheus['model_count'].labels(self.stream_id, "profile_upload").inc()

            self.logger.info(f"uploaded well:  track_ids: {[result['track_id'] for result in results]} :  @ {[result['frame_count'] for result in results]}\nface_links={[result['face_link'] for result in results]}\n")


            
                
                        
    
    async def shutdown(self):
        # Stop aiortc MediaPlayer (if used)
        if self.player:
            await self.player.stop()

        # Release OpenCV capture (if used)
        if self.cap:
            self.cap.release()
            self.cap = None

        # Close PyAV container (if used)
        if self.av_container:
            self.av_container.close()
            self.av_container = None
            self.av_stream = None

        self.logger.info("Shutdown completed. Resources released.")
