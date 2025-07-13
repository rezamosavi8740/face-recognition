import cv2
import threading
from collections import deque
import asyncio
import av
import numpy as np



class RTSPStream:
    def __init__(self, rtsp_url, buffer_size=20):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.running = True
        # Start frame capture thread
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    frame = frame[..., ::-1]
                    self.buffer.append(frame)

    async def get_frame(self, timeout=1.0):
        """Retrieve the oldest frame asynchronously with optional timeout."""
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            with self.lock:
                if self.buffer:
                    return self.buffer.popleft()
            await asyncio.sleep(0.001)
        return None  # No frame available in time

    def stop(self):
        self.running = False
        self.cap.release()


    #         options = {
    #             "rtsp_transport": "tcp",
    #             "fflags": "nobuffer",
    #             "flags": "low_delay",
    #             "probesize": "32",
    #             "analyzeduration": "0"
    #         }
    #         options = {
    #             "fflags": "nobuffer",
    #             "flags": "low_delay",
    #             "probesize": "32",
    #             "analyzeduration": "0"
class RTSPStream2:
    def __init__(self, url, buffer_size=20):
        from aiortc.contrib.media import MediaPlayer

        if url.startswith("rtsp://"):
            self.player = MediaPlayer(url, format="rtsp", options={"rtsp_transport": "tcp"})
        elif url.startswith("rtmp://"):
            options = {
                # "fflags": "nobuffer",
                # "flags": "low_delay",
                # "probesize": "32",
                # "analyzeduration": "0"
            }
            self.player = MediaPlayer(url, format="flv", options=options)
        elif url.endswith((".mp4", ".avi", ".mkv")):
            self.player = MediaPlayer(url)
        else:
            raise ValueError(f"Unsupported stream URL format: {url}")

        self.video = self.player.video
        self.buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.running = True

        # Start a thread to run async event loop
        threading.Thread(target=self._run_async_update_loop, daemon=True).start()

    def _run_async_update_loop(self):
        """Runs the async update loop in a dedicated thread's event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._update_loop())

    async def _update_loop(self):
        """Async frame reader that puts frames into the buffer."""
        while self.running:
            try:
                frame = await self.video.recv()
                img = frame.to_ndarray(format="rgb24")

                with self.lock:
                    self.buffer.append(img)
            except Exception as e:
                print(f"[RTSPStream2] Error receiving frame: {e}")
                await asyncio.sleep(0.001)  # backoff on error

    async def get_frame(self, timeout=1.0):
        """Retrieve the oldest frame asynchronously with optional timeout."""
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            with self.lock:
                if self.buffer:
                    return self.buffer.popleft()
            await asyncio.sleep(0.001)
        return None  # No frame available in time

    def stop(self):
        """Stop the reader thread and release the stream."""
        self.running = False
        self.player.stop()


class RTSPStream3:
    def __init__(self, rtsp_url, buffer_size=20):
        self.rtsp_url = rtsp_url
        self.buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.running = True

        # Open PyAV container
        self.container = av.open(rtsp_url, timeout=5)
        # self.container = av.open(
        #     rtsp_url,
        #     timeout=5,
        #     # thread_type="AUTO",
        #     options={
        #         # "fflags": "nobuffer",
        #         # "flags": "low_delay",
        #         # "rtsp_transport": "tcp",  # or "udp"
        #         # "max_delay": "500000"
        #     }
        # )
        self.stream = self.container.streams.video[0]
        # self.stream.thread_type = 'AUTO'
        self.stream.thread_type = "NONE"
        # self.stream.thread_type = "SLICE"

        # Start frame capture thread
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        for packet in self.container.demux(self.stream):
            if not self.running:
                break
            for frame in packet.decode():
                img = frame.to_ndarray(format='rgb24')  # Convert frame to RGB numpy array
                with self.lock:
                    self.buffer.append(img)

    async def get_frame(self, timeout=1.0):
        """Retrieve the oldest frame asynchronously with optional timeout."""
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            with self.lock:
                if self.buffer:
                    return self.buffer.popleft()
            await asyncio.sleep(0.001)
        return None

    def stop(self):
        self.running = False
        self.container.close()




# apt install libgirepository-2.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-4.0
# apt install gstreamer1.0-plugins-bad

# pip3 install pycairo
# pip3 install PyGObject

# import gi
# import numpy as np
# gi.require_version('Gst', '1.0')
# from gi.repository import Gst, GLib
# Gst.init(None)

# conda install -c conda-forge gst-plugins-good
# conda install -c conda-forge gst-plugins-bad
# conda install -c conda-forge gst-plugins-ugly
# conda install -c conda-forge gst-libav

# conda install -c conda-forge pygobject  
# conda install -c conda-forge gstreamer

class RTSPStream4:
    def __init__(self, rtsp_url, buffer_size=20):
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst, GLib
        self.Gst = Gst
        self.GLib = GLib
        self.Gst.init(None)

        self.rtsp_url = rtsp_url
        self.buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.running = True

        # Create GStreamer pipeline
        self.pipeline_str = f"""
            uridecodebin uri={self.rtsp_url} name=src
            src. ! queue ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink emit-signals=true max-buffers=1 drop=true
        """
        self.pipeline = self.Gst.parse_launch(self.pipeline_str)
        self.appsink = self.pipeline.get_by_name("sink")
        self.appsink.connect("new-sample", self._on_new_sample)

        # Start GStreamer loop in separate thread
        self.mainloop = self.GLib.MainLoop()
        self.glib_thread = threading.Thread(target=self._run_gst_loop, daemon=True)
        self.glib_thread.start()

        # Set pipeline to playing
        self.pipeline.set_state(self.Gst.State.PLAYING)

    def _run_gst_loop(self):
        try:
            self.mainloop.run()
        except Exception as e:
            print("GLib loop exited with error:", e)

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if not sample:
            return self.Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps()
        shape = (
            caps.get_structure(0).get_value("height"),
            caps.get_structure(0).get_value("width"),
            3
        )

        success, mapinfo = buf.map(self.Gst.MapFlags.READ)
        if not success:
            return self.Gst.FlowReturn.ERROR

        try:
            frame = np.frombuffer(mapinfo.data, np.uint8).reshape(shape)
            with self.lock:
                self.buffer.append(frame)
        finally:
            buf.unmap(mapinfo)

        return self.Gst.FlowReturn.OK

    async def get_frame(self, timeout=1.0):
        """Asynchronously get the next frame with optional timeout."""
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            with self.lock:
                if self.buffer:
                    return self.buffer.popleft()
            await asyncio.sleep(0.001)
        return None

    def stop(self):
        """Stop the stream and cleanup."""
        self.running = False
        if self.pipeline:
            self.pipeline.set_state(self.Gst.State.NULL)
        if self.mainloop:
            self.mainloop.quit()
        if self.glib_thread.is_alive():
            self.glib_thread.join(timeout=1)
