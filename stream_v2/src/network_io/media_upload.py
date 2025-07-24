from miniopy_async import Minio
from miniopy_async.error import S3Error
from io import BytesIO
from datetime import datetime
from src.config import CONFIG
from turbojpeg import TurboJPEG, TJPF_BGR, TJSAMP_444
import uuid


class MinioImageUploader:
    def __init__(self, config=CONFIG, logger=None):
        self.bucket_name = config.minio.bucket
        self.base_url = config.minio.base_url
        self.stream_id = config.stream_id
        self.object_template = config.minio.object_name
        self.logger = logger
        self.jpeg = TurboJPEG()
        

        self.client = Minio(
            endpoint=config.minio.endpoint,
            access_key=config.minio.access_key.shared_user2,
            secret_key=config.minio.secret_key,
            secure=config.minio.base_url.startswith("https://")
        )

    # @staticmethod
    # def encode_image(self, image, quality=90):
    #     """
    #     Encode a BGR image (OpenCV format) to JPEG using OpenCV.
    #     Returns a BytesIO stream.
    #     """
    #     success, encoded_image = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    #     if not success:
    #         raise ValueError("Failed to encode image with OpenCV")
    #     return BytesIO(encoded_image.tobytes())

    # @staticmethod
    def encode_image(self, image, quality=90):
        """
        Encode a BGR image to JPEG using TurboJPEG.
        Returns a BytesIO stream.
        """
        if image is None:
            raise ValueError("Image is None")

        encoded_image = self.jpeg.encode(
            image,
            quality=quality,
            pixel_format=TJPF_BGR,
            jpeg_subsample=TJSAMP_444  # you can also use TJSAMP_420 for better compression
        )
        return BytesIO(encoded_image)
    
    async def upload_image(self, image, frame_num: int,
                           dir_name: str) -> str | None:
        """
        Encode and upload an image to MinIO.
        Returns the public image URL or None if an error occurred.
        """
        now_time_str = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = uuid.uuid4().hex
        name = f'{frame_num}_{now_time_str}_{unique_id}.jpg'

        object_name = self.object_template.format(
            stream_id=self.stream_id,
            dir_name=dir_name,
            name=name,
        )

        try:
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image[..., ::-1]
            image = self.encode_image(image)
            data_length = len(image.getvalue())
            image.seek(0)

            res = await self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=image,
                length=data_length,
                content_type='image/jpeg',
            )
            image_url = f"{self.base_url}/{res.bucket_name}/{res.object_name}"
            return image_url
        except S3Error as e:
            if self.logger:
                self.logger.exception(f"S3 Error: {e}")
        except Exception as e:
            if self.logger:
                self.logger.exception(f"Upload Error: {e}")
        finally:
            del image  # 🔍 Ensures buffers are freed

        return None
