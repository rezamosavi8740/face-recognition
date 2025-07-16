from miniopy_async import Minio
from miniopy_async.error import S3Error
from io import BytesIO
from datetime import datetime
from src.config import CONFIG
from turbojpeg import TurboJPEG, TJPF_BGR, TJSAMP_444
import uuid
import os


class MinioImageUploader:
    def __init__(self, config=CONFIG, logger=None):
        self.bucket_name = config.minio.bucket
        self.base_url = config.minio.base_url
        self.stream_id = config.stream_id
        self.object_template = config.minio.object_name
        self.logger = logger
        self.jpeg = TurboJPEG()

        # --- NEW ---
        # It is recommended to move this to your CONFIG file
        self.local_storage_path = "/home/user3"

        self.client = Minio(
            endpoint=config.minio.endpoint,
            access_key=config.minio.access_key,
            secret_key=config.minio.secret_key,
            secure=config.minio.base_url.startswith("https://")
        )

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
            jpeg_subsample=TJSAMP_444
        )
        return BytesIO(encoded_image)

    async def upload_image(self, image, frame_num: int,
                           dir_name: str) -> tuple[str | None, str | None]:
        """
        Encode and upload an image to MinIO and save it locally.
        Returns a tuple containing the public image URL and the local file path.
        """
        now_time_str = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = uuid.uuid4().hex
        name = f'{frame_num}_{now_time_str}_{unique_id}.jpg'

        object_name = self.object_template.format(
            stream_id=self.stream_id,
            dir_name=dir_name,
            name=name,
        )

        # --- NEW: Define local path ---
        local_file_path = os.path.join(self.local_storage_path, object_name)
        image_url = None

        try:
            image_data = self.encode_image(image[..., ::-1])
            data_length = len(image_data.getvalue())
            image_data.seek(0)

            # --- NEW: Save the image locally ---
            try:
                # Create the directory if it does not exist
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                with open(local_file_path, 'wb') as f:
                    f.write(image_data.getvalue())
                if self.logger:
                    self.logger.info(f"Successfully saved image to local path: {local_file_path}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to save image locally to {local_file_path}: {e}")
                local_file_path = None  # Set to None if saving failed

            # --- Existing MinIO Upload Logic ---
            image_data.seek(0)  # Reset pointer for MinIO upload
            res = await self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=image_data,
                length=data_length,
                content_type='image/jpeg',
            )
            image_url = f"{self.base_url}/{res.bucket_name}/{res.object_name}"

        except S3Error as e:
            if self.logger:
                self.logger.exception(f"S3 Error: {e}")
        except Exception as e:
            if self.logger:
                self.logger.exception(f"Upload Error: {e}")
        finally:
            del image  # 🔍 Ensures buffers are freed

        return image_url, local_file_path