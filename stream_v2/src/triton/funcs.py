import numpy as np
import cv2
from src.config import CONFIG

# class PoseDetection:
#     def __init__(self, conf_threshold=0.15):
#         self.input_size = self.info()["input"][0]["dims"][1:3]
#         self.out_name = self.info()["output"][0]["name"]
#         self.conf_threshold = conf_threshold

#         self._original_shape = None
#         self._scale = None
#         self._pad = None

#     @property
#     def name(self):
#         return self.info()["name"]
#     def info(self,):
#         return  {
#                 "name": "pose_detection",
#                 "input": [
#                     {
#                         "name": "images",
#                         "data_type": "TYPE_FP32",
#                         "format": "FORMAT_NCHW",
#                         "dims": [3, 800, 480]
#                     }
#                 ],
#                 "output": [
#                     {
#                         "name": "output0",
#                         "data_type": "TYPE_FP32",
#                         "dims": [100, 57]
#                     }
#                 ]
#             }
    
    # def preprocess(self, img: np.ndarray) -> np.ndarray:
    #     """
    #     Resize RGB image with YOLO-style letterbox padding (one axis only),
    #     normalize to [0, 1], transpose to CHW, and expand dims to NCHW.
    #     Does NOT use cv2.copyMakeBorder; padding is done with numpy.
    #     """
    #     assert img.ndim == 3 and img.shape[2] == 3, "Input must be an RGB image"

    #     h0, w0 = img.shape[:2]
    #     self._original_shape = (h0, w0)
    #     target_w, target_h = self.input_size

    #     # Compute scaling
    #     scale = min(target_w / w0, target_h / h0)
    #     new_w, new_h = int(w0 * scale), int(h0 * scale)
    #     resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    #     # Compute padding
    #     pad_w = target_w - new_w
    #     pad_h = target_h - new_h
    #     left = pad_w // 2
    #     right = pad_w - left
    #     top = pad_h // 2
    #     bottom = pad_h - top

    #     self._scale = scale
    #     self._pad = (left, top)

    #     # Create blank padded canvas
    #     padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    #     padded[top:top + new_h, left:left + new_w] = resized

    #     # Normalize to [0, 1], convert to CHW, add batch dim
    #     padded = padded.astype(np.float32) / 255.0
    #     chw = padded.transpose(2, 0, 1)  # HWC → CHW
    #     return np.expand_dims(chw, axis=0)  # Shape: (1, 3, H, W)
    
    
   
   
    # def postprocess(self, outputs: dict) -> list[dict]:
    #     """
    #     Decode model outputs, remove padding, rescale to original shape.
    #     Assumes output shape (1, 100, 57): 4 box + 1 conf + 1 class + 51 keypoints.
    #     """
    #     output = outputs[self.out_name]
    #     detections = output[0]  # shape: (100, 57)

    #     scale = self._scale
    #     pad_x, pad_y = self._pad
    #     orig_h, orig_w = self._original_shape

    #     results = []
    #     for det in detections:
    #         conf = det[4]
    #         if conf < self.conf_threshold:
    #             continue

    #         # Bounding box (x1, y1, x2, y2)
    #         x1 = (det[0] - pad_x) / scale
    #         y1 = (det[1] - pad_y) / scale
    #         x2 = (det[2] - pad_x) / scale
    #         y2 = (det[3] - pad_y) / scale
            
    #             # Clamp values to image bounds
    #         x1 = max(0, min(orig_w, x1))
    #         x2 = max(0, min(orig_w, x2))
    #         y1 = max(0, min(orig_h, y1))
    #         y2 = max(0, min(orig_h, y2))

    #         # Validate bbox: must be positive area
    #         if x2 <= x1 or y2 <= y1:
    #             continue  # skip invalid bbox
        
    #         box = [int(x1), int(y1), int(x2), int(y2)]


    #         # Keypoints
    #         keypoints = []
    #         for i in range(17):
    #             kpt_x = (det[6 + i * 3] - pad_x) / scale
    #             kpt_y = (det[6 + i * 3 + 1] - pad_y) / scale
    #             kpt_conf = det[6 + i * 3 + 2]
    #             keypoints.append({"x": int(kpt_x), "y": int(kpt_y), "score": float(kpt_conf)})

    #         results.append({
    #             "bbox": box,
    #             "score": float(conf),
    #             "keypoints": keypoints
    #         })

    #     return results


class PoseDetection:
    def __init__(self, conf_threshold=0.15):
        if CONFIG.small_pose:
            self.model_info = {
            "name": "pose_detection",
            "input": [{
                "name": "images",
                "data_type": "TYPE_FP32",
                "format": "FORMAT_NCHW",
                "dims": [3, 480, 288]
            }],
            "output": [{
                "name": "output0",
                "data_type": "TYPE_FP32",
                "dims": [100, 57]
            }]
        }
        else:
            self.model_info = {
                "name": "pose_detection",
                "input": [{
                    "name": "images",
                    "data_type": "TYPE_FP32",
                    "format": "FORMAT_NCHW",
                    "dims": [3, 800, 480]
                }],
                "output": [{
                    "name": "output0",
                    "data_type": "TYPE_FP32",
                    "dims": [100, 57]
                }]
            }
        self.input_size = self.model_info["input"][0]["dims"][1:3]  # [800, 480]
        self.out_name = self.model_info["output"][0]["name"]
        self.conf_threshold = conf_threshold

        self._original_shape = None  # (h, w)

    @property
    def name(self):
        return self.model_info["name"]

    def info(self):
        return self.model_info

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Resize image to fixed target size without padding.
        Normalize to [0, 1], convert to CHW, and add batch dimension.
        """
        assert img.ndim == 3 and img.shape[2] == 3, "Input must be an RGB image"

        h0, w0 = img.shape[:2]
        target_w, target_h = self.input_size
        self._original_shape = (h0, w0)

        # Resize image directly (no aspect-ratio padding)
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # Normalize
        # resized = resized.astype(np.float32) / 255.0

        # Convert to CHW and add batch dim
        # chw = np.transpose(resized, (2, 0, 1))
        return np.expand_dims(resized, axis=0)

    def postprocess(self, outputs: dict) -> list[dict]:
        """
        Decode model outputs and map back to original image size.
        """
        output = outputs[self.out_name]
        detections = output[0]  # (100, 57)

        orig_h, orig_w = self._original_shape
        target_w, target_h = self.input_size
        scale_x = orig_w / target_w
        scale_y = orig_h / target_h

        results = []
        for det in detections:
            conf = det[4]
            if conf < self.conf_threshold:
                continue

            # Bounding box
            x1 = det[0] * scale_x
            y1 = det[1] * scale_y
            x2 = det[2] * scale_x
            y2 = det[3] * scale_y

            # Clamp + validate
            x1 = max(0, min(orig_w, x1))
            x2 = max(0, min(orig_w, x2))
            y1 = max(0, min(orig_h, y1))
            y2 = max(0, min(orig_h, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            box = [int(x1), int(y1), int(x2), int(y2)]

            # Keypoints
            keypoints = []
            for i in range(17):
                kpt_x = det[6 + i * 3] * scale_x
                kpt_y = det[6 + i * 3 + 1] * scale_y
                kpt_conf = det[6 + i * 3 + 2]
                keypoints.append({
                    "x": int(kpt_x),
                    "y": int(kpt_y),
                    "score": float(kpt_conf)
                })

            results.append({
                "bbox": box,
                "score": float(conf),
                "keypoints": keypoints
            })

        return results

class FaceAllignment:
    def __init__(self):
        self.input_size = self.info()["input"][0]["dims"][1:3]
        self.out_name = [self.info()["output"][i]["name"] for i in range(3)]

        self._original_shape = None
        self._scale = None
        self._pad = None

    @property
    def name(self):
        return self.info()["name"]
    
    
    def info(self,):
        return  {
                    "name": "face_allignment",
                    "input": [
                        {
                            "name": "input",
                            "data_type": "TYPE_FP32",
                            "format": "FORMAT_NCHW",
                            "dims": [3, 160, 160]
                        }
                    ],
                    "output": [
                        {
                            "name": "bbox",
                            "data_type": "TYPE_FP32",
                            "dims": [4]
                        },
                        {
                            "name": "score",
                            "data_type": "TYPE_FP32",
                            "dims": [1]
                        },
                        {
                            "name": "landmarks",
                            "data_type": "TYPE_FP32",
                            "dims": [5, 2]
                        }
                    ]
                }
            
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        note: input img must be RGB, we make it bgr!
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB , use if input is bgr!
        Resize with YOLO-style letterbox padding (one axis only),
        normalize, transpose to CHW, and expand dims.
        """
        h0, w0 = img.shape[:2]
        self._original_shape = (h0, w0)
        target_w, target_h = self.input_size

        # Calculate scale and new dimensions
        scale = min(target_w / w0, target_h / h0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)

        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img[..., ::-1]
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Compute padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        self._scale = scale
        self._pad = (left, top)

        # Letterbox padded image
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Normalize to [-1, 1] and convert to CHW
        padded = (padded.astype(np.float32) / 255.0 - 0.5) / 0.5
        transposed = padded.transpose(2, 0, 1)
        return np.expand_dims(transposed, axis=0)  # shape: (1, 3, H, W)

    def postprocess(self, outputs: dict) -> list[dict]:
        """
        Decode model outputs, remove padding, rescale to original shape.
        Assumes output shape (1, 100, 57): 4 box + 1 conf + 1 class + 51 keypoints.
        """
        results = {}
        for name in self.out_name:
            if name == "bbox":
                det = outputs[name][0]
                input_w, input_h = self.input_size
                scale = self._scale
                pad_x, pad_y = self._pad
                # orig_h, orig_w = self._original_shape
                # for det in detections:
                # Bounding box (x1, y1, x2, y2)
                x1n = np.clip(det[0], 0.0, 1.0)
                y1n = np.clip(det[1], 0.0, 1.0)
                x2n = np.clip(det[2], 0.0, 1.0)
                y2n = np.clip(det[3], 0.0, 1.0)

                # Convert to padded image coordinates
                x1_p = x1n * input_w
                y1_p = y1n * input_h
                x2_p = x2n * input_w
                y2_p = y2n * input_h

                # Remove padding, rescale to original image
                x1 = (x1_p - pad_x) / scale
                y1 = (y1_p - pad_y) / scale
                x2 = (x2_p - pad_x) / scale
                y2 = (y2_p - pad_y) / scale
                box = [int(x1), int(y1), int(x2), int(y2)]
                results["bbox"] = box
            
            elif name == "score":
                conf = float(outputs[name][0])
                # if conf < self.conf_threshold:
                #     return None
                results["score"] = conf
            
            elif name == "landmarks":
                landmarks = outputs[name][0] # [5,2]
                keypoints = []
                for i in range(5):
                    kpt_x = (np.clip(landmarks[i, 0], 0.0, 1.0)* input_w - pad_x) / scale
                    kpt_y = (np.clip(landmarks[i, 1], 0.0, 1.0)* input_h - pad_y) / scale
                    keypoints.append({"x": int(kpt_x), "y": int(kpt_y)})
                results["keypoints"] = keypoints
        
        return results

class FaceEmbeding:
    def __init__(self, ):
        self.input_size = self.info()["input"][0]["dims"][1:3]
        self.out_name = self.info()["output"][0]["name"]
        # self.conf_threshold = conf_threshold

    @property
    def name(self):
        return self.info()["name"]
    def info(self,):
        return {
                    "name": "face_embeding",
                    "input": [
                        {
                            "name": "input",
                            "data_type": "TYPE_FP32",
                            "format": "FORMAT_NCHW",
                            "dims": [3, 112, 112]
                        }
                    ],
                    "output": [
                        {
                            "name": "output",
                            "data_type": "TYPE_FP32",
                            "dims": [512]
                        },
                        {
                            "name": "onnx::Div_1379",
                            "data_type": "TYPE_FP32",
                            "dims": [1]
                        }
                    ]
                }
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """suppose input image is RGB and we make it BGR

        Args:
            img (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        target_w, target_h = self.input_size

        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img[..., ::-1]
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) 
        img = ((img / 255.0) - 0.5) / 0.5
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img
    
    def postprocess(self, outputs):
        output = outputs[self.out_name]
        output = output[0]  # shape: (1, 512)
        return output



class HeadGender:
    def __init__(self,):
        self.input_size = self.info()["input"][0]["dims"][1:3]
        self.out_name = self.info()["output"][0]["name"]


    @property
    def name(self):
        return self.info()["name"]
    def info(self, ):
        return {
                    "name": "head_gender",
                    "input": [
                        {
                            "name": "pixel_values",
                            "data_type": "TYPE_FP32",
                            "dims": [3, 224, 224]
                        }
                    ],
                    "output": [
                        {
                            "name": "logits",
                            "data_type": "TYPE_FP32",
                            "dims": [2]
                        }
                    ]
                }
    def preprocess(self, img):
        """suppose input is RGB, and we do not change it

        Args:
            img (_type_): _description_

        Returns:
            _type_: _description_
        """
        img = cv2.resize(img, (224, 224))                       # (H,W,3)   BGR
        img = img.astype(np.float32) / 255.0                    # [0,1]
        img = np.transpose(img, (2, 0, 1)) 
        img = np.expand_dims(img, axis=0)    
        return img
        
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable soft-max along the last axis."""
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)
    
    def postprocess(self, outputs):
        output = outputs[self.out_name]
        output = output[0]  # shape: (1, 2) => (2,)
        probs = self._softmax(output)
        male_probs   = probs[1]
        female_probs = probs[0]
        if male_probs > female_probs:
            label = "male"
            score = float(male_probs)
            farsi_label = "مرد"
        else:
            label = "female"
            score = float(female_probs)
            farsi_label= "زن"
        
        return {"label":label, "farsi_label":farsi_label, "score":score}



class HeadHijab:
    def __init__(self):
        self.input_size = self.info()["input"][0]["dims"][1:3]
        self.out_name = self.info()["output"][0]["name"]
        self.class_labels = ["woman_hair_covered", "woman_hair_nude", "woman_hair_seminude"]
        self.farsi_names = {"woman_hair_seminude" : 'حجاب ناقص',
                            "woman_hair_nude" : 'بی حجاب',
                            "woman_hair_covered" : 'با حجاب' ,
                            }

    @property
    def name(self):
        return self.info()["name"]
    def info(self,):
        return {
                    "name": "head_hijab",
                    "input": [
                        {
                            "name": "input",
                            "data_type": "TYPE_FP32",
                            "format": "FORMAT_NCHW",
                            "dims": [3, 112, 112]
                        }
                    ],
                    "output": [
                        {
                            "name": "output",
                            "data_type": "TYPE_FP32",
                            "dims": [3]  # number of classes
                        }
                    ]
                }
    def preprocess(self, img):
        """
        input image is RGB
        """
        img = cv2.resize(img, (112, 112))            
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # Shape: (112, 112, 3)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img
    
    def postprocess(self, outputs): 
        output = outputs[self.out_name]
        output = output[0]  # shape: (1, 2) => (2,)
        pred_idx = int(np.argmax(output))
        confidence = float(output[pred_idx])
        pred_label = self.class_labels[pred_idx]
        return {"label":pred_label, "farsi_label":self.farsi_names[pred_label] ,"score":confidence}


if __name__ == "__main__":
#     ali = FaceAllignment()
# #     pose = PoseModelProcessor()
#     img = np.zeros((640, 1024, 3)).astype(np.uint8)
#     out = ali.preprocess(img=img)
#     # out = pose.preprocess(img=img)
#     out_net1 = np.ones((1, 4))
#     out_net2 = np.ones((1, 1))
#     out_net3 = np.ones((1, 5, 2))
#     out_net = {"bbox":out_net1, "score":out_net2, "landmarks":out_net3}
#     last_out = ali.postprocess(outputs = out_net )
#     print()
#     pose = PoseModelProcessor()
    ali = FaceEmbeding()
    ali = FaceEmbeding()
    img = np.zeros((160, 160, 3)).astype(np.uint8)
    out = ali.preprocess(img=img)
    # out = pose.preprocess(img=img)
    out_net1 = np.ones((1, 512))
    out_net = {"output":out_net1}
    last_out = ali.postprocess(outputs = out_net )
    print()