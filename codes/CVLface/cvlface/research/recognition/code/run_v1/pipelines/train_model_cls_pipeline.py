from .base import BasePipeline
from models.base import BaseModel
import torch
import random
import io
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

def tinyface_degrade(img):
    scale = random.uniform(0.2, 0.5)
    small = img.resize((int(img.width * scale), int(img.height * scale)), Image.BICUBIC)
    resized = small.resize((112, 112), Image.BICUBIC)

    mode = random.choice(["jpeg", "blur", "poisson"])
    if mode == "jpeg":
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=random.randint(5, 40))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    elif mode == "blur":
        return resized.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.8, 1.2)))
    else:
        arr = np.array(resized).astype(np.float32)
        noisy = np.clip(arr + (np.random.poisson(arr) - arr), 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

class TrainModelClsPipeline(BasePipeline):

    def __init__(self,
                 model:BaseModel,
                 classifier:BaseModel,
                 optimizer,
                 lr_scheduler):
        super(TrainModelClsPipeline, self).__init__()

        self.model = model
        self.classifier = classifier
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    @property
    def module_names_list(self):
        return ['model', 'classifier', 'optimizer', 'lr_scheduler']

    def integrity_check(self, dataset):
        dataset_color_space = dataset.color_space
        assert dataset_color_space == self.model.config.color_space
        self.color_space = dataset_color_space
        self.make_train_transform()

    def make_train_transform(self):
        return self.model.make_train_transform()

    def __call__(self, batch):
        if len(batch) == 2:
            inputs, targets  = batch
        elif len(batch) == 4:
            inputs, placeholder, targets, thetas = batch
        elif len(batch) == 7:
            inputs, targets, ldmk1, theta1, sample2, ldmk2, theta2 = batch
            if sample2.ndim != 1:
                inputs = torch.cat([inputs, sample2], dim=0)
                targets = torch.cat([targets, targets], dim=0)
        else:
            raise ValueError('not supported batch format')


        if self.current_epoch >= 2:
            if self.current_epoch < 15:
                p = min(0.15 + 0.05 * (self.current_epoch - 2), 0.75)
            else:
                p = 0.75

            batch_size = inputs.size(0)
            num_noisy = int(p * batch_size)
            noisy_idxs = torch.randperm(batch_size)[:num_noisy]

            degraded_inputs = []
            for idx in range(batch_size):
                img_pil = to_pil(inputs[idx].cpu())
                if idx in noisy_idxs:
                    img_pil = tinyface_degrade(img_pil)
                degraded_inputs.append(to_tensor(img_pil))

            inputs = torch.stack(degraded_inputs).to(inputs.device)

        feats = self.model(inputs)
        loss = self.classifier(feats, targets.to(self.classifier.device))
        return loss

    def train(self):
        if not self.model.config.freeze:
            self.model.train()
        if not self.classifier.config.freeze:
            self.classifier.train()

    def eval(self):
        self.model.eval()
        self.classifier.eval()