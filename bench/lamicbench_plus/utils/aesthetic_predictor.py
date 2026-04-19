import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol="emb", ycol="avg_rating"):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class AestheticPredictor:
    def __init__(
        self,
        mlp_path="/data0/lmw/mig-bench-src/benchmark/lamic/sac+logos+ava1-l14-linearMSE.pth",
        clip_model_root="/data0/lmw/ckpt/clip-vit-large-patch14",
        clip_model=None,
        clip_processor=None,
    ):
        self.clip_model = (
            CLIPModel.from_pretrained(clip_model_root).to(device="cuda") if clip_model is None else clip_model
        )
        self.clip_processor = (
            CLIPProcessor.from_pretrained(clip_model_root) if clip_processor is None else clip_processor
        )

        self.model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
        s = torch.load(mlp_path)  # load the model you trained previously or the model available in this repo
        self.model.load_state_dict(s)
        self.model.to("cuda")
        self.model.eval()

    def predict(self, img_path):
        pil_image = Image.open(img_path)
        inputs = self.clip_processor(images=pil_image, return_tensors="pt").to("cuda")

        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs).pooler_output

        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        prediction = self.model(torch.from_numpy(im_emb_arr).to("cuda").type(torch.cuda.FloatTensor))
        return prediction.item()
