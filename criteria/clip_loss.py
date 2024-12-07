import torch
import clip
from torchvision.transforms import Compose, Normalize

text_prompts = [
    "a blurry image of a {}.",
    "a detailed illustration of a {}.",
    "a pixelated representation of a {}.",
    "a high-quality picture of the {}.",
    "a simple sketch of a {}.",
    "a modern rendering of the {}.",
    "a zoomed-in photo of the {}.",
    "a basic drawing of a {}.",
    "an old photograph of the {}.",
    "an abstract painting of a {}.",
    "a clean image of the {}.",
    "a poor-quality image of the {}.",
    "a lifelike depiction of a {}.",
    "a cartoon version of the {}.",
    "a digital model of a {}.",
]

class CLIPSimilarityLoss(torch.nn.Module):
    def __init__(self):
        super(CLIPSimilarityLoss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.clip_model, self.preprocess_fn = clip.load("ViT-B/32", device=self.device, jit=False)
        self.resize_op = torch.nn.Upsample(scale_factor=7)
        self.pooling_op = torch.nn.AvgPool2d(kernel_size=1024 // 32)
        self.normalize = Compose([
            Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ])

    def forward(self, image_tensor, text_features):
        similarity_score = 1 - self.clip_model(image_tensor, text_features)[0] / 100
        return similarity_score

    def extract_text_features(self, class_label):
        """Encode textual descriptions into feature space."""
        with torch.no_grad():
            descriptions = [template.format(class_label) for template in text_prompts]
            tokenized_prompts = clip.tokenize(descriptions).to(self.device)
            embedded_text = self.clip_model.encode_text(tokenized_prompts)
            embedded_text = embedded_text / embedded_text.norm(dim=-1, keepdim=True)
            averaged_features = embedded_text.mean(dim=0)
        return (averaged_features / averaged_features.norm(dim=-1)).unsqueeze(0).float()

    def extract_image_features(self, image_tensor):
        """Encode images into the feature space."""
        with torch.no_grad():
            image_embeddings = self.clip_model.encode_image(image_tensor)
            normalized_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        return normalized_embeddings.float()
