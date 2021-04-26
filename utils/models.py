import os

import numpy as np
import torch
from PIL import Image as Image
from robustness.imagenet_models import resnet50
from torch.utils import model_zoo
from torchvision import transforms as transforms
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline

import clip
from bit_model import KNOWN_MODELS as BiT_MODELS


class ModelEncapsulation(torch.nn.Module):
    def __init__(self, model, out_dim=None):
        super().__init__()
        self.module = model
        self.out_dim = out_dim
        self.has_text_encoder = False
        self.has_image_encoder = True

    def encode_image(self, images):
        return self.module(images)


class CLIPLanguageModel(ModelEncapsulation):
    def __init__(self, model, out_dim=None):
        super().__init__(model, out_dim)

        self.has_text_encoder = True

    def encode_image(self, image):
        return self.module.encode_image(image)

    def encode_text(self, text, device, class_token_position=0):
        text = clip.tokenize(text).to(device)
        return self.module.encode_text(text)


class GPT2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.transformer_model = AutoModelWithLMHead.from_pretrained("gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.has_text_encoder = True
        self.has_image_encoder = False

    def encode_text(self, text, device, class_token_position=0):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        hidden_states = self.transformer_model(**inputs, output_hidden_states=True)['hidden_states']
        return hidden_states[class_token_position + 1]  # +1 for start token


class BERTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.transformer_model = pipeline("feature-extraction", "bert-base-uncased")

        self.has_text_encoder = True
        self.has_image_encoder = False

    def encode_text(self, text, device, class_token_position=0):
        embedding = torch.tensor(self.transformer_model(text))
        return torch.tensor(embedding)[:, class_token_position + 1]  # +1 for start token


def get_model(model_name, device):
    if "CLIP" in model_name and model_name.replace("CLIP-", "") in clip_models:
        model, transform = clip.load(model_name.replace("CLIP-", ""), device=device, jit=False)
        model = CLIPLanguageModel(model, model.visual.output_dim)
    elif model_name == "RN50":
        resnet = resnet50(pretrained=True)
        resnet.fc = torch.nn.Identity()  # remove last linear layer before softmax function
        model = ModelEncapsulation(resnet, 2048)
        model = model.to(device)
        transform = imagenet_transform
    elif model_name == "virtex":
        model = ModelEncapsulation(torch.hub.load("kdexd/virtex", "resnet50", pretrained=True), 2048)
        model.module.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        model.to(device)
        transform = imagenet_transform
    elif "geirhos" in model_name and model_name.replace("geirhos-", "") in geirhos_model_urls.keys():
        model = resnet50(pretrained=False)
        checkpoint = model_zoo.load_url(geirhos_model_urls[model_name.replace("geirhos-", "")],
                                        map_location=torch.device('cpu'))
        model = ModelEncapsulation(model, 2048)
        model.load_state_dict(checkpoint["state_dict"])
        model.module.fc = torch.nn.Identity()  # remove last linear layer before softmax function
        model.to(device)
        transform = imagenet_transform
    elif "madry" in model_name and model_name.replace("madry-", "") in madry_models:
        model = resnet50(pretrained=False)
        checkpoint = torch.load(os.path.join(madry_model_folder, model_name.replace("madry-", "") + ".pt"),
                                map_location=torch.device('cpu'))['model']
        checkpoint = {mod_name.replace(".model", ""): mod_param for mod_name, mod_param in checkpoint.items() if
                      "module.model" in mod_name}
        model = ModelEncapsulation(model, 2048)
        model.load_state_dict(checkpoint)
        model.module.fc = torch.nn.Identity()  # remove last linear layer before softmax function
        model.to(device)
        transform = imagenet_transform
    elif "BiT" in model_name and model_name in BiT_model_urls:
        model = BiT_MODELS[model_name]()
        model.load_from(np.load(BiT_model_urls[model_name]))
        model = ModelEncapsulation(model, 2048)
        model.to(device)
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif model_name == "semi-supervised-YFCC100M":
        model = resnet50(pretrained=False)
        checkpoint = torch.load(os.path.join(vissl_model_folder, "converted_vissl_rn50_semi_sup_08389792.torch"),
                                map_location=torch.device('cpu'))['model_state_dict']
        checkpoint = {mod_name.replace("_feature_blocks", "module"): mod_param for mod_name, mod_param in
                      checkpoint.items()}
        model = ModelEncapsulation(model, 2048)
        model.load_state_dict(checkpoint)
        model.module.fc = torch.nn.Identity()  # remove last linear layer before softmax function
        model.to(device)
        transform = imagenet_transform
    elif model_name == "semi-weakly-supervised-instagram":
        model = resnet50(pretrained=False)
        checkpoint = torch.load(os.path.join(vissl_model_folder, "converted_vissl_rn50_semi_weakly_sup_16a12f1b.torch"),
                                map_location=torch.device('cpu'))['model_state_dict']
        checkpoint = {mod_name.replace("_feature_blocks", "module"): mod_param for mod_name, mod_param in
                      checkpoint.items()}
        model = ModelEncapsulation(model, 2048)
        model.load_state_dict(checkpoint)
        model.module.fc = torch.nn.Identity()  # remove last linear layer before softmax function
        model.to(device)
        transform = imagenet_transform
    # Language models
    elif model_name == "BERT":
        model = BERTModel()
        transform = None
    elif model_name == "GPT2":
        model = GPT2Model()
        transform = None
    elif model_name == "Word2Vec":
        # TODO
        raise ValueError(f"{model_name} is not a valid model name.")
    else:
        raise ValueError(f"{model_name} is not a valid model name.")
    return model, transform


BiT_model_urls = {
    'BiT-M-R50x1': os.path.expanduser("~/.cache/torch/checkpoints/BiT-M-R50x1.npz"),
}
clip_models = ["ViT-B/32", "RN50"]
geirhos_model_urls = {
    'resnet50_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
    'resnet50_trained_on_SIN_and_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
    'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
    # 'alexnet_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/0008049cd10f74a944c6d5e90d4639927f8620ae/alexnet_train_60_epochs_lr0.001-b4aa5238.pth.tar',
}
madry_model_folder = os.path.join(os.getenv("TORCH_HOME", os.path.expanduser("~/.cache/torch")), "checkpoints")
vissl_model_folder = os.path.join(os.getenv("TORCH_HOME", os.path.expanduser("~/.cache/torch")), "checkpoints")
madry_models = ["imagenet_l2_3_0", "imagenet_linf_4", "imagenet_linf_8"]

imagenet_norm_mean, imagenet_norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

imagenet_transform = transforms.Compose([
    transforms.Resize(256, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    lambda image: image.convert("RGB"),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_norm_mean, imagenet_norm_std)
])
