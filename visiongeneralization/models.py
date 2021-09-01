import os

import gensim

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import numpy as np
import torch
import tensorflow as tf
from PIL import Image as Image
from robustness.imagenet_models import resnet50
from torch.utils import model_zoo
from torchvision import transforms as transforms
from transformers import AutoModelWithLMHead, AutoTokenizer, BertModel

import visiongeneralization.clip as clip
from .bit_model import KNOWN_MODELS as BiT_MODELS

# tf.config.experimental.set_memory_growth(0.75)
for gpu in tf.config.list_physical_devices('GPU'):
    # tf.config.experimental.set_virtual_device_configuration(
    #     gpu,
    #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    tf.config.experimental.set_memory_growth(gpu, True)


class ModelEncapsulation(torch.nn.Module):
    def __init__(self, model, out_dim=None):
        super().__init__()
        self.module = model
        self.out_dim = out_dim
        self.has_text_encoder = False
        self.has_image_encoder = True

    def encode_image(self, images, **kwargs):
        return self.module(images, **kwargs)


class CLIPLanguageModel(ModelEncapsulation):
    def __init__(self, model, out_dim=None):
        super().__init__(model, out_dim)

        self.text_out_dim = model.transformer.width
        self.has_text_encoder = True

    def encode_image(self, image):
        return self.module.encode_image(image)

    def encode_text(self, inputs, device, class_token_position=0):
        return self.module.encode_text(inputs.to(device))


class GPT2Model(torch.nn.Module):
    def __init__(self, version="small"):
        super().__init__()

        if version == "small":
            self.transformer_model = AutoModelWithLMHead.from_pretrained("gpt2")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        elif version == "medium":
            self.transformer_model = AutoModelWithLMHead.from_pretrained("gpt2-medium")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
        elif version == "large":
            self.transformer_model = AutoModelWithLMHead.from_pretrained("gpt2-large")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        else:
            raise ValueError('This version of GPT2 is not supported')

        self.has_text_encoder = True
        self.has_image_encoder = False

    def encode_text(self, inputs, device, class_token_position=0):
        hidden_states = self.transformer_model(**inputs, output_hidden_states=True)['hidden_states']
        return hidden_states[-1][:, class_token_position]


class BERTModel(torch.nn.Module):
    def __init__(self, version="base"):
        super().__init__()

        if version not in ["base", "large"]:
            raise ValueError('This version of Bert is not supported')

        self.transformer_model = BertModel.from_pretrained(f"bert-{version}-uncased")

        self.has_text_encoder = True
        self.has_image_encoder = False

    def encode_text(self, inputs, device, class_token_position=0):
        hidden_states = self.transformer_model(**inputs, output_hidden_states=True)['hidden_states']
        return hidden_states[-1][:, class_token_position + 1]  # +1 for start token


class TSMModel(torch.nn.Module):
    def __init__(self, device, mode="v"):
        super(TSMModel, self).__init__()
        if mode == "visual":
            mode = "v"
        assert mode in ["v", "va", "vat"]
        if mode == "v":
            mode = "before_head"

        self.out_dim = 0
        if mode == "before_head":
            self.out_dim = 2048
        elif mode == "va":
            self.out_dim = 512
        elif mode == "vat":
            self.out_dim = 256

        self.has_text_encoder = False
        self.has_image_encoder = True

        import tensorflow_hub as hub

        self.module = hub.load("https://tfhub.dev/deepmind/mmv/tsm-resnet50/1")
        self.device = device
        self.mode = mode

    def encode_image(self, image):
        with tf.device('/device:GPU:0'):
            input_image = image.permute(0, 2, 3, 1).unsqueeze(1).detach().cpu().numpy()
            input_image = input_image.astype(np.float32)
            result = self.module.signatures['video'](tf.constant(tf.cast(input_image, dtype=tf.float32)))
            return torch.from_numpy(result[self.mode].numpy()).to(self.device)


class SkipGramModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.SG_model = gensim.models.KeyedVectors.load_word2vec_format(
            '/home/romain/W2V_evaluation/GoogleNews-vectors-negative300.bin.gz', binary=True)

        self.has_text_encoder = True
        self.has_image_encoder = False

    def encode_text(self, text, device, class_token_position=0):
        words = [sent.split()[class_token_position + 1] for sent in text]  # +1 for start token
        embedding = [self.SG_model[word] for word in words]
        return torch.tensor(embedding)


class TransformerTokenizer:
    available_models = [
        "bert-base-uncased",
        "bert-large-uncased",
        "gpt2",
        "gpt2-medium",
        "gpt2-large"
    ]

    def __init__(self, model):
        if model not in self.available_models:
            raise ValueError("This Tokenizer is not available.")

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.eos_token = "<|endoftext|>"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, text):
        return self.tokenizer(text, return_tensors="pt", padding=True)


def clip_tokenizer(text):
    return clip.tokenize(text)


def identity_tokenizer(text):
    return text


def get_model(model_name, device, keep_fc=False):
    tokenizer = identity_tokenizer

    if "CLIP" in model_name and model_name.replace("CLIP-", "") in clip_models:
        model, transformation = clip.load(model_name.replace("CLIP-", ""), device=device, jit=False)
        model = CLIPLanguageModel(model, model.visual.output_dim)
        transform = lambda ims, augment: transformation
        tokenizer = clip_tokenizer
    elif model_name == "RN50":
        resnet = resnet50(pretrained=True)
        if not keep_fc:
            resnet.fc = torch.nn.Identity()  # remove last linear layer before softmax function
        model = ModelEncapsulation(resnet, 2048)
        model = model.to(device)
        transform = get_imagenet_transform
    elif model_name == "virtex":
        model = ModelEncapsulation(torch.hub.load("kdexd/virtex", "resnet50", pretrained=True), 2048)
        model.module.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        model.to(device)
        transform = get_imagenet_transform
    elif "geirhos" in model_name and model_name.replace("geirhos-", "") in geirhos_model_urls.keys():
        model = resnet50(pretrained=False)
        checkpoint = model_zoo.load_url(geirhos_model_urls[model_name.replace("geirhos-", "")],
                                        map_location=torch.device('cpu'))
        model = ModelEncapsulation(model, 2048)
        model.load_state_dict(checkpoint["state_dict"])
        if not keep_fc:
            model.module.fc = torch.nn.Identity()  # remove last linear layer before softmax function
        model.to(device)
        transform = get_imagenet_transform

    elif model_name == "ICMLM":
        model = resnet50(pretrained=False)
        checkpoint = model_zoo.load_url("http://download.europe.naverlabs.com//ICMLM/icmlm-attfc_r50_coco_5K.pth",
                                        map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["cnn"], strict=False)
        model = ModelEncapsulation(model, 2048)
        if not keep_fc:
            model.module.fc = torch.nn.Identity()  # remove last linear layer before softmax function
        model.to(device)
        transform = get_imagenet_transform
    elif "TSM" in model_name:
        mode = model_name.split("-")[1]
        model = TSMModel(device, mode)
        transform = lambda x, y: get_imagenet_transform(x, y, False)
    elif "GPV" in model_name:
        model_path = "gpv_sce.pth" if model_name == "GPV-SCE" else "gpv.pth"
        model = resnet50(pretrained=False)
        checkpoint = torch.load(os.path.join(gpv_model_folder, model_path), map_location=torch.device("cpu"))['model']
        checkpoint = {mod_name.replace("detr.backbone.0.body.", ""): mod_param
                      for mod_name, mod_param in checkpoint.items()
                      if "module.detr.backbone.0.body" in mod_name}
        model = ModelEncapsulation(model, 2048)
        if not keep_fc:
            model.module.fc = torch.nn.Identity()  # remove last linear layer before softmax function
        model.load_state_dict(checkpoint)
        model.to(device)
        transform = get_imagenet_transform
    elif "madry" in model_name and model_name.replace("madry-", "") in madry_models:
        model = resnet50(pretrained=False)
        checkpoint = torch.load(os.path.join(madry_model_folder, model_name.replace("madry-", "") + ".pt"),
                                map_location=torch.device('cpu'))['model']
        checkpoint = {mod_name.replace(".model", ""): mod_param for mod_name, mod_param in checkpoint.items() if
                      "module.model" in mod_name}
        model = ModelEncapsulation(model, 2048)
        model.load_state_dict(checkpoint)
        if not keep_fc:
            model.module.fc = torch.nn.Identity()  # remove last linear layer before softmax function
        model.to(device)
        transform = get_imagenet_transform
    elif "BiT" in model_name and model_name in BiT_model_urls:
        model = BiT_MODELS[model_name](use_fc=keep_fc)
        model.load_from(np.load(BiT_model_urls[model_name]))
        model = ModelEncapsulation(model, 2048)
        model.to(device)
        transform = lambda x, y: get_imagenet_transform(x, y, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    elif model_name == "semi-supervised-YFCC100M":
        model = resnet50(pretrained=False)
        checkpoint = torch.load(os.path.join(vissl_model_folder, "converted_vissl_rn50_semi_sup_08389792.torch"),
                                map_location=torch.device('cpu'))['model_state_dict']
        checkpoint = {mod_name.replace("_feature_blocks", "module"): mod_param for mod_name, mod_param in
                      checkpoint.items()}
        model = ModelEncapsulation(model, 2048)
        model.load_state_dict(checkpoint)
        if not keep_fc:
            model.module.fc = torch.nn.Identity()  # remove last linear layer before softmax function
        model.to(device)
        transform = get_imagenet_transform
    elif model_name == "semi-weakly-supervised-instagram":
        model = resnet50(pretrained=False)
        checkpoint = torch.load(os.path.join(vissl_model_folder, "converted_vissl_rn50_semi_weakly_sup_16a12f1b.torch"),
                                map_location=torch.device('cpu'))['model_state_dict']
        checkpoint = {mod_name.replace("_feature_blocks", "module"): mod_param for mod_name, mod_param in
                      checkpoint.items()}
        model = ModelEncapsulation(model, 2048)
        model.load_state_dict(checkpoint)
        if not keep_fc:
            model.module.fc = torch.nn.Identity()  # remove last linear layer before softmax function
        model.to(device)
        transform = get_imagenet_transform
    # Language models
    elif model_name == "BERT":
        model = BERTModel()
        transform = lambda x, y: None
        tokenizer = TransformerTokenizer("bert-base-uncased")
    elif model_name == "BERT-large":
        model = BERTModel("large")
        transform = lambda x, y: None
        tokenizer = TransformerTokenizer("bert-large-uncased")
    elif model_name == "GPT2":
        model = GPT2Model()
        transform = lambda x, y: None
        tokenizer = TransformerTokenizer("gpt2")
    elif model_name == "GPT2-medium":
        model = GPT2Model("medium")
        transform = lambda x, y: None
        tokenizer = TransformerTokenizer("gpt2-medium")
    elif model_name == "GPT2-large":
        model = GPT2Model("large")
        transform = lambda x, y: None
        tokenizer = TransformerTokenizer("gpt2-large")
    elif model_name == "Word2Vec":
        model = SkipGramModel()
        transform = lambda x, y: None
    else:
        raise ValueError(f"{model_name} is not a valid model name.")
    return model, transform, tokenizer


BiT_model_urls = {
    'BiT-M-R50x1': os.path.join(os.getenv("TORCH_HOME", os.path.expanduser("~/.cache/torch")),
                                "checkpoints/BiT-M-R50x1.npz"),
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
gpv_model_folder = os.path.join(os.getenv("TORCH_HOME", os.path.expanduser("~/.cache/torch")), "checkpoints")

imagenet_norm_mean, imagenet_norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_imagenet_transform(img_size=256, augmentation=False, normalize=True, mean=None, std=None):
    mean = imagenet_norm_mean if mean is None else mean
    std = imagenet_norm_std if std is None else std
    img_size = 256  # force full size
    img_size_resize = img_size + 20 if augmentation and img_size > 128 else img_size
    transformations = [
        transforms.Resize(img_size_resize, interpolation=Image.BICUBIC),
        (transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC) if augmentation and img_size > 128
         else transforms.CenterCrop(img_size)),
    ]
    if augmentation:
        transformations.append(transforms.RandomHorizontalFlip())

    transformations.extend([
        lambda image: image.convert("RGB"),
        transforms.ToTensor()
    ])

    if normalize:
        transformations.append(transforms.Normalize(mean, std))

    return transforms.Compose(transformations)
