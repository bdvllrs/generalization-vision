from visiongeneralization.models import get_model

model_names = [
    "TSM-v",
    "TSM-vat",
    "ICMLM",
    # "semi-supervised-YFCC100M",
    # "semi-weakly-supervised-instagram",
    "geirhos-resnet50_trained_on_SIN",
    "geirhos-resnet50_trained_on_SIN_and_IN",
    "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN",
    "madry-imagenet_l2_3_0",
    "madry-imagenet_linf_4",
    "madry-imagenet_linf_8",
    "CLIP-ViT-B/32",
    "CLIP-RN50",
    "virtex",
    "RN50",
    "BiT-M-R50x1",
]

if __name__ == '__main__':
    for k, model_name in enumerate(model_names):

        model, transform, _ = get_model(model_name, "cpu")
