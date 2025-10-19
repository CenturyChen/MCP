import os
import yaml
import torch
import math
import numpy as np
from info_nce import InfoNCE
import torch.nn.functional as F
import clip
from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader, AugMixAugmenter
import torchvision.transforms as transforms
from PIL import Image
import json

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)

def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def select_confident_samples(prob):
    batch_entropy = -(prob * torch.log(prob + 1e-6)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]  
    init_pred = prob[idx].mean(0, keepdim=True)
    align_pred = int(init_pred[0].argmax())
    aug_loss = -(init_pred[0] * (init_pred[0].clamp_min(1e-8).log())).sum()  # scalar tensor
    return init_pred, align_pred, aug_loss

def loss_negative_positive(v_positive, v_negative):
    cosine_similarity = F.cosine_similarity(v_positive, v_negative, dim=1)

    epsilon = 1e-7
    loss = -torch.log(1 - cosine_similarity + epsilon).mean()
    return loss

def InfoNCELoss(A, B):
    loss = InfoNCE(temperature=0.01, reduction='mean')
    return loss(A, B)

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def clip_classifier(classnames, template,cupl_path, clip_model, setname, arch):
    f = open(cupl_path)
    cupl = json.load(f)
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            if not (setname == 'eurosat' and arch == 'ViT-B/16'):
                texts += cupl[classname]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        img_weights = torch.stack(clip_weights, dim=1).cuda()
    return img_weights

def get_clip_logits(images, clip_model, clip_weights, dataset_name):
    with torch.no_grad():
        if isinstance(images, list):
            images = torch.cat(images, dim=0).cuda()
        else:
            images = images.cuda()
        image = images[:1] 
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
    
        image_features = clip_model.encode_image(images)
        original_image_features = clip_model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features[:, 0, :] ## B(aug_num)*C
        original_image_features /= original_image_features.norm(dim=-1, keepdim=True)
        original_image_features = original_image_features[:, 0, :] # 1*C
        global_img_feat = image_features.clone()
    clip_weights = clip_weights / clip_weights.norm(dim=0, keepdim=True)
    clip_logits = 100. * image_features @ clip_weights
    img_text=clip_logits.clone()
    if dataset_name.lower() in ('a'):
        if image_features.size(0) > 1:
            
            batch_entropy = softmax_entropy(clip_logits)
            selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
            output = clip_logits[selected_idx]
            image_features = image_features[selected_idx].mean(0).unsqueeze(0)
            clip_logits = output.mean(0).unsqueeze(0)

            loss = avg_entropy(output)
            pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
        else:
            loss = softmax_entropy(clip_logits)
            pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])
    else:
        clip_logits = 100. * original_image_features @ clip_weights
        image_features = original_image_features
        loss = softmax_entropy(clip_logits)
        pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

    return image_features, clip_logits, loss, pred, global_img_feat, img_text.softmax(-1)

def get_preprocess():
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=31, augmix=True)

    return aug_preprocess

def get_ood_preprocess():
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    data_transform = AugMixAugmenter(base_transform, preprocess, n_views=31,
                                        augmix=False) ### aug mix not used for ImageNet test set.

    return data_transform

def get_config_file(config_path, dataset_name):
    if dataset_name == "I":
        config_name = "imagenet.yaml"
    elif dataset_name in ["A", "V", "R", "S"]:
        config_name = f"imagenet_{dataset_name.lower()}.yaml"
    else:
        config_name = f"{dataset_name}.yaml"
    
    config_file = os.path.join(config_path, config_name)
    
    with open(config_file, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} was not found.")

    return cfg

ID_to_gptprompts_path={
    'oxford_flowers': 'CuPL_prompts_flowers102.json',
    'dtd': 'CuPL_prompts_dtd.json',
    'oxford_pets': 'CuPL_prompts_oxfordpets.json',
    'stanford_cars': 'CuPL_prompts_stanfordcars.json',
    'ucf101': 'CuPL_prompts_ucf101.json',
    'caltech101': 'CuPL_prompts_caltech101.json',
    'food101': 'CuPL_prompts_food101.json',
    'sun397': 'CuPL_prompts_sun397.json',
    'aircraft': 'CuPL_prompts_fgvcaircraft.json',
    'eurosat': 'CuPL_prompts_eurosat.json',
    'a': 'CuPL_prompts_imagenet.json',
    'i': 'CuPL_prompts_imagenet.json',
    'v': 'CuPL_prompts_imagenet.json',
    's': 'CuPL_prompts_imagenet.json',
    'r': 'CuPL_prompts_imagenet.json'
}

def build_test_data_loader(dataset_name, root_path, preprocess):
    if dataset_name == 'I':
        preprocess = get_ood_preprocess()
        dataset = ImageNet(root_path, preprocess)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=1, num_workers=8, shuffle=True)
    
    elif dataset_name in ['A','V','R','S']:
        preprocess = get_ood_preprocess()
        dataset = build_dataset(f"imagenet-{dataset_name.lower()}", root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)

    elif dataset_name in ['caltech101','dtd','eurosat','aircraft','food101','oxford_flowers','oxford_pets','stanford_cars','sun397','ucf101']:
        preprocess = get_preprocess()
        dataset = build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)
    
    else:
        raise "Dataset is not from the chosen list"
    cupl_file = ID_to_gptprompts_path[dataset_name.lower()]
    cupl_path = os.path.join("gpt3_prompts", cupl_file)
    return test_loader, dataset.classnames, dataset.template, cupl_path