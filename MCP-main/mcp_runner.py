import random
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from util.tools import Summary, AverageMeter, ProgressMeter, accuracy

import clip
from utils import *

DOWNLOAD_ROOT='~/.cache/clip'

def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings of MCP on specific dataset in yaml format.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')
    parser.add_argument('--cen',  default=0.8, type=float, help='center weight')
    parser.add_argument('--tta_steps',  default=1, type=int, help='TTA steps for residue learning')
    parser.add_argument('--res', default='False', type=str, help='MCP or MCP++')
    args = parser.parse_args()

    return args

def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update entropy and negative cache with new features and loss for each predicted class with limited capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]

def update_class_center(center, pseudo_label, text_center, cache_keys, all_classes):
    """Compute updated class center combining text and cached visual features."""
    if pseudo_label in all_classes and cache_keys is not None:
        index = all_classes.index(pseudo_label)
        existing_class_center = cache_keys[:, index]
        new_class_center = center * existing_class_center + (1 - center) * text_center
    else:
        new_class_center = text_center

    return new_class_center

def update_align_cache(align_cache, pred, features_loss, shot_capacity, cen, res_text_feat, all_classes, cache_keys = None):
    """Update align cache with samples closest to their class center (low entropy)."""
    with torch.no_grad():
        text_feat = res_text_feat[:, pred]  # [dim]
        class_center = update_class_center(cen, pred, text_feat, cache_keys, all_classes)
        feature = features_loss[0][:1]               # [1, dim]
        curr_entropy = features_loss[1]  
        feat_dist = torch.norm(feature - class_center)  
        if pred not in align_cache:
            align_cache[pred] = []
        if len(align_cache[pred]) < shot_capacity:
            align_cache[pred].append((feature, curr_entropy))
            align_cache[pred].sort(key=lambda x: float(x[1])) 
            return
        worst_feat, worst_ent = align_cache[pred][-1]
        if curr_entropy < worst_ent:
            worst_dist = torch.norm(worst_feat - class_center)
            if feat_dist < worst_dist:
                align_cache[pred][-1] = (feature, curr_entropy)
                align_cache[pred].sort(key=lambda x: float(x[1]))

def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using entropy, align or negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).to(dtype=clip_weights.dtype, device=clip_weights.device)
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).to(dtype=clip_weights.dtype, device=clip_weights.device)

        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp().to(cache_values.dtype) @ cache_values

        return alpha * cache_logits

def compute_cache_key_logits(image_features, cache_keys, cache_values, alpha, beta):
    affinity = image_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits

def update_cache_joint(cache_memory, cache_keys, ent_cache, align_cache, ent_pred, align_pred):
    """Merge entropy cache and align cache into unified cache memory and update prototypes."""
    total_shot = cache_memory.size(1)  
    update_classes = set()
    if ent_pred is not None:
        update_classes.add(int(ent_pred))
    if align_pred is not None:
        update_classes.add(int(align_pred))

    for cls in update_classes:
        write_idx = 0
        if align_cache and cls in align_cache:
            for (feature, _) in align_cache[cls]:
                if write_idx >= total_shot: break
                if feature.dim() == 2: feature = feature.squeeze(0)
                cache_memory[cls, write_idx, :] = feature
                write_idx += 1
        if ent_cache and cls in ent_cache and write_idx < total_shot:
            for (feature, _) in ent_cache[cls]:
                if write_idx >= total_shot: break
                if feature.dim() == 2: feature = feature.squeeze(0)
                cache_memory[cls, write_idx, :] = feature
                write_idx += 1

        if write_idx < total_shot:
            cache_memory[cls, write_idx:, :].zero_()
        if write_idx > 0:
            new_proto = cache_memory[cls, :write_idx, :].mean(dim=0)  # [feat_dim]
            cache_keys[:, cls] = new_proto

def shrink_cache_keys_and_values(cache_keys):
    """Prune empty cache slots and generate valid cache key/value pairs for active classes."""
    num_classes = cache_keys.size(1)
    nonzero_mask = cache_keys.abs().sum(dim=0) != 0  # [num_classes] -> bool
    selected_idxs = nonzero_mask.nonzero(as_tuple=True)[0]  # [num_cached_classes]
    pos_cache_keys = cache_keys.index_select(1, selected_idxs).contiguous()  # [feat_dim, num_cached_classes]
    cache_values_new = F.one_hot(selected_idxs, num_classes=num_classes).to(dtype=cache_keys.dtype, device=cache_keys.device)  # [num_cached_classes, num_classes]
    all_classes = selected_idxs.tolist()
    return pos_cache_keys, cache_values_new, all_classes

def get_cache_pred(image_features, cache_memory, global_text_feat):
    """Compute image prediction using adaptive similarity between positive caches and global text features."""
    img_feat = image_features[:1].to(dtype=cache_memory.dtype, device=cache_memory.device, non_blocking=True)
    global_text_feat = global_text_feat.to(dtype=cache_memory.dtype, device=cache_memory.device, non_blocking=True)

    cached_image_feat = torch.cat((cache_memory, global_text_feat), dim=1)
    cached_image_feat_KV = cached_image_feat / cached_image_feat.norm(dim=-1, keepdim=True)
    cached_image_feat_KV[cached_image_feat.sum(-1) == 0] = 0

    similarity_score = (img_feat * cached_image_feat_KV).sum(-1)
    similarity_score = torch.exp(-5.5 * (-similarity_score + 1))
    adaptive_image_feat = (cached_image_feat_KV * similarity_score.unsqueeze(-1)).sum(1)
    adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True)
    logits = 100. * adaptive_image_feat @ img_feat.unsqueeze(-1)
    logits = logits[:,:,0]
    return logits.softmax(dim=1)

def align_neg_keys(image_features, pos_classes, neg_cache):
    """Align negative cache features to match current positive class dimensions."""
    with torch.no_grad():
        aligned_neg_keys = []
        for class_idx in pos_classes:
            if class_idx in neg_cache:
                num_items = len(neg_cache[class_idx])
                class_prototype = torch.zeros_like(image_features)  
                for item in neg_cache[class_idx]:
                    class_prototype += item[0] / num_items  
                aligned_neg_keys.append(class_prototype)  
            else:
                aligned_neg_keys.append(torch.zeros_like(image_features))
        aligned_neg_keys = torch.cat(aligned_neg_keys, dim=0).permute(1, 0)
        return aligned_neg_keys

class PositiveCacheResidue(nn.Module):
    def __init__(self, pos_cache_keys):
        super(PositiveCacheResidue, self).__init__()
        self.feat_dim, self.cache_size = pos_cache_keys.shape
        self.residual = nn.Parameter(torch.zeros([self.feat_dim, self.cache_size]).to(dtype=pos_cache_keys.dtype, device=pos_cache_keys.device), requires_grad=True)
        
    def forward(self, x):
        new_pos_cache_keys = x.clone() + self.residual
        new_pos_cache_keys = F.normalize(new_pos_cache_keys, dim=0)
        return new_pos_cache_keys

class TextResidue(nn.Module):
    def __init__(self, clip_weights):
        super(TextResidue, self).__init__()
        self.feat_dim, self.class_num = clip_weights.shape
        self.residual = nn.Parameter(torch.zeros([self.feat_dim, self.class_num]).to(dtype=clip_weights.dtype, device=clip_weights.device), requires_grad=True)
        
    def forward(self, x):
        x = F.normalize(x, dim=0)  
        new_clip_weights = x + self.residual
        new_clip_weights = F.normalize(new_clip_weights, dim=0)
        return new_clip_weights

def run_test_mcp(args,pos_cfg, neg_cfg, lr_cfg, loader, clip_model, clip_weights, dataset_name,classnames):
    with torch.cuda.amp.autocast():
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top1_cache = AverageMeter('AccCache@1', ':6.2f', Summary.AVERAGE)
        top1_pro = AverageMeter('AccPro@1', ':6.2f', Summary.AVERAGE)
        progress = ProgressMeter(len(loader), [top1, top1_cache, top1_pro], prefix='Test: ') 
        pred_vanilla, pred_cache, pred_pro, labels= [], [], [], []
        entro_cache, neg_cache, align_cache = {}, {}, {}
        n_cls=len(classnames)
        
        #Unpack all hyperparameters
        pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['align_shot', 'entropy_shot','alpha', 'beta']}
        if neg_enabled:
            neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}
        
        pos_cache_keys, all_classes = None, []
        cache_memory = torch.zeros((n_cls, pos_params['entropy_shot'] + pos_params['align_shot'], clip_weights.shape[0]), dtype=clip_weights.dtype, device=clip_weights.device)
        cache_keys = torch.zeros((clip_weights.shape[0], n_cls), dtype=clip_weights.dtype, device=clip_weights.device)
        is_res = args.res.lower() == "true"
        clip_weights_global = clip_weights.clone() #(feat_dim, cls_num)
        
        #Test-time adaptation
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            if is_res:
                clip_weights_local = clip_weights_global.clone().detach()
                text_residue = TextResidue(clip_weights_local)
                new_clip_weights = text_residue(clip_weights_local)
        
            image_features, clip_logits, loss, ent_pred, global_img_feat, img_text = get_clip_logits(images ,clip_model, new_clip_weights if is_res else clip_weights, dataset_name)
            
            with torch.no_grad():    
                target, prop_entropy = target.to(clip_weights.device), get_entropy(loss, clip_weights)
                init_pred, align_pred, aug_loss = select_confident_samples(img_text)
            
            if pos_enabled:
                # Update positive caches (entropy and align)
                update_cache(entro_cache, ent_pred, [image_features, loss], pos_params['entropy_shot'])
                update_align_cache(align_cache, align_pred, [global_img_feat, aug_loss], pos_params['align_shot'], args.cen, clip_weights, all_classes, pos_cache_keys)
                with torch.no_grad():
                    update_cache_joint(cache_memory, cache_keys, entro_cache, align_cache, ent_pred, align_pred)
                pos_cache_keys, pos_cache_values, all_classes = shrink_cache_keys_and_values(cache_keys)
            
            # Update negative cache only for uncertain samples
            neg_logits = 100. * image_features @ clip_weights
            if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
                neg_logits += compute_cache_logits(image_features, entro_cache, pos_params['alpha'], pos_params['beta'], clip_weights)
                neg_loss = softmax_entropy(neg_logits)
                neg_entropy = get_entropy(neg_loss,clip_weights)
                neg_preb = int(neg_logits.topk(1, 1, True, True)[1].t()[0])
                if neg_enabled and neg_params['entropy_threshold']['lower'] < neg_entropy < neg_params['entropy_threshold']['upper']:
                    neg_map = neg_logits.softmax(-1)
                    update_cache(neg_cache, neg_preb, [image_features, neg_loss, neg_map], neg_params['shot_capacity'], True)
                elif neg_entropy <= neg_params['entropy_threshold']['lower']:
                    update_cache(entro_cache, neg_preb, [image_features, neg_loss], pos_params['entropy_shot'])
            final_logits = clip_logits.clone()
            
            if is_res:
                pos_cache_residue = PositiveCacheResidue(pos_cache_keys)
                neg_cache_keys = align_neg_keys(image_features, all_classes, neg_cache)
                if args.tta_steps > 0:
                    optimizer = torch.optim.AdamW([
                        {'params': text_residue.parameters(), 'lr': lr_cfg['text'], 'eps': 1e-3, 'weight_decay': 1e-1},
                        {'params': pos_cache_residue.parameters(), 'lr': lr_cfg['image'], 'eps': 1e-3, 'weight_decay': 1e-1}
                        ])
                    for j in range(args.tta_steps):
                        new_clip_weights = text_residue(clip_weights_local)
                        if pos_enabled:
                            new_pos_cache_keys = pos_cache_residue(pos_cache_keys)
                            final_logits += compute_cache_key_logits(image_features, new_pos_cache_keys,pos_cache_values, pos_params['alpha'], pos_params['beta'])
                        if neg_enabled and neg_cache:
                            final_logits -= compute_cache_logits(image_features, neg_cache, neg_params['alpha'], neg_params['beta'], new_clip_weights, (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']))
                        entropy_loss = avg_entropy(final_logits)
                        pos2neg_loss = loss_negative_positive(new_pos_cache_keys.T, neg_cache_keys.T)
                        img2text_loss = InfoNCELoss(new_pos_cache_keys,new_clip_weights[:,all_classes])
                        lamda, gamma = 0.5, 0.2
                        loss = entropy_loss + lamda * img2text_loss + gamma * pos2neg_loss
                            
                        optimizer.zero_grad()
                        if j == args.tta_steps - 1:
                            loss.backward()
                        else:
                            loss.backward(retain_graph=True)
                        optimizer.step()
                pos_cache_residue.eval()
                text_residue.eval()
                with torch.no_grad():
                    new_clip_weights = text_residue(clip_weights_local)
                    new_img_text = 100. * global_img_feat @ new_clip_weights
                    new_img_text = new_img_text.softmax(dim=-1)
                    confi_logits,_,_ = select_confident_samples(new_img_text)
            with torch.no_grad():
                if pos_enabled and (entro_cache or align_cache):
                    if is_res:
                        new_pos_cache_keys = pos_cache_residue(pos_cache_keys)
                        final_logits += compute_cache_key_logits(image_features, new_pos_cache_keys, pos_cache_values, pos_params['alpha'], pos_params['beta'])
                    else:
                        final_logits += compute_cache_key_logits(image_features, pos_cache_keys, pos_cache_values, pos_params['alpha'], pos_params['beta'])
                if neg_enabled and neg_cache:
                    final_logits -= compute_cache_logits(image_features, neg_cache, neg_params['alpha'], neg_params['beta'], clip_weights, (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']))
                final_logits = final_logits.softmax(-1)
                img_pro_pred = final_logits
                if is_res:
                    img_text_pred = confi_logits
                else:
                    img_text_pred = init_pred
                global_text_feat = clip_weights.clone().unsqueeze(1).permute(2, 1, 0).to(clip_weights.device)
                img_global_pred = get_cache_pred(global_img_feat, cache_memory, global_text_feat)
            if is_res:
                fin_loss=avg_entropy(final_logits)
                if get_entropy(fin_loss, clip_weights) < 0.1:
                        # Cumalative Avg
                        num_avg += 1
                        clip_weights_global = clip_weights_global * (num_avg / (num_avg + 1)) + new_clip_weights * (1 / (num_avg + 1))
        
            with torch.no_grad():
                pred_vanilla.append(img_text_pred)
                pred_cache.append(img_global_pred)
                pred_pro.append(img_pro_pred)
                labels.append(target)

                acc1, _ = accuracy(img_text_pred, target, topk=(1, 5))
                acc1_global, _ = accuracy(img_global_pred, target, topk=(1, 5))
                acc1_pro, _ = accuracy(img_pro_pred, target, topk=(1, 5))
                top1.update(acc1[0], 1)
                top1_cache.update(acc1_global[0],1)
                top1_pro.update(acc1_pro[0], 1)
            torch.cuda.empty_cache()
            if i%1000==0:
                progress.display(i)
        with torch.no_grad():
            progress.display_summary()
            pred_vanilla = torch.cat(pred_vanilla, dim=0)
            pred_cache = torch.cat(pred_cache, dim=0)
            pred_pro = torch.cat(pred_pro, dim=0)
            labels = torch.cat(labels, dim=0)
            weight_search = True
            if weight_search:
                beta1_list = [1.0]
                beta2_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
                beta3_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
                print('-' * 20)
                print('Starting searching...')
                best_acc, best_beta2, best_beta3 = 0., 0., 0.
                for beta1 in beta1_list:
                    for beta2 in beta2_list:
                        for beta3 in beta3_list:
                            logits = pred_vanilla * beta1 + pred_cache * beta2 + pred_pro * beta3
                            acc, _ = accuracy(logits, labels, topk=(1, 5))
                            acc = acc.item()
                            if acc > best_acc:
                                print('New best setting, beta1: {:.4f}; beta2: {:.4f}; beta3: {:.4f}; Acc: {:.2f}'.format(beta1, beta2,beta3, acc))
                                best_acc, best_beta1, best_beta2, best_beta3 = acc, beta1, beta2, beta3
                print(f"Searched Acc: {best_acc:.2f} with beta1 {best_beta1:.3f}, dynamic {best_beta2:.3f} and static {best_beta3:.3f}")
    return [best_acc, best_beta1, best_beta2, best_beta3]

def main():
    args = get_arguments()
    config_path = args.config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone, device=device, download_root=DOWNLOAD_ROOT)
    clip_model.eval()

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)

    # Run MCP on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:\n", cfg)
        
        test_loader, classnames, template, cupl_path = build_test_data_loader(dataset_name, args.data_root, preprocess)
        print(f"class number:{len(classnames)}")
        clip_weights = clip_classifier(classnames, template,cupl_path, clip_model, dataset_name, args.backbone)
        clip_weights = clip_weights.half().to(device)

        start_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
        print(f"\n[INFO] Experiment started at {start_timestamp}\n")
        results_temp = run_test_mcp(args, cfg['positive'], cfg['negative'], cfg['learning_rate'], test_loader, clip_model, clip_weights, dataset_name, classnames)
        print("\n=> {} Acc. on testset [{}]: {}".format("MCP++" if args.res.lower() == "true" else "MCP", dataset_name, results_temp[0]))

if __name__ == "__main__":
    main()