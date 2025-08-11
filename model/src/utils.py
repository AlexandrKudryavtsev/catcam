import torch
import numpy as np
import os
import time
import copy
import random
from torch.nn.utils import prune
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, fbeta_score


def calculate_metrics(outputs, labels, threshold=0.5, beta=0.5):
    with torch.no_grad():
        if (type(outputs) != np.ndarray):
            outputs = outputs.numpy()
        if (type(labels) != np.ndarray):
            labels = labels.numpy()

        preds = (outputs > threshold)

        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        fbeta = fbeta_score(labels, preds, beta=beta)
        roc_auc = roc_auc_score(labels, outputs)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fbeta_score' : fbeta,
        'roc_auc': roc_auc
    }

def show_metrics(metrics):
  for metric_name in metrics.keys():
    print(f"{metric_name}: {metrics[metric_name]:.4f}")

def save_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)

def load_model(model, model_filepath, device):

    model.load_state_dict(torch.load(model_filepath, map_location=device))

    return model

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def measure_inference_latency(model,
                              device,
                              input_size=(1, 3, 32, 32),
                              num_samples=100,
                              num_warmups=10):

    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave


    model = torch.jit.load(model_filepath, map_location=device)

    return model

def fuse_model(model, model_name):

    fused_model = copy.deepcopy(model)

    model.eval()
    fused_model.eval()

    if (model_name == 'mobilenet_v3_small'):
        torch.quantization.fuse_modules(model.features, [["0.0", "0.1"]], inplace=True)
        
        for name, module in model.features.named_children():
            if isinstance(module, torch.nn.Sequential):
                if len(module) >= 2:
                    if isinstance(module[0], torch.nn.Conv2d) and isinstance(module[1], torch.nn.BatchNorm2d):
                        if len(module) > 2 and isinstance(module[2], torch.nn.ReLU):
                            torch.quantization.fuse_modules(module, [["0", "1", "2"]], inplace=True)
                        else:
                            torch.quantization.fuse_modules(module, [["0", "1"]], inplace=True)

    elif (model_name == 'efficientnet_b0'):
        torch.quantization.fuse_modules(model.features, [["0.0", "0.1"]], inplace=True)
        
        for block in model.features:
            if hasattr(block, "block"):
                if hasattr(block.block[0], "0") and hasattr(block.block[0], "1"):
                    torch.quantization.fuse_modules(block.block[0], [["0", "1"]], inplace=True)
                if hasattr(block.block[2], "0") and hasattr(block.block[2], "1"):
                    torch.quantization.fuse_modules(block.block[2], [["0", "1"]], inplace=True)
                    
    return model

