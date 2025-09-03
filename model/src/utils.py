import torch
import numpy as np
import os
import time
import random
import tensorflow as tf
from torch.nn.utils import prune
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, fbeta_score

def calculate_metrics(logits, labels, threshold=0.5, beta=0.5, apply_sigmoid=True):
    with torch.no_grad():
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        if apply_sigmoid:
            probabilities = 1 / (1 + np.exp(-logits))
        else:
            probabilities = logits
            
        preds = (probabilities > threshold).astype(int)
        
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        fbeta = fbeta_score(labels, preds, beta=beta, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(labels, probabilities)
        except ValueError:
            roc_auc = 0.0
            
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fbeta_score': fbeta,
        'roc_auc': roc_auc,
        'threshold': threshold
    }

def show_metrics(metrics, model_name="Model", beta=0.5):
    print(f"{model_name} metrics: ")
    print("=" * 45)
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"F1-Score:     {metrics['f1_score']:.4f}")
    print(f"F{beta}-Score:    {metrics['fbeta_score']:.4f}")
    print(f"ROC-AUC:      {metrics['roc_auc']:.4f}")
    print(f"Threshold:    {metrics['threshold']:.4f}\n")

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

def inference_tflite_model(tflite_model_filepath, dataloader, a=1.0, b=0):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_filepath)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]['quantization'][0], input_details[0]['quantization'][1]
    output_scale, output_zero_point = output_details[0]['quantization'][0], output_details[0]['quantization'][1]

    logits = []
    labels = []
    
    with torch.no_grad():
        for image, label in dataloader:
            image_nhwc = image.permute(0, 2, 3, 1).numpy().astype(np.float32)

            image_int8 = (image_nhwc / input_scale + input_zero_point).astype(input_details[0]["dtype"])
            
            interpreter.set_tensor(input_details[0]['index'], image_int8)
            interpreter.invoke()
            
            output_int8 = interpreter.get_tensor(output_details[0]['index'])

            output_fp32 = (output_int8.astype(np.float32) - output_zero_point) * output_scale
            correct_output = 1 / (1 + np.exp(a*output_fp32 + b)) #applying calibration

            logits.append(correct_output.flatten())
            labels.extend(label.numpy().flatten())

    tflite_logits = np.array(logits)
    tflite_labels = np.array(labels)

    return tflite_logits, tflite_labels

def measure_pytorch_time(model, input_shape=(1, 3, 224, 224), device='cpu', num_runs=50):
    model.eval()
    model.to(device)
    
    dummy_input = torch.randn(input_shape).to(device)
    scripted_model = torch.jit.script(model)
    
    with torch.no_grad():
        for _ in range(5):
            _ = scripted_model(dummy_input)
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = scripted_model(dummy_input)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time * 1000

def measure_tflite_time(tflite_model, input_shape=(1, 224, 224, 3), num_runs=50):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    
    for _ in range(5):
        interpreter.set_tensor(input_details[0]['index'], 
                             np.random.random(input_shape).astype(np.float32))
        interpreter.invoke()
    
    start_time = time.time()
    for _ in range(num_runs):
        interpreter.set_tensor(input_details[0]['index'], 
                             np.random.random(input_shape).astype(np.float32))
        interpreter.invoke()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time * 1000
