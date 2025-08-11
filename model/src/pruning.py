import torch
import os
from torch.nn.utils import prune
from train import train_loop, evaluate

def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):    
    num_zeros = 0
    num_elements = 0

    if use_mask:
        if weight and hasattr(module, "weight_mask"):
            mask = module.weight_mask
            num_zeros += torch.sum(mask == 0).item()
            num_elements += mask.nelement()

        if bias and hasattr(module, "bias_mask"):
            mask = module.bias_mask
            num_zeros += torch.sum(mask == 0).item()
            num_elements += mask.nelement()

    else:
        if weight and hasattr(module, "weight"):
            weight_tensor = module.weight
            num_zeros += torch.sum(weight_tensor == 0).item()
            num_elements += weight_tensor.nelement()

        if bias and hasattr(module, "bias") and module.bias is not None:
            bias_tensor = module.bias
            num_zeros += torch.sum(bias_tensor == 0).item()
            num_elements += bias_tensor.nelement()

    sparsity = num_zeros / num_elements if num_elements > 0 else 0.0
    return num_zeros, num_elements, sparsity

def measure_global_sparsity(model, weight=True, bias=False,
                           conv2d_use_mask=False, linear_use_mask=False):
    num_zeros = 0
    num_elements = 0

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            module_zeros, module_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
            num_zeros += module_zeros
            num_elements += module_elements

        elif isinstance(module, torch.nn.Linear):
            module_zeros, module_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=linear_use_mask)
            num_zeros += module_zeros
            num_elements += module_elements

    sparsity = num_zeros / num_elements if num_elements > 0 else 0.0
    return num_zeros, num_elements, sparsity

def remove_parameters(model):

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass

    return model

def iterative_pruning_finetuning(model,
                                train_loader,
                                val_loader,
                                device,
                                learning_rate=1e-3,
                                l1_regularization_strength=0.0,
                                l2_regularization_strength=1e-4,
                                conv2d_prune_amount=0.4,
                                linear_prune_amount=0.2,
                                num_iterations=10,
                                num_epochs_per_iteration=10,
                                early_stopping_patience=5,
                                model_filename_prefix="pruned_model",
                                model_dir="saved_models",
                                grouped_pruning=False):

    os.makedirs(model_dir, exist_ok=True)
    criterion = torch.nn.BCEWithLogitsLoss()

    prune_history = []

    for i in range(num_iterations):
        print(f"\n--- Iteration {i+1}/{num_iterations} ---")

        print("Pruning...")
        if grouped_pruning:
            parameters_to_prune = []
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    parameters_to_prune.append((module, "weight"))
                elif isinstance(module, torch.nn.Linear):
                    parameters_to_prune.append((module, "weight"))

            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=min(conv2d_prune_amount, linear_prune_amount), # !!!! if one of them is equal 0 then there will no pruning
            )
        else:
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module, name="weight", amount=conv2d_prune_amount)
                elif isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name="weight", amount=linear_prune_amount)

        val_metrics = evaluate(model, val_loader, device, criterion)
        num_zeros, num_elements, sparsity = measure_global_sparsity(
            model, weight=True, bias=False, conv2d_use_mask=True, linear_use_mask=True)

        print(f"Validation after pruning: {val_metrics}")
        print(f"Global sparsity: {sparsity:.2%}")

        print("Fine-tuning...")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=l2_regularization_strength
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

        train_history = train_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=num_epochs_per_iteration,
            early_stopping_patience=early_stopping_patience
        )

        prune_history.append(train_history)

        val_metrics = evaluate(model, val_loader, device, criterion)
        print(f"Validation after fine-tuning: {val_metrics}")

        model_filename = f"{model_filename_prefix}_iter_{i+1}.pt"
        model_filepath = os.path.join(model_dir, model_filename)
        torch.save(model.state_dict(), model_filepath)

    return prune_history
