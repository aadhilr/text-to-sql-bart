import datetime
import os
import torch
import wandb


def create_experiment_folder(model_output_dir, name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    exp = "{}__{}".format(name, timestamp)

    out_path = os.path.join(model_output_dir, exp)
    os.makedirs(out_path, exist_ok=True)

    return exp, out_path


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    print("We use the device: '{}' and {} gpu's.".format(device, n_gpu))

    return device, n_gpu


def save_model(model, model_save_path, model_name="best_model.pt"):
    torch.save(model.state_dict(), os.path.join(model_save_path, model_name))
    # also save the model to "Weights & Biases"
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'best_model.pt'))