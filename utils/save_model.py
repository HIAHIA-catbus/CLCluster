import os
import torch


def save_model(model_path, model, optimizer, current_epoch,data_type):
    out = os.path.join(model_path, "model.tar")
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)
