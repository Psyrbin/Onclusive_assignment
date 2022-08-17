import torch
from dataset_preparation import prepare_dataset
from models import get_model
from train import train
import sys
from omegaconf import OmegaConf

def main(config_name):
    config = OmegaConf.load(config_name)
    train_dataset, val_dataset, test_dataset = prepare_dataset(config.dataset)
    model = get_model(config.model)
    losses, val_metrics, test_metrics = train(train_dataset, val_dataset, test_dataset, model, config.training)
    # print('Losses: ', losses)
    # print('Validation precisions, recalls, f1s and accuracies: ', val_metrics)
    print('Test precision, recall, f1 and accuracy: ', val_metrics)

    torch.save(model.state_dict(), config.model.save_path)
    print('Final model saved to', config.model.save_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    else:
        raise Exception("Error: Config not specified")

    main(config_name)
