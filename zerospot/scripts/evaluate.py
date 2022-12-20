import torch

from zerospot.data.zero_shot_data import ZeroShotDataClass
from zerospot.models.zero_shot_model import ZeroShotModel
from pathlib import Path
from torch import nn
import tqdm
from torch.utils.data import DataLoader


def zeroshot_evaluate_model(model, checkpoint_path, test_ds_path):
    ckpt_path = checkpoint_path
    state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
    state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    test_df = ZeroShotDataClass(test_ds_path, False)
    valid_dataloader = DataLoader(test_df, batch_size=16, shuffle=True, collate_fn=test_df.collate_fn,
                                  num_workers=0)
    criterion = nn.BCEWithLogitsLoss()
    correct = 0
    total = 0
    running_loss = 0
    eval_losses = []
    eval_accu = []
    for batch in tqdm.tqdm(valid_dataloader):
        spg, tokens_ids_tensor, tokens_lenients_tensor, label = batch

        spg = spg.unsqueeze(1)
        label_pred = model(tokens_ids_tensor, tokens_lenients_tensor, spg)
        loss = criterion(label_pred, label.float())
        running_loss += loss.item()

        label_pred = (torch.sigmoid(label_pred) > 0.5).int()
        total += label.size(0)
        correct += label_pred.eq(label).sum().item()

    test_loss = running_loss / len(valid_dataloader)
    accu = 100. * correct / total

    eval_losses.append(test_loss)
    eval_accu.append(accu)

    print('Test Loss: %.3f | Accuracy: %.3f' % (test_loss, accu))

    return eval_losses, eval_accu


if __name__ == '__main__':
    checkpoint_path = Path(__file__).parent.parent / 'models' / 'ZeroShotModel_epoch=31-val_loss=0.16.ckpt'
    test_ds_path = 'zeroshot_test_speech_commands_.csv'
    model = ZeroShotModel(30, 32, 1)
    zeroshot_evaluate_model(model, checkpoint_path, test_ds_path)
