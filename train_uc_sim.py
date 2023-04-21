from torch.utils.data import DataLoader
from data_loader import SimData, sim_collate_fn
from models import LocationBasedGenerator
from utils import show2, compute_iou
from tqdm import tqdm
import numpy as np
import torch
import argparse
import datetime
from torch.utils.tensorboard import SummaryWriter


def eval_f(model_, iter_, name_="1", device_="cuda", debugging=False):
    total_loss = 0
    vis = False
    correct = 0
    ious = []
    if not debugging:
        for idx, train_batch in enumerate(iter_):
            start, default, weight_maps = [tensor.to(device_) for tensor in train_batch[:3]]
            graphs, ob_names, _ = train_batch[3:]
            with torch.no_grad():
                loss, start_pred = model_(start, default, weight_maps)
                pred_sg = model_.return_sg(start, ob_names)

            ious.extend(compute_iou(start_pred, start)[0])
            total_loss += loss.item()
            for i in range(len(graphs)):
                correct += sorted(graphs[i]) == sorted(pred_sg[i])

            if not vis:
                show2([
                    torch.sum(start, dim=1)[:16].cpu(),  # Initial image
                    torch.sum(start_pred, dim=1)[:16].detach().cpu(),  # Reconstructed image
                    start_pred.detach().cpu().view(-1, 3, 128, 128)[:16]+start.cpu().view(-1, 3, 128, 128)[:16],  # Overlap initial and reconstructed states
                    default.cpu().view(-1, 3, 128, 128)[:16],  # Default matrix (u^{def} in the paper)
                    weight_maps.cpu().view(-1, 1, 128, 128)[:16]  # Part of the ground truth to be used for loss computation (parts that match the goal locations)
                ], "figures/test%s.png" % name_, 4)
                vis = True
    else:
        count_ = 0
        for idx, train_batch in enumerate(iter_):
            start, default, weight_maps = [tensor.to(device_) for tensor in train_batch[:3]]
            graphs, ob_names, im_names = train_batch[3:]
            with torch.no_grad():
                loss, start_pred = model_(start, default, weight_maps)
                pred_sg = model_.return_sg(start, ob_names)

            ious.extend(compute_iou(start_pred, start)[0])
            total_loss += loss.item()
            for i in range(len(graphs)):
                res = sorted(graphs[i]) == sorted(pred_sg[i])
                correct += res
                if res == 0:
                    count_ += 1
                    if count_ <= 50:
                        print(count_, im_names[i], sorted(graphs[i]), sorted(pred_sg[i]), "\n")
                        show2([
                            torch.sum(start[i], dim=0).unsqueeze(0).cpu(),
                            torch.sum(start_pred[i], dim=0).unsqueeze(0).cpu(),
                            start[i].cpu(),
                            start_pred[i].cpu(),
                            start[i].cpu() + start_pred[i].cpu(),
                            default[i].cpu(),
                            weight_maps[i].cpu()
                        ], "figures/debug-%d.png" % count_, 4)
    return total_loss, correct, np.mean(ious)


def train(args):
    print("Using", args.device)
    now = datetime.datetime.now()
    writer = SummaryWriter("logs/sim" + now.strftime("%Y%m%d-%H%M%S") + "/")

    nb_epochs = args.num_epochs
    device = args.device

    train_data = SimData(root_dir=args.train_dir, nb_samples=args.num_samples, train_size=1.0)
    train_iterator = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=sim_collate_fn)

    val_data = SimData(train=False, root_dir=args.eval_dir, nb_samples=args.num_samples, train_size=0.0, save_data=args.save_data)
    val_iterator = DataLoader(val_data, batch_size=16, shuffle=False, collate_fn=sim_collate_fn)

    model = LocationBasedGenerator()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epc in range(nb_epochs):
        print(f"Epoch {epc}")
        model.train()
        for idx, train_batch in enumerate(tqdm(train_iterator)):
            optimizer.zero_grad()
            start, default, weight_maps = [tensor.to(device) for tensor in train_batch[:3]]
            loss, start_pred = model(start, default, weight_maps)  # start_pred is the reconstructed image
            loss.backward()

            optimizer.step()
            iou = compute_iou(start_pred, start)[1]
            writer.add_scalar('train/loss', loss.item(), idx + epc * len(train_iterator))
            writer.add_scalar('train/iou', iou, idx + epc * len(train_iterator))


        model.eval()
        loss, acc, iou = eval_f(model, val_iterator, name_=str(epc), device_=device, debugging=epc==nb_epochs-1)
        writer.add_scalar('val/loss', loss/len(val_data), epc)
        writer.add_scalar('val/acc', acc/len(val_data), epc)
        writer.add_scalar('val/iou', iou, epc)

        print(epc, acc/len(val_data), loss/len(val_data), iou)

    torch.save(model.state_dict(), "pre_models/model-sim-%s" % now.strftime("%Y%m%d-%H%M%S"))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", help="train directory",
                        default="data/sim_data/5objs_seg", type=str)
    parser.add_argument("--eval-dir", help="2nd domain evaluation directory",
                        default="data/sim_data/6objs_seg", type=str)
    parser.add_argument("--num-samples", help="how many samples", default=10, type=int)
    parser.add_argument("--save-data", help="whether to save processed data", default=True, type=bool)

    parser.add_argument("--num-epochs", help="how many epochs", default=20, type=int)

    parser.add_argument("--device", help="device: cuda:[device number] or cpu", default=0, type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    train(args)
