import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
from utils import find_num_stacks


class LocationBasedGenerator(nn.Module):
    def __init__(self, output_dim=2, input_channels=3):
        super(LocationBasedGenerator, self).__init__()

        # Spatial transformer localization-network
        resnet = torchvision.models.resnet34(pretrained=True)

        layers = list(resnet.children())

        # remove the last layer
        layers.pop()
        # remove the first layer as we take a 6-channel input
        layers.pop(0)
        layers.insert(0, nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))

        self.main = nn.Sequential(*layers)
        self.final_layer = nn.Linear(512, output_dim)

        # Initialize the weights/bias with identity transformation
        self.final_layer.weight.data.zero_()
        self.final_layer.bias.data.copy_(torch.tensor([0.5, 0.5], dtype=torch.float))

    def transform(self, x_default, alpha):
        x_default = x_default[:, :3, :, :]
        grid = F.affine_grid(alpha, x_default.size())
        x_default = F.grid_sample(x_default, grid)
        return x_default

    def find_alpha(self, x):  # TODO: fix notation to match the paper
        xs = self.main(x)  # pretrained resnet / transfer learned
        if xs.size(0) == 1:
            xs = xs.squeeze().unsqueeze(0)
        else:
            xs = xs.squeeze()
        trans_vec = self.final_layer(xs)  # custom final layer, output represents the position from the bottom left corner
        trans_vec = torch.sigmoid(trans_vec)  # this is the a and b in the alpha matrix in the paper
        alpha = torch.tensor([1, 0, -1.6, 0, 1, 1.6], dtype=torch.float, requires_grad=True).repeat(x.size(0), 1).to(
            x.device)
        alpha[:, 2] = -trans_vec[:, 0] * 1.6  # -1.6 a
        alpha[:, 5] = trans_vec[:, 1] * 1.6  # 1.6 b
        return alpha.view(-1, 2, 3)

    def infer(self, x, x_default):
        alpha = self.find_alpha(x)  # pass in the start image
        x_pred = self.transform(x_default, alpha)  # use alpha and u^{def} to get the final prediction
        return x_pred


    def forward(self, x, x_default, weights, using_weights=True, only_pred=False):
        nb_objects = x.size(1)

        # Pass through model
        x = x.view(-1, 3, 128, 128)
        x_default = x_default.view(-1, 3, 128, 128)
        pred = self.infer(x, x_default)  # result after coordinate transform in figure 2 (?)
        if only_pred:
            pred = pred.view(-1, nb_objects, 3, 128, 128)
            return pred

        # Compute Loss
        pos_loss = nn.functional.mse_loss(pred, x[:, :3, :, :], reduction="none")
        if using_weights:
            weights = weights.view(-1, 128, 128)
            pos_loss = torch.mean(pos_loss, dim=1) * weights.squeeze()
        else:
            pos_loss = torch.mean(pos_loss, dim=1)

        # Not in the paper... Encourage prediction confidence using the hinge loss?
        # Additional penalty for the positive loss being too high
        pos_loss = pos_loss.mean()
        zeros = torch.zeros_like(pos_loss)
        # Confidence threshold. All positive losses below this will not incur additional penalty. neg_loss >= 0 always
        hinge = nn.functional.mse_loss(x_default, x[:, :3, :, :])
        neg_loss = torch.max(zeros,
                             torch.ones_like(pos_loss) * hinge - nn.functional.mse_loss(pred, x_default)).mean()

        # all
        x = x.view(-1, nb_objects, 3, 128, 128)
        pred = pred.view(-1, nb_objects, 3, 128, 128)
        scene_loss = nn.functional.mse_loss(torch.sum(pred, dim=1), torch.sum(x, dim=1))
        return pos_loss + neg_loss + scene_loss, pred
