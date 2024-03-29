from matplotlib import pyplot as plt
import torch as pt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from pytorch_warp_plane_main import construct_train_ds, loss_func, fit
from pytorch_warp_sphere import warp_imgs_to_sphere


# create model
class WarpModel(nn.Module):
    def __init__(self, dev):
        super().__init__()
        pixel_per_degree = 10
        degree_num = 180
        # theta and phi in (-pi/2, pi/2)
        self.img_data = nn.Parameter(
            pt.zeros(
                [3, degree_num * pixel_per_degree, degree_num * pixel_per_degree],
                dtype=pt.float32,
            ).unsqueeze(0)
            + 0.5
        )
        self.mask = None

        self.warp_func = warp_imgs_to_sphere

        # in right-hand opencv coordinate system
        self.cam1_frame = pt.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=pt.float32
        ).to(dev)
        self.sphere_radius = pt.unsqueeze(pt.tensor([2.5], dtype=pt.float32), -1).to(
            dev
        )

    def forward(self, batched_X):
        cam2_frame = batched_X[0]
        cam2_K = batched_X[1]
        pixel_coords = batched_X[2]
        warped_result = self.warp_func(
            self.img_data,
            pixel_coords,
            self.cam1_frame,
            cam2_frame,
            cam2_K,
            self.sphere_radius,
        )

        if self.mask is None:
            ones = pt.ones_like(self.img_data)
            self.mask = (
                self.warp_func(
                    ones,
                    pixel_coords,
                    self.cam1_frame,
                    cam2_frame,
                    cam2_K,
                    self.sphere_radius,
                )
                != 0
            )

        return warped_result


if __name__ == "__main__":
    # query the available device
    dev = pt.device("cuda") if pt.cuda.is_available() else pt.device("cpu")
    print(f"Using {dev} device")

    train_ds = construct_train_ds(dev)
    batch_size = 1
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # construct model and optimization
    model = WarpModel(dev).to(dev)
    # optimization params
    lr = 1e3
    momentum = 0.9
    epochs = 500
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    fit(epochs, model, loss_func, opt, scheduler, train_dl)

    trained_result = v2.ToDtype(dtype=pt.uint8, scale=True)(
        pt.squeeze(model.img_data.cpu().detach())
    )
    plt.imsave(
        "data/camera1_trained_sphere.png", trained_result.permute(1, 2, 0).numpy()
    )
