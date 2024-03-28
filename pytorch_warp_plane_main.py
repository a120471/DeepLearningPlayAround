from matplotlib import pyplot as plt
import torch as pt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2

from pytorch_warp_plane import warp_imgs


class WarpDataSet(Dataset):
    def __init__(self, image, cam_frame, cam_K, pixel_coords):
        self.data_list = [(cam_frame, cam_K, pixel_coords)]
        self.target_list = [image]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return (self.data_list[index], self.target_list[index])


def construct_train_ds(dev):
    COS_45 = 0.5 * 2**0.5
    # in right-hand opencv coordinate system
    cam2_frame = pt.tensor(
        [[COS_45, COS_45, 0, 0], [-COS_45, COS_45, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=pt.float32,
    ).to(dev)
    cam2_K = pt.tensor(
        [[3200, 0, 959.5], [0, 3200, 539.5], [0, 0, 1]], dtype=pt.float32
    ).to(dev)
    cam2_img = v2.ToDtype(dtype=pt.float32, scale=True)(
        read_image("data/camera2.jpg")
    ).to(dev)

    img_h, img_w = cam2_img.shape[-2:]
    ys, xs = pt.meshgrid(pt.arange(img_h), pt.arange(img_w))
    pixel_coords = pt.stack([xs, ys], -1).to(dev)

    return WarpDataSet(cam2_img, cam2_frame, cam2_K, pixel_coords)


class WarpModel(nn.Module):
    def __init__(self, img_shape, dev):
        """
        Args:
          img_shape: [C, H, W]
        """
        super().__init__()
        self.img_data = nn.Parameter(
            pt.zeros(img_shape, dtype=pt.float32).unsqueeze(0) + 0.5
        )
        self.mask = None

        self.warp_func = warp_imgs

        # in right-hand opencv coordinate system
        self.cam1_frame = pt.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=pt.float32
        ).to(dev)
        self.cam1_K = pt.tensor(
            [[2400, 0, 959.5], [0, 2400, 539.5], [0, 0, 1]], dtype=pt.float32
        ).to(dev)

        self.plane_param = pt.unsqueeze(
            pt.tensor([[0, 0, -1, 2.5]], dtype=pt.float32), -1
        ).to(dev)

    def forward(self, Batched_X):
        cam2_frame = Batched_X[0]
        cam2_K = Batched_X[1]
        pixel_coords = Batched_X[2]
        warped_result = self.warp_func(
            self.img_data,
            pixel_coords,
            self.cam1_frame,
            self.cam1_K,
            cam2_frame,
            cam2_K,
            self.plane_param,
        )

        # mask is used to ignore the pixels that are not visible in the target image
        if self.mask is None:
            ones = pt.ones_like(self.img_data)
            self.mask = (
                self.warp_func(
                    ones,
                    pixel_coords,
                    self.cam1_frame,
                    self.cam1_K,
                    cam2_frame,
                    cam2_K,
                    self.plane_param,
                )
                != 0
            )

        return warped_result


def loss_func(input, target, mask):
    return pt.mean(pt.square(input - target)[mask])


def fit(epochs, model, loss_func, opt, scheduler, train_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss = loss_func(model(xb), yb, model.mask)
            loss.backward()
            opt.step()
            for p in model.parameters():
                p.data.clamp_(0, 1)
            opt.zero_grad()
        scheduler.step()
        if epoch % 10 == 0:
            print(epoch, loss)


if __name__ == "__main__":
    # query the available device
    dev = pt.device("cuda") if pt.cuda.is_available() else pt.device("cpu")
    print(f"Using {dev} device")

    # construct dataset
    train_ds = construct_train_ds(dev)
    batch_size = 1
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # used to get output size, and comparison visualization
    cam1_img_gt = read_image("data/camera1.jpg")
    # construct model and optimization
    model = WarpModel(cam1_img_gt.shape, dev).to(dev)
    # optimization params
    lr = 1e5
    momentum = 0.9
    epochs = 500
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    fit(epochs, model, loss_func, opt, scheduler, train_dl)

    trained_result = v2.ToDtype(dtype=pt.uint8, scale=True)(
        pt.squeeze(model.img_data.cpu().detach())
    )
    plt.imsave("data/camera1_trained.png", trained_result.permute(1, 2, 0).numpy())

    # # visualize results
    # plt.figure()
    # plt.get_current_fig_manager().window.showMaximized()
    # plt.imshow(cam1_img_gt.permute(1, 2, 0))
    # plt.tight_layout()
    # plt.figure()
    # plt.get_current_fig_manager().window.showMaximized()
    # plt.imshow(trained_result.permute(1, 2, 0))
    # plt.tight_layout()
    # plt.show()
