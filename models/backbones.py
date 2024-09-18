import torch.nn as nn

from .resnet import generate_model


class Resnet50Encoder(nn.Module):
    """
    FCN-Resnet50 backbone from MedicalNet
    """

    def __init__(self, opt, aux_dim_keep=64, use_aspp=False):
        super().__init__()
        _model = generate_model(opt)

        _model_list = list(_model.children())
        self.aux_dim_keep = aux_dim_keep
        self.backbone = _model  # TODO change to _model_list[0]

        self.localconv = nn.Conv3d(
            2048, 256, kernel_size=1, stride=1, bias=False
        )  # reduce feature map dimension

        self.asppconv = nn.Conv3d(256, 256, kernel_size=1, bias=False)
        _aspp = _model_list[1]
        _conv256 = _model_list[1]
        self.aspp_out = nn.Sequential(*[_aspp, _conv256])
        self.use_aspp = use_aspp

    def forward(self, x_in, low_level):
        """
        Args:
            low_level: whether returning aggregated low-level features in FCN
        """
        fts = self.backbone(x_in)
        if self.use_aspp:
            fts256 = self.aspp_out(fts["out"])
            high_level_fts = fts256
        else:
            fts2048 = fts
            high_level_fts = self.localconv(fts2048)

        if low_level:
            low_level_fts = fts[:, : self.aux_dim_keep]
            return high_level_fts, low_level_fts
        else:
            return high_level_fts
