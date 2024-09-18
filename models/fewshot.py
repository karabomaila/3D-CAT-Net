import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from models.backbones import Resnet50Encoder


class FewShotSeg(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()

        self.encoder = Resnet50Encoder(opt)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t = Parameter(torch.Tensor([-1.0]), requires_grad=False)
        self.scaler = 10.0
        self.criterion = nn.NLLLoss()
        self.self_attention = SelfAttention(256)
        self.cross_attention = CrossAttention(256)
        self.high_avg_pool = nn.AdaptiveAvgPool1d(256)
        self.conv_fusion = nn.Conv3d(256 + 1, 256, kernel_size=1)

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling
        Args:
            fts: input features, expect shape: 1 x C x Z' x H' x W'
            mask: binary mask, expect shape: 1 x Z x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-3:], mode="trilinear")

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3, 4)) / (
            mask[None, ...].sum(dim=(2, 3, 4)) + 1e-5
        )  # 1 x C, getting back the original values

        return masked_fts

    def generate_prior(self, query_feat, supp_feat, s_y, fts_size):
        bsize, _, z, sp_sz, _ = query_feat.size()[:]
        cosine_eps = 1e-7

        tmp_mask = (s_y == 1).float().unsqueeze(1)

        tmp_mask = F.interpolate(
            tmp_mask,
            size=fts_size,
            mode="trilinear",
            align_corners=True,
        )

        # highlight relevant features
        tmp_supp_feat = supp_feat * tmp_mask

        q = self.high_avg_pool(
            query_feat.flatten(2).transpose(-2, -1)
        )  # [bs, z*h*w, 256]

        s = self.high_avg_pool(
            tmp_supp_feat.flatten(2).transpose(-2, -1)
        )  # [bs, z*h*w, 256]

        tmp_query = q
        tmp_query = tmp_query.contiguous().permute(0, 2, 1)  # [bs, 256, z*h*w]
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

        tmp_supp = s
        tmp_supp = tmp_supp.contiguous()
        tmp_supp = tmp_supp.contiguous()
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

        similarity = torch.bmm(tmp_supp, tmp_query) / (
            torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps
        )

        similarity = similarity.max(1)[0].view(bsize, z * sp_sz * sp_sz)
        # normalise the results
        similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
            similarity.max(1)[0].unsqueeze(1)
            - similarity.min(1)[0].unsqueeze(1)
            + cosine_eps
        )

        corr_query = similarity.view(bsize, 1, z, sp_sz, sp_sz)
        corr_query = F.interpolate(
            corr_query,
            size=fts_size,
            mode="trilinear",
            align_corners=True,
        )
        corr_query_mask = corr_query.unsqueeze(1)

        return corr_query_mask

    def getPrototype(self, fg_fts) -> list[torch.Tensor]:
        """
        Average the features to obtain the prototype
        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        _, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [
            torch.sum(
                torch.cat([tr for tr in way], dim=0),
                dim=0,
                keepdim=True,
            )
            / (n_shots)
            for way in fg_fts
        ]  ## concat all fg_fts [1, 256]

        return fg_prototypes

    def negSim(self, fts, prototype) -> torch.Tensor:
        """
        Calculate the distance between features and prototypes
        Args:
            fts: input features
                expect shape: N x C x Z x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        sim: torch.Tensor = (
            -F.cosine_similarity(fts, prototype[..., None, None, None], dim=1)
            * self.scaler
        )
        return sim

    def forward(
        self, supp_imgs, fore_mask, qry_imgs, train=False, t_loss_scaler=1, n_iters=0
    ):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x Z x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x Z x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x Z x H x W], list of tensors
        """

        n_ways: int = len(supp_imgs)
        self.n_shots: int = len(supp_imgs[0])
        self.n_ways: int = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries: int = len(qry_imgs)
        # for now only one-way, because not every shot has multiple sub-images
        assert self.n_ways == 1
        assert self.n_queries == 1

        n_queries = len(qry_imgs)
        batch_size_q = qry_imgs[0].shape[0]
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-3:]

        fore_mask = torch.stack(
            [torch.stack(way, dim=0) for way in fore_mask], dim=0
        )  # Wa x Sh x B x Z' x H' x W'

        ###### Extract features ######
        imgs_concat: torch.Tensor = torch.cat(
            [torch.cat(way, dim=0) for way in supp_imgs]
            + [
                torch.cat(qry_imgs, dim=0),
            ],
            dim=0,
        )

        img_fts = self.encoder(imgs_concat, low_level=False)

        img_fts = 2 * (img_fts - img_fts.min()) / (img_fts.max() - img_fts.min()) - 1

        fts_size = img_fts.shape[-3:]

        supp_fts = img_fts[: n_ways * self.n_shots * batch_size].view(
            n_ways, self.n_shots, batch_size, -1, *fts_size
        )  # Wa x Sh x B x C x Z' x H' x W'
        qry_fts = img_fts[n_ways * self.n_shots * batch_size :].view(
            n_queries, batch_size_q, -1, *fts_size
        )  # N x B x C x Z x H' x W'

        # Reshape for self attention
        supp_fts_reshaped = supp_fts.view(
            -1, *supp_fts.shape[-4:]
        )  # (Wa*Sh*B) x C x Z' x H' x W'

        qry_fts_reshaped = qry_fts.view(
            -1, *qry_fts.shape[-4:]
        )  # (N*B) x C x Z' x H' x W'

        # Self attention
        supp_fts_reshaped = self.self_attention(supp_fts_reshaped)

        qry_fts_reshaped = self.self_attention(qry_fts_reshaped)

        qry_fts = qry_fts_reshaped.view(
            n_queries, batch_size_q, -1, *fts_size
        )  # N x B x C x Z' x H' x W'

        ###### Generate prior ######
        qry_fts1 = qry_fts.view(
            -1, qry_fts.shape[2], *fts_size
        )  # (N * B) x C x Z' x H' x W'

        supp_fts1 = supp_fts.view(batch_size, -1, *fts_size)  # B x C x Z' x H' x W'

        fore_mask1 = fore_mask[0][0]  # B x Z' x H' x W'

        corr_query_mask = self.generate_prior(
            qry_fts1, supp_fts1, fore_mask1, (4, 32, 32)
        )

        # Reshape corr_query_mask from (N * B) x 1 x Z' x H' x W' to N x B x 1 x Z' x H' x W'
        corr_query_mask = corr_query_mask.view(n_queries, batch_size_q, 1, *fts_size)

        ######## Fusion of prior and query features ##########
        qry_fts = torch.cat(
            [qry_fts, corr_query_mask], dim=2
        )  # N x B x (C + 1) x Z x H' x W'

        qry_fts = self.conv_fusion(qry_fts.view(-1, qry_fts.shape[2], *fts_size)).view(
            n_queries, batch_size_q, -1, *fts_size
        )

        supp_fts_reshaped = supp_fts.view(-1, *supp_fts.shape[3:])

        qry_fts_reshaped = qry_fts.view(-1, *qry_fts.shape[2:])

        ########## Pass through CrossAttention #############
        supp_fts_out, qry_fts_out = self.cross_attention(
            supp_fts_reshaped, qry_fts_reshaped
        )

        # Reshape back to original shape
        supp_fts1 = supp_fts_out.view(*supp_fts.shape)
        qry_fts1 = qry_fts_out.view(*qry_fts.shape)

        ###### Compute loss ######
        align_loss = torch.zeros(1).to(self.device)
        outputs = []
        for epi in range(batch_size):
            ###### Extract prototypes ######
            supp_fts_ = [
                [
                    self.getFeatures(
                        supp_fts[way, shot, [epi]], fore_mask[way, shot, [epi]]
                    )
                    for shot in range(self.n_shots)
                ]
                for way in range(n_ways)
            ]

            fg_prototypes: list[torch.Tensor] = self.getPrototype(supp_fts_)

            # the similarity between the query features and prototypes
            anom_s: list[torch.Tensor] = [
                self.negSim(qry_fts[epi], prototype) for prototype in fg_prototypes
            ]

            ###### Get threshold #######
            self.thresh_pred: list[torch.Tensor] = [self.t for _ in range(n_ways)]
            self.t_loss: torch.Tensor = self.t / self.scaler

            ###### Get predictions #######
            pred: torch.Tensor = self.getPred(
                anom_s, self.thresh_pred
            )  # N x Wa x Z x H' x W'

            # using cross-attention features to update the prototypes
            qry_fts1 = [qry_fts1]
            fg_prototypes1 = [fg_prototypes]

            qry_prediction = [
                torch.stack(
                    [
                        self.getPrediction(
                            qry_fts1[n][epi],
                            fg_prototypes1[n][way],
                            self.thresh_pred[way],
                        )
                        for way in range(self.n_ways)
                    ],
                    dim=1,
                )
                for n in range(len(qry_fts1))
            ]  # N x Wa x Z' x H' x W'

            ###### Prototype Refinement  ######
            fg_prototypes_ = []
            if (not train) and n_iters > 0:  # iteratively update the prototypes
                for n in range(len(qry_fts1)):
                    fg_prototypes_.append(
                        self.updatePrototype(
                            qry_fts1[n],
                            fg_prototypes1[n],
                            qry_prediction[n],
                            n_iters,
                            epi,
                        )
                    )

                qry_prediction: list[torch.Tensor] = [
                    torch.stack(
                        [
                            self.getPrediction(
                                qry_fts1[n][epi],
                                fg_prototypes_[n][way],
                                self.thresh_pred[way],
                            )
                            for way in range(self.n_ways)
                        ],
                        dim=1,
                    )
                    for n in range(len(qry_fts1))
                ]  # N x Wa x Z x H' x W'

            pred_ups = [
                F.interpolate(
                    qry_prediction[n],
                    size=img_size,
                    mode="trilinear",
                    align_corners=True,
                )
                for n in range(len(qry_fts1))
            ]

            pred_ups = F.interpolate(
                pred, size=img_size, mode="trilinear", align_corners=True
            )
            pred_ups = torch.cat((1.0 - pred_ups, pred_ups), dim=1)

            outputs.append(pred_ups)

            ###### Prototype alignment loss ######
            if train:
                align_loss_epi = self.alignLoss(
                    qry_fts[:, epi],
                    torch.cat((1.0 - pred, pred), dim=1),
                    supp_fts[:, :, epi],
                    fore_mask[:, :, epi],
                )
                align_loss += align_loss_epi

        output: torch.Tensor = torch.stack(
            outputs, dim=1
        )  # N x B x (1 + Wa) x Z x H x W
        output = output.view(-1, *output.shape[2:])

        return (
            output,
            align_loss / batch_size,
        )  # (t_loss_scaler * self.t_loss)

    def updatePrototype(self, fts, prototype, pred, update_iters, epi) -> torch.Tensor:
        prototype_ = Parameter(torch.stack(prototype, dim=0))

        optimizer = torch.optim.Adam([prototype_], lr=0.01)

        while update_iters > 0:
            with torch.enable_grad():
                pred_mask = torch.sum(pred, dim=-4)

                pred_mask = torch.stack((1.0 - pred_mask, pred_mask), dim=1).argmax(
                    dim=1, keepdim=True
                )

                pred_mask = pred_mask.repeat([*fts.shape[1:-3], 1, 1, 1])

                bg_fts = fts[epi] * (1 - pred_mask)
                fg_fts = torch.zeros_like(fts[epi])

                for way in range(self.n_ways):
                    fg_fts += (
                        prototype_[way]
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                        .repeat(*pred.shape)
                        * pred_mask[way][None, ...]
                    )

                new_fts = bg_fts + fg_fts
                fts_norm = torch.sigmoid(
                    (fts[epi] - fts[epi].min()) / (fts[epi].max() - fts[epi].min())
                )
                new_fts_norm = torch.sigmoid(
                    (new_fts - new_fts.min()) / (new_fts.max() - new_fts.min())
                )
                bce_loss = nn.BCELoss()
                loss = bce_loss(fts_norm, new_fts_norm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.stack(
                [
                    self.getPrediction(fts[epi], prototype_[way], self.thresh_pred[way])
                    for way in range(self.n_ways)
                ],
                dim=1,
            )  # N x Wa x Z' x H' x W'

            update_iters += -1

        return prototype_

    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        pred_mask = pred.argmax(dim=1, keepdim=True)

        binary_masks = [pred_mask == i for i in range(1 + n_ways)]

        skip_ways: list[int] = [
            i for i in range(n_ways) if binary_masks[i + 1].sum() == 0
        ]

        pred_mask = torch.stack(
            binary_masks, dim=1
        ).float()  # N x (1 + Wa) x 1 x Z' x H' x W'

        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3, 4, 5))
        qry_prototypes = qry_prototypes / (
            pred_mask.sum((0, 3, 4, 5)) + 1e-5
        )  # (1 + Wa) x C

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                img_fts = supp_fts[way, [shot]]
                supp_sim = self.negSim(img_fts, qry_prototypes[[way + 1]])

                pred = self.getPred(
                    [supp_sim], [self.thresh_pred[way]]
                )  # N x Wa x Z' x H' x W'

                pred_ups = F.interpolate(
                    pred,
                    size=fore_mask.shape[-3:],
                    mode="trilinear",
                    align_corners=True,
                )
                # background + foreground features
                pred_ups = torch.cat((1.0 - pred_ups, pred_ups), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(
                    fore_mask[way, shot], 255, device=img_fts.device
                )
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += (
                    self.criterion(log_prob, supp_label[None, ...].long())
                    / n_shots
                    / n_ways
                )
        return loss

    def getPred(self, sim, thresh) -> torch.Tensor:
        pred = []

        for s, t in zip(sim, thresh):
            pred.append(1.0 - torch.sigmoid(0.5 * (s - t)))

        p = torch.stack(pred, dim=1)  # N x Wa x Z x H' x W'
        return p

    def getPrediction(self, fts, prototype, thresh) -> torch.Tensor:
        """
        Calculate the distance between features and prototypes
        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim: torch.Tensor = (
            -F.cosine_similarity(fts, prototype[..., None, None, None], dim=1)
            * self.scaler
        )
        pred: torch.Tensor = 1.0 - torch.sigmoid(0.5 * (sim - thresh))
        return pred


class SelfAttention(nn.Module):
    def __init__(self, dim) -> None:
        super(SelfAttention, self).__init__()
        self.query = nn.Conv3d(dim, dim // 8, 1)
        self.key = nn.Conv3d(dim, dim // 8, 1)
        self.value = nn.Conv3d(dim, dim, 1)
        self.softmax = nn.Softmax(dim=-2)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm([256, 4, 32, 32])

    def forward(self, x):
        B, C, Z, H, W = x.shape

        scale = (C // 8) ** -0.5
        # reduce the dimension
        q = (
            self.query(x).view(B, -1, Z * H * W).permute(0, 2, 1) * scale
        )  # B, Z*H*W, C'

        k = self.key(x).view(B, -1, Z * H * W)  # B, C', Z*H*W
        v = self.value(x).view(B, -1, Z * H * W)  # B, C, Z*H*W
        # attention
        attn = self.softmax(torch.bmm(q, k))  # B, Z*H*W, Z*H*W

        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, Z, H, W)  # B, C, Z, H, W
        out = self.mlp(out.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return self.norm(out)


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Conv3d(dim, dim // 8, 1)
        self.key = nn.Conv3d(dim, dim // 8, 1)
        self.value = nn.Conv3d(dim, dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm([256, 4, 32, 32])

    def forward(self, x, y):
        # print(x.shape)
        B, C, Z, H, W = x.shape
        scale = (C // 8) ** -0.5

        # reduce the features dimensions
        qx = (
            self.query(x).view(B, -1, Z * H * W).permute(0, 2, 1) * scale
        )  # B, Z*H*W, C
        ky = self.key(y).view(B, -1, Z * H * W)  # B, C', Z*H*W
        vy = self.value(y).view(B, -1, Z * H * W)  # B, C, Z*H*W

        # attention
        attn = self.softmax(torch.bmm(qx, ky))  # B, Z*H*W, Z*H*W

        outx = torch.bmm(vy, attn.permute(0, 2, 1)).view(B, C, Z, H, W)  # B, C, Z, H, W
        outx = self.mlp(outx.permute(0, 2, 3, 4, 1)).permute(
            0, 4, 1, 2, 3
        )  # Apply MLP and permute back
        outx = self.norm(outx)  # Apply normalization

        qy = (
            self.query(y).view(B, -1, Z * H * W).permute(0, 2, 1) * scale
        )  # B, Z*H*W, C
        kx = self.key(x).view(B, -1, Z * H * W)  # B, C', Z*H*W
        vx = self.value(x).view(B, -1, Z * H * W)  # B, C, Z*H*W

        attn = self.softmax(torch.bmm(qy, kx))  # B, Z*H*W, Z*H*W
        outy = torch.bmm(vx, attn.permute(0, 2, 1)).view(B, C, Z, H, W)  # B, C, Z, H, W
        outy = self.mlp(outy.permute(0, 2, 3, 4, 1)).permute(
            0, 4, 1, 2, 3
        )  # Apply MLP and permute back
        outy = self.norm(outy)  # Apply normalization

        return outx, outy
