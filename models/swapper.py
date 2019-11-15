from collections import OrderedDict

import torch
import torch.nn.functional as F


class Swapper:
    """
    Class for feature swapping.

    Parameters
    ---
    patch_size : int
        default patch size. increased depending on map size when applying.
    stride : int
        default stride. increased depending on map size when applying.
    """

    def __init__(self, patch_size: int = 3, stride: int = 1):
        super(Swapper, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.device = torch.device('cpu')

    def __call__(self,
                 map_in: OrderedDict,
                 map_ref: OrderedDict,
                 map_ref_blur: OrderedDict,
                 is_weight: bool = True):
        """
        Feature matching and swapping.
        For the fist, matching process is conducted on relu3_1 layer.
        Next, swapping features on all layers based on the match.

        Parameters
        ---
        map_in : OrderedDict
            VGG output of I^{LR↑}, see also ~/models/vgg.py.
        map_ref : OrderedDict
            VGG output of I^{Ref}
        map_ref_blur : OrderedDict
            VGG output of I^{Ref↓↑}
        is_weight : bool, optional
            whethere weights is output.

        Returns
        ---
        maps : dict of np.array
            swapped feature maps for each layer.
        weights : np.array
            weight maps for each layer if `is_weight`'s True, otherwise `None`.
        max_idx : np.array
            index maps of the most similar patch for each position and layer.
        """

        assert map_in['relu1_1'].shape[2] % 4 == 0
        assert map_in['relu1_1'].shape[3] % 4 == 0

        max_idx, max_val, weights = self.match(map_in, map_ref_blur, is_weight)
        maps = self.swap(map_in, map_ref, max_idx)

        if is_weight:
            weights = weights.to('cpu').numpy()

        return maps, weights, max_idx.to('cpu').numpy()

    def match(self,
              map_in: OrderedDict,
              map_ref_blur: OrderedDict,
              is_weight: bool = True) -> tuple:
        """
        Patch matching between content and condition images.

        Parameters
        ---
        content : torch.Tensor
            The VGG feature map of the content image, shape: (C, H, W)
        patch_condition : torch.Tensor
            The decomposed patches of the condition image,
            shape: (C, patch_size, patch_size, n_patches)

        Returns
        ---
        max_idx : torch.Tensor
            The indices of the most similar patches
        max_val : torch.Tensor
            The pixel value within max_idx.
        """

        content = map_in['relu3_1'].squeeze(0)
        condition = map_ref_blur['relu3_1'].squeeze(0)

        # patch decomposition
        patch_content = self.sample_patches(content)
        patch_condition = self.sample_patches(condition)

        # normalize content and condition
        patch_content /= patch_content.norm(p=2, dim=(0, 1, 2)) + 1e-5
        patch_condition /= patch_condition.norm(p=2, dim=(0, 1, 2)) + 1e-5

        _, H, W = content.shape
        batch_size = int(1024. ** 2 * 512 / (H * W))
        n_patches = patch_condition.shape[-1]

        max_idx, max_val = None, None
        for idx in range(0, n_patches, batch_size):
            batch = patch_condition[..., idx:idx+batch_size]
            corr = F.conv2d(content.unsqueeze(0),
                            batch.permute(3, 0, 1, 2),
                            stride=self.stride)

            max_val_tmp, max_idx_tmp = corr.squeeze(0).max(dim=0)

            if max_idx is None:
                max_idx, max_val = max_idx_tmp, max_val_tmp
            else:
                indices = max_val_tmp > max_val
                max_val[indices] = max_val_tmp[indices]
                max_idx[indices] = max_idx_tmp[indices] + idx

        if is_weight:  # weight calculation
            weight = self.compute_weights(
                patch_content, patch_condition).reshape(max_idx.shape)
        else:
            weight = None

        return max_idx, max_val, weight

    def compute_weights(self,
                        patch_content: torch.Tensor,
                        patch_condition: torch.Tensor):
        """
        Compute weights

        Parameters
        ---
        patch_content : torch.Tensor
            The decomposed patches of the content image,
            shape: (C, patch_size, patch_size, n_patches)
        patch_condition : torch.Tensor
            The decomposed patches of the condition image,
            shape: (C, patch_size, patch_size, n_patches)
        """

        # reshape patches to (C * patch_size ** 2, n_patches)
        content_vec = patch_content.reshape(-1, patch_content.shape[-1])
        style_vec = patch_condition.reshape(-1, patch_condition.shape[-1])

        # compute matmul between content and condition,
        # output shape is (n_patches_content, n_patches_condition)
        corr = torch.matmul(content_vec.transpose(0, 1), style_vec)

        # the best match over condition patches
        weights, _ = torch.max(corr, dim=-1)

        return weights

    def swap(self,
             map_in: OrderedDict,
             map_ref: OrderedDict,
             max_idx: torch.Tensor) -> dict:
        """
        Feature swapping

        Parameter
        ---
        map_in : namedtuple
        map_ref : namedtuple
        max_idx : namedtuple
        """

        swapped_maps = {}
        for idx, layer in enumerate(['relu3_1', 'relu2_1', 'relu1_1']):
            ratio = 2 ** idx
            _patch_size = self.patch_size * ratio
            _stride = self.stride * ratio

            content = map_in[layer].squeeze(0)
            style = map_ref[layer].squeeze(0)
            patches_style = self.sample_patches(style, _patch_size, _stride)

            target_map = torch.zeros_like(content).to(self.device)
            count_map = torch.zeros(target_map.shape[1:]).to(self.device)
            for i in range(max_idx.shape[0]):
                for j in range(max_idx.shape[1]):
                    _i, _j = i * ratio, j * ratio
                    target_map[:, _i:_i+_patch_size, _j:_j+_patch_size]\
                        += patches_style[..., max_idx[i, j]]
                    count_map[_i:_i+_patch_size, _j:_j+_patch_size] += 1
            target_map /= count_map

            assert not torch.isnan(target_map).any()

            swapped_maps.update({layer: target_map.cpu().numpy()})

        return swapped_maps

    def sample_patches(self,
                       inputs: torch.Tensor,
                       patch_size: int = None,
                       stride: int = None) -> torch.Tensor:
        """
        Patch sampler for feature maps.

        Parameters
        ---
        inputs : torch.Tensor
            the input feature maps, shape: (c, h, w).
        patch_size : int, optional
            the spatial size of sampled patches
        stride : int, optional
            the stride of sampling.

        Returns
        ---
        patches : torch.Tensor
            extracted patches, shape: (c, patch_size, patch_size, n_patches).
        """

        if patch_size is None:
            patch_size = self.patch_size
        if stride is None:
            stride = self.stride

        c, h, w = inputs.shape
        patches = inputs.unfold(1, patch_size, stride)\
                        .unfold(2, patch_size, stride)\
                        .reshape(c, -1, patch_size, patch_size)\
                        .permute(0, 2, 3, 1)
        return patches

    def to(self, device):
        self.device = device
        return self
