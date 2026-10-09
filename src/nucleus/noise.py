import torch
import torch.nn.functional as F
import math
import random

class LogUniformNoise:
    def __init__(self, min, max, skip_prob):
        self.min = min
        self.max = max
        self.skip_prob = skip_prob # probability to NOT add noise.

    def __call__(self, tensor: torch.Tensor):
        assert tensor.dim() == 5
        log_scale = torch.rand(tensor.shape[0], 1, 1, 1, 1, device=tensor.device) * (math.log(self.max) - math.log(self.min)) + math.log(self.min)
        scale = log_scale.exp()
        noise = torch.randn_like(tensor) * scale
        skip = (torch.rand(tensor.shape[0], 1, 1, 1, 1, device=tensor.device) < self.skip_prob)
        return tensor + noise * (~skip).to(noise.dtype)


class FrameDropout:
    def __init__(self, p: float = 0.1, active_prob: float = 0.4):
        self.p = p
        self.active_prob = active_prob
        
    def __call__(self, x):
        b, t, _, _, c = x.shape
        active = (torch.rand(b, 1, 1, 1, c, device=x.device) < self.active_prob)
        mask = (torch.rand(b, t, 1, 1, c, device=x.device) > self.p).to(x.dtype)
        return x * torch.where(active, mask, torch.ones_like(mask))
    

class FieldDropout:
    def __init__(self, p: float = 0.1, active_prob: float = 0.4):
        self.p = p
        self.active_prob = active_prob
        
    def __call__(self, x):
        b, _, _, _, c = x.shape
        active = (torch.rand(b, 1, 1, 1, c, device=x.device) < self.active_prob)
        mask = (torch.rand(b, 1, 1, 1, c, device=x.device) > self.p).to(x.dtype)
        return x * torch.where(active, mask, torch.ones_like(mask))

def _disk_structuring_element(radius: int, device: torch.device) -> torch.Tensor:
    r"""
    Boolean disk of the given radius: cells whose Euclidean distance from the
    center is within `radius`. Used as an isotropic structuring element so
    dilation/erosion grows rounded regions rather than squares.
    """
    offsets = torch.arange(-radius, radius + 1, device=device)
    grid_y, grid_x = torch.meshgrid(offsets, offsets, indexing="ij")
    return (grid_y ** 2 + grid_x ** 2) <= radius ** 2


def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    r"""
    Morphological dilation of a boolean mask (..., H, W) by a disk of the given
    radius. Out-of-bounds neighbors are treated as background, so dilation never
    grows past the domain.
    """
    spatial_shape = mask.shape[-2:]
    disk = _disk_structuring_element(radius, mask.device).to(torch.float32)
    kernel = disk.reshape(1, 1, *disk.shape)
    flattened = mask.to(torch.float32).reshape(-1, 1, *spatial_shape)
    overlap = F.conv2d(flattened, kernel, padding=radius)
    return (overlap > 0).reshape(mask.shape)


class PhaseAugmentation:
    r"""
    Base class for perturbations of a binary phase mask (0 - liquid, nonzero -
    vapor). These make the model robust to the phase errors it produces during
    autoregressive rollout. Subclasses implement `_perturb` on a boolean vapor
    mask; this base handles the per-call activation draw and the dtype round-trip.
    """
    def __init__(self, active_prob: float = 0.4):
        self.active_prob = active_prob

    def __call__(self, phase: torch.Tensor) -> torch.Tensor:
        assert phase.dim() == 4, "phase must be (B, T, H, W)"
        if random.random() >= self.active_prob:
            return phase
        vapor = self._perturb(phase != 0)
        return vapor.to(phase.dtype)

    def _perturb(self, vapor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SpuriousBulkNucleation(PhaseAugmentation):
    r"""
    Rare per-cell liquid -> vapor flips far from any bubble, mimicking spurious
    vapor specks the model should learn to erase.
    """
    def __init__(
        self,
        active_prob: float = 0.1,
        bulk_flip_prob: float = 1e-4, # 64 * 64 * 8 * 1e-4 ~= 3 per batch item
        bulk_margin: int = 2,
    ):
        super().__init__(active_prob)
        self.bulk_flip_prob = bulk_flip_prob
        self.bulk_margin = bulk_margin

    def _perturb(self, vapor: torch.Tensor) -> torch.Tensor:
        bulk_liquid = (~vapor) & ~_dilate_mask(vapor, self.bulk_margin)
        flips = bulk_liquid & (torch.rand_like(vapor, dtype=torch.float32) < self.bulk_flip_prob)
        return vapor | flips


class InterfaceJitter(PhaseAugmentation):
    r"""
    Per-cell flips in a band straddling the interface, which roughens its
    position. Bubble interiors are left untouched.
    """
    def __init__(
        self,
        active_prob: float = 0.2,
        interface_flip_prob: float = 0.05,
        interface_radius: int = 1,
    ):
        super().__init__(active_prob)
        self.interface_flip_prob = interface_flip_prob
        self.interface_radius = interface_radius

    def _perturb(self, vapor: torch.Tensor) -> torch.Tensor:
        # cells within interface_radius of BOTH a vapor and a liquid cell, i.e. the
        # band straddling the interface. Deep interiors are excluded.
        band = _dilate_mask(vapor, self.interface_radius) & _dilate_mask(~vapor, self.interface_radius)
        flips = band & (torch.rand_like(vapor, dtype=torch.float32) < self.interface_flip_prob)
        return vapor ^ flips


class BubbleResize(PhaseAugmentation):
    r"""
    Grows or shrinks every bubble by a small random radius, mimicking volume
    errors. Growing and shrinking are equally likely on each application.
    """
    def __init__(
        self,
        active_prob: float = 0.2,
        max_resize_radius: int = 1,
    ):
        super().__init__(active_prob)
        self.max_resize_radius = max_resize_radius

    def _perturb(self, vapor: torch.Tensor) -> torch.Tensor:
        radius = random.randint(1, self.max_resize_radius)
        if random.random() < 0.5:
            return _dilate_mask(vapor, radius)  # grow
        # eroding the vapor region is removing cells within radius of any liquid.
        return vapor & ~_dilate_mask(~vapor, radius)


def _smooth_noise(batch_size: int, height: int, width: int, length_scale_cells: float, device: torch.device) -> torch.Tensor:
    # unit-variance noise varying over roughly length_scale_cells grid cells, shape (B, H, W);
    # defined in cells so the corruption looks the same at any resolution
    coarse_size = max(2, round(max(height, width) / length_scale_cells))
    coarse = torch.randn(batch_size, 1, coarse_size, coarse_size, device=device)
    smooth = F.interpolate(coarse, size=(height, width), mode="bilinear", align_corners=False)[:, 0]
    return smooth / smooth.flatten(1).std(dim=1).clamp(min=1e-6)[:, None, None]


class SDFAugmentation:
    r"""
    Base class for perturbations of a normalized (B, T, H, W, C) context whose channel 0
    is the signed distance field (vapor where the SDF is positive). Each sample is
    perturbed independently with probability `active_prob`; distances are handled in
    physical units, so `cell_size` corresponds to one grid cell.
    """
    def __init__(self, sdf_mean: float, sdf_std: float, domain_size: float, active_prob: float):
        self.sdf_mean = sdf_mean
        self.sdf_std = sdf_std
        self.domain_size = domain_size
        self.active_prob = active_prob

    def __call__(self, context: torch.Tensor) -> torch.Tensor:
        assert context.dim() == 5, "context must be (B, T, H, W, C)"
        active = torch.rand(context.shape[0], device=context.device) < self.active_prob
        if not active.any():
            return context
        perturbed = self._perturb(context, cell_size=self.domain_size / context.shape[3])
        return torch.where(active[:, None, None, None, None], perturbed, context)

    def _raw_sdf(self, context: torch.Tensor) -> torch.Tensor:
        return context[..., 0] * self.sdf_std + self.sdf_mean

    def _with_raw_sdf(self, context: torch.Tensor, raw_sdf: torch.Tensor) -> torch.Tensor:
        return torch.cat([((raw_sdf - self.sdf_mean) / self.sdf_std)[..., None], context[..., 1:]], dim=-1)

    def _perturb(self, context: torch.Tensor, cell_size: float) -> torch.Tensor:
        raise NotImplementedError


class SDFSpuriousSpecks(SDFAugmentation):
    r"""
    Inserts a few tiny spurious bubbles into bulk liquid, as a proper distance field
    (union of the existing SDF with small discs), so the model learns to remove them.
    """
    def __init__(self, sdf_mean, sdf_std, domain_size, active_prob=0.3, max_specks=3,
                 radius_cells=(0.5, 1.5), bulk_margin_cells=2.0):
        super().__init__(sdf_mean, sdf_std, domain_size, active_prob)
        self.max_specks = max_specks
        self.radius_cells = radius_cells
        self.bulk_margin_cells = bulk_margin_cells

    def _perturb(self, context, cell_size):
        batch, _, height, width, _ = context.shape
        raw_sdf = self._raw_sdf(context)
        centers = torch.rand(batch, self.max_specks, 2, device=context.device) * torch.tensor([height, width], device=context.device)
        radii = torch.empty(batch, self.max_specks, device=context.device).uniform_(*self.radius_cells) * cell_size
        # only place specks where the last context frame is liquid well away from any bubble
        row, column = centers[..., 0].long().clamp(max=height - 1), centers[..., 1].long().clamp(max=width - 1)
        center_sdf = raw_sdf[torch.arange(batch, device=context.device)[:, None], -1, row, column]
        radii = torch.where(center_sdf < -self.bulk_margin_cells * cell_size, radii, torch.full_like(radii, -1e6))
        rows = torch.arange(height, device=context.device) + 0.5
        columns = torch.arange(width, device=context.device) + 0.5
        distance = torch.sqrt((rows[None, None, :, None] - centers[..., 0, None, None]) ** 2
                              + (columns[None, None, None, :] - centers[..., 1, None, None]) ** 2) * cell_size
        speck_sdf = (radii[..., None, None] - distance).amax(dim=1)  # (B, H, W)
        return self._with_raw_sdf(context, torch.maximum(raw_sdf, speck_sdf[:, None]))


class SDFBubbleResize(SDFAugmentation):
    r"""
    Adds a smooth spatially varying offset of up to `max_shift_cells` to the SDF, which
    moves every nearby interface by about that distance: bubbles grow in some regions
    and shrink in others, mimicking volume errors.
    """
    def __init__(self, sdf_mean, sdf_std, domain_size, active_prob=0.3, max_shift_cells=1.0, length_scale_cells=16):
        super().__init__(sdf_mean, sdf_std, domain_size, active_prob)
        self.max_shift_cells = max_shift_cells
        self.length_scale_cells = length_scale_cells

    def _perturb(self, context, cell_size):
        batch, _, height, width, _ = context.shape
        shift = _smooth_noise(batch, height, width, self.length_scale_cells, context.device).clamp(-2.0, 2.0) / 2.0
        return self._with_raw_sdf(context, self._raw_sdf(context) + (self.max_shift_cells * cell_size * shift)[:, None])


class SmoothFieldWarp(SDFAugmentation):
    r"""
    Displaces all fields by one smooth random displacement field of up to
    `max_shift_cells`, mimicking interfaces and flow structures that drifted slightly.
    """
    def __init__(self, sdf_mean, sdf_std, domain_size, active_prob=0.3, max_shift_cells=0.75, length_scale_cells=16):
        super().__init__(sdf_mean, sdf_std, domain_size, active_prob)
        self.max_shift_cells = max_shift_cells
        self.length_scale_cells = length_scale_cells

    def _perturb(self, context, cell_size):
        batch, frames, height, width, channels = context.shape
        displacement = torch.stack([_smooth_noise(batch, height, width, self.length_scale_cells, context.device)
                                    for _ in range(2)], dim=-1).clamp(-2.0, 2.0) / 2.0 * self.max_shift_cells
        rows = torch.linspace(-1 + 1 / height, 1 - 1 / height, height, device=context.device)
        columns = torch.linspace(-1 + 1 / width, 1 - 1 / width, width, device=context.device)
        grid_row, grid_column = torch.meshgrid(rows, columns, indexing="ij")
        # grid_sample takes (x, y) = (column, row) in [-1, 1]; one cell is 2 / size
        grid = torch.stack([grid_column + displacement[..., 1] * 2 / width,
                            grid_row + displacement[..., 0] * 2 / height], dim=-1)
        fields = context.permute(0, 1, 4, 2, 3).reshape(batch, frames * channels, height, width)
        warped = F.grid_sample(fields, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return warped.reshape(batch, frames, channels, height, width).permute(0, 1, 3, 4, 2)


class SmoothFieldNoise(SDFAugmentation):
    r"""
    Spatially correlated noise on every field, summed over several length scales, with
    a per-sample log-uniform amplitude. Complements white LogUniformNoise, which only
    teaches the model to ignore pixel-scale errors.
    """
    def __init__(self, sdf_mean, sdf_std, domain_size, active_prob=0.5, min_std=0.005, max_std=0.05,
                 length_scales_cells=(2, 4, 8, 16)):
        super().__init__(sdf_mean, sdf_std, domain_size, active_prob)
        self.min_std, self.max_std = min_std, max_std
        self.length_scales_cells = length_scales_cells

    def _perturb(self, context, cell_size):
        batch, frames, height, width, channels = context.shape
        noise = sum(_smooth_noise(batch * frames * channels, height, width, scale, context.device)
                    for scale in self.length_scales_cells)
        noise = noise / math.sqrt(len(self.length_scales_cells))
        noise = noise.reshape(batch, frames, channels, height, width).permute(0, 1, 3, 4, 2)
        log_std = torch.empty(batch, device=context.device).uniform_(math.log(self.min_std), math.log(self.max_std))
        return context + noise * log_std.exp()[:, None, None, None, None]


def sdf_context_corruptions(sdf_mean: float, sdf_std: float, domain_size: float) -> list:
    shared = dict(sdf_mean=sdf_mean, sdf_std=sdf_std, domain_size=domain_size)
    return [SDFSpuriousSpecks(**shared), SDFBubbleResize(**shared), SmoothFieldWarp(**shared), SmoothFieldNoise(**shared)]
