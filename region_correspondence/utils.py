
import torch


def get_reference_grid(grid_size, device=None):
    '''
    grid_size: [depth, height, width] for 3d  
               [height, width] for 2d
    Returns a 3d reference grid of shape (D,H,W,3) where dim=3 is the coordinate vector xyz (<- ijk)
            a 2d reference grid of shape (H,W,2) where dim=2 is the coordinate vector xy (<- ij)
    '''
    if len(grid_size) == 3:
        ref_grid = torch.stack(torch.meshgrid(
            torch.linspace(-1,1,grid_size[0]),
            torch.linspace(-1,1,grid_size[1]),
            torch.linspace(-1,1,grid_size[2]),
            indexing='ij',
            ), dim=3).to(device)[...,[2,1,0]]  # reverse: ijk->xyz
    elif len(grid_size) == 2:
        ref_grid = torch.stack(torch.meshgrid(
            torch.linspace(-1,1,grid_size[0]),
            torch.linspace(-1,1,grid_size[1]),
            indexing='ij',
            ), dim=2).to(device)[...,[1,0]]  # reverse: ij->xy
    else:
        raise ValueError("grid_size must be a tuple of 2 or 3 ints")
    return ref_grid


def sampler(img, sample_grid):
    '''
    img: torch.tensor of shape (C,D0,H0,W0) where C is the number of masks
                               (C,H0,W0) for 2d 
    sample_grid: torch.tensor of shape (D1,H1,W1,3) where dim=3 is the coordinate vector xyz (<- ijk)
                                       (H1,W1,2) for 2d 
    Returns a warped image of shape (C,D1,H1,W1) for 3d
                                    (C,H1,W1) for 2d 
    '''
    warped = torch.nn.functional.grid_sample(
        input=img.unsqueeze(0),
        grid=sample_grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ) 
    return warped.squeeze(0)


def warp_by_ddf(vol, ddf, ref_grid=None):
    '''
    vol: torch.tensor of shape (C,D0,H0,W0) where C is the number of masks
                               (C,H0,W0) for 2d
    ddf: torch.tensor of shape (D1,H1,W1,3) where dim=3 is the displacement vector xyz (<- ijk)
                               (H1,W1,2) for 2d where dim=2 is the displacement vector xy (<- ij)
    ref_grid: optional - torch.tensor of shape (D1,H1,W1,3) where dim=3 is the coordinate vector xyz (<- ijk)
                                               (H1,W1,2) for 2d where dim=2 is the coordinate vector xy (<- ij)
    Returns a warped image of shape (C,D1,H1,W1) for 3d
                                    (C,H1,W1) for 2d
    '''
    if ref_grid is None:
        ref_grid = get_reference_grid(ddf.shape[:-1], device=ddf.device)
    warped_grid = ref_grid + ddf
    warped = sampler(vol, warped_grid)
    return warped


def upsample_control_grid(control_grid, ref_grid):
    '''
    implements the up-sampling of the control grid to the sampling grid with linear interpolation
    control_grid: torch.tensor of shape (D,H,W,3) where dim=3 is the coordinate vector xyz (<- ijk)
                                        (H,W,2) for 2d where dim=2 is the coordinate vector xy (<- ij)
    ref_grid: torch.tensor of shape (D1,H1,W1,3) where dim=3 is the coordinate vector xyz (<- ijk)
                                    (H1,W1,2) for 2d where dim=2 is the coordinate vector xy (<- ij)
    Returns a sample_grid of shape (D1,H1,W1,3) where dim=3 is the coordinate vector xyz (<- ijk)
                                   (H1,W1,2) for 2d where dim=2 is the coordinate vector xy (<- ij)
    '''
    if control_grid.shape[-1] == 3:
        sample_grid = sampler(control_grid.permute(3,0,1,2),ref_grid).permute(1,2,3,0)
    elif control_grid.shape[-1] == 2:
        sample_grid = sampler(control_grid.permute(2,0,1),ref_grid).permute(1,2,0)
    return sample_grid
