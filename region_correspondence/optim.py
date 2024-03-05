
import torch

from region_correspondence.utils import get_reference_grid, upsample_control_grid, sampler
from region_correspondence.metrics import DDFLoss, ROILoss


def iterative_ddf(mov, fix, control_grid_size=None, device=None, max_iter=int(1e5), lr=1e-3, w_ddf=1.0, verbose=False):
    '''
    Implements the free-form deformation (FFD) estimation based on control point grid (control_grid), using iterative optimisation
        when control_grid_size = None, the dense displacement field (DDF) estimation is estimated using the iterative optimisation
    mov: torch.tensor of shape (C,D0,H0,W0) where C is the number of masks
    fix: torch.tensor of shape (C,D1,H1,W1) where C is the number of masks
    control_grid_size: 
        None for DDF estimation
        when specified, tuple of 3 ints for 3d, tuple of 2 ints for 2d, or tuple of 1 int for the same size in all dimensions
    Returns a dense displacement field (DDF) of shape (D1,H1,W1,3) where the dim=3 contains the displacement vectors
    '''
    num_masks = mov.shape[0]
    if num_masks != fix.shape[0]:
        raise ValueError("mov and fix must have the same number of masks.")
    if mov.dim() != fix.dim():
        raise ValueError("mov and fix must have the same dimensionality.")
    if isinstance(control_grid_size,int):
        if mov.dim() == 4:
            control_grid_size = (control_grid_size,control_grid_size,control_grid_size)
        elif mov.dim() == 3:
            control_grid_size = (control_grid_size,control_grid_size)

    if verbose:
            if control_grid_size is None:
                print("Optimising DDF (dense displacement field):")
            elif len(control_grid_size) == 3:
                print("Optimising FFD (free-form deformation) with control grid size ({},{},{}):".format(*control_grid_size))
            elif len(control_grid_size) == 2:
                print("Optimising FFD (free-form deformation) with control grid size ({},{}):".format(*control_grid_size))
    
    ref_grid = get_reference_grid(grid_size=fix.shape[1:], device=device)
    if control_grid_size is not None:
        control_grid = get_reference_grid(grid_size=control_grid_size, device=device)
    else:  #ddf
        control_grid = get_reference_grid(grid_size=ref_grid.shape[:-1], device=device)
    control_grid += torch.normal(mean=0, std=1e-5, size=control_grid.shape, dtype=torch.float32, device=device)  # initialise to break symmetry
    control_grid.requires_grad = True

    optimizer = torch.optim.Adam(params=[control_grid], lr=lr)
    loss_roi = ROILoss(w_overlap=1.0, w_class=0.1) 
    loss_ddf = DDFLoss(type='bending')
    metric_overlap = ROILoss(w_overlap=1.0, w_class=0.0)

    for iter in range(max_iter):
        
        optimizer.zero_grad()

        if control_grid_size is not None:
            sample_grid = upsample_control_grid(control_grid, ref_grid)
        else:  #ddf
            sample_grid = control_grid
        warped = sampler(mov, sample_grid)
        ddf = sample_grid-ref_grid

        loss_value_roi = loss_roi(warped,fix)
        loss_value_ddf = loss_ddf(ddf)
        loss = loss_value_roi + loss_value_ddf*w_ddf
        
        if verbose:
            if iter % 100 == 0:
                print("iter={}: loss={:0.5f} (deform={:0.5f}, overlap={:0.5f})".format(iter, loss, loss_value_ddf, 1-metric_overlap(warped,fix)))
        
        loss.backward()
        optimizer.step()
    
    return ddf, control_grid 
