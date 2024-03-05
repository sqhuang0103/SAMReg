
import torch

from region_correspondence.optim import iterative_ddf


class PairedRegions():

    def __init__(self, masks_mov, masks_fix, device=None):
        '''
        masks_mov: torch.tensor of shape (C,D0,H0,W0) for 3d, 
                            (C,H0,W0) for 2d, where C is the number of masks
        masks_fix: torch.tensor of shape (C,D1,H1,W1) for 3d, 
                            (C,H1,W1) for 2d, where C is the number of masks
        '''
        self.masks_mov = masks_mov
        self.masks_fix = masks_fix
        self.device = device
        if self.device is not None:
            self.masks_mov = self.masks_mov.to(device)
            self.masks_fix = self.masks_fix.to(device)

    def get_dense_correspondence(self, transform_type='ddf', **kwargs):
        '''
        transform_type: str, one of ['ddf', 'ffd', 'affine', 'spline']
            ddf implements the direct dense displacement field optimisation. 
            ffd implements the free-form deformation based on a control point grid.
        Returns a dense displacement field (DDF) of shape (H1,W1,D1,3) where the dim=0 is the displacement vector
        '''
        match transform_type.lower():
            case 'ddf':
                self.ddf, _ = iterative_ddf(mov=self.masks_mov.type(torch.float32), fix=self.masks_fix.type(torch.float32), control_grid_size=None, device=self.device, **kwargs)  # grid_sample requires float32
            case 'ffd':
                self.ddf, self.control_grid = iterative_ddf(mov=self.masks_mov.type(torch.float32), fix=self.masks_fix.type(torch.float32), control_grid_size=10, device=self.device, **kwargs)
            case 'affine':
                raise NotImplementedError("TPS transform is not implemented yet.")
            case 'spline':
                raise NotImplementedError("TPS transform is not implemented yet.")
            case _:
                raise ValueError("Unknown transform type: {}".format(transform_type))
        # transform_type_lowered = transform_type.lower()
        # if transform_type_lowered == 'ddf':
        #     self.ddf, _ = iterative_ddf(mov=self.masks_mov.type(torch.float32), fix=self.masks_fix.type(torch.float32),
        #                                 control_grid_size=None, device=self.device,
        #                                 **kwargs)  # grid_sample requires float32
        # elif transform_type_lowered == 'ffd':
        #     self.ddf, self.control_grid = iterative_ddf(mov=self.masks_mov.type(torch.float32),
        #                                                 fix=self.masks_fix.type(torch.float32), control_grid_size=10,
        #                                                 device=self.device, **kwargs)
        # elif transform_type_lowered == 'affine':
        #     raise NotImplementedError("TPS transform is not implemented yet.")
        # elif transform_type_lowered == 'spline':
        #     raise NotImplementedError("TPS transform is not implemented yet.")
        # else:
        #     raise ValueError("Unknown transform type: {}".format(transform_type))

        return self.ddf
