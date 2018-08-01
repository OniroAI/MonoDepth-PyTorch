import torch
import torch.nn.functional as F


def apply_disparity(img, disp, tensor_type='torch.cuda.FloatTensor'):
    batch_size, _, height, width = img.size()
    
    # Original coordinates of pixels
    x_base = torch.linspace(-1, 1, width).repeat(batch_size, height, 1).type(tensor_type)
    y_base = torch.linspace(-1, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type(tensor_type)

    # Apply shift in X direction
    x_shifts = disp[:,0,:,:] # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    output = F.grid_sample(img, flow_field, mode='bilinear', padding_mode='zeros')

    return output
