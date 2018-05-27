import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from bilinear_sampler_pytorch import apply_disparity


class MonodepthLoss(nn.modules.Module):
    def __init__(self, n=4, SSIM_w=0.85, disp_gradient_w=1.0, lr_w=1.0,
                 tensor_type = 'torch.cuda.FloatTensor'):
        super(MonodepthLoss, self).__init__()
        self.SSIM_w = SSIM_w
        self.disp_gradient_w = disp_gradient_w
        self.lr_w = lr_w
        self.n = n
        self.tensor_type = tensor_type

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = img.size()
        h = s[2]
        w = s[3]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(nn.Upsample(size=[nh, nw],
                               mode='bilinear')(img))
        return scaled_imgs

    def gradient_x(self, img):
        gx = img[:,:,:,:-1] - img[:,:,:,1:]  # NCHW
        return gx

    def gradient_y(self, img):
        gy = img[:,:,:-1,:] - img[:,:,1:,:]  # NCHW
        return gy

    def generate_image_left(self, img, disp, tensor_type):
        return apply_disparity(img, -disp, tensor_type=tensor_type)

    def generate_image_right(self, img, disp, tensor_type):
        return apply_disparity(img, disp, tensor_type=tensor_type)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x  = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y  = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)
    
    def disp_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i]\
                        for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i]\
                        for i in range(self.n)]

        return smoothness_x + smoothness_y

    def forward(self, input, target):
        """
        Args:
            input [disp1, disp2, disp3, disp4]
            target [left, right]
        
        Return:
            (float): The loss
        """
        left, right = target
        left_pyramid = self.scale_pyramid(left, self.n)
        right_pyramid = self.scale_pyramid(right, self.n)

        # Prepare disparities
        disp_left_est  = [d[:, 0, :, :].unsqueeze(1) for d in input]
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in input]
        
        self.disp_left_est = disp_left_est
        self.disp_right_est = disp_right_est
        # Generate images
        left_est  = [self.generate_image_left(right_pyramid[i],
                     disp_left_est[i], self.tensor_type)  for i in range(self.n)]
        right_est = [self.generate_image_right(left_pyramid[i],
                     disp_right_est[i], self.tensor_type) for i in range(self.n)]
        self.left_est = left_est
        self.right_est = right_est

        # L-R Consistency
        right_left_disp = [self.generate_image_left(disp_right_est[i],
                           disp_left_est[i], self.tensor_type) for i in range(self.n)]
        left_right_disp = [self.generate_image_right(disp_left_est[i],
                           disp_right_est[i], self.tensor_type) for i in range(self.n)]

        # Disparities smoothness
        disp_left_smoothness  = self.disp_smoothness(disp_left_est,
                                                     left_pyramid)
        disp_right_smoothness = self.disp_smoothness(disp_right_est,
                                                     right_pyramid)

        # L1 
        l1_left = [torch.mean(torch.abs(left_est[i] - left_pyramid[i]))\
                              for i in range(self.n)]
        l1_right = [torch.mean(torch.abs(right_est[i]\
                    - right_pyramid[i])) for i in range(self.n)]

        # SSIM
        ssim_left = [torch.mean(self.SSIM(left_est[i],\
                     left_pyramid[i])) for i in range(self.n)]
        ssim_right = [torch.mean(self.SSIM(right_est[i],\
                      right_pyramid[i])) for i in range(self.n)]

        image_loss_left  = [self.SSIM_w * ssim_left[i]\
                            + (1-self.SSIM_w) * l1_left[i]\
                            for i in range(self.n)]
        image_loss_right = [self.SSIM_w * ssim_right[i]\
                            + (1-self.SSIM_w) * l1_right[i]\
                            for i in range(self.n)]
        image_loss = sum(image_loss_left + image_loss_right)

        # L-R Consistency
        lr_left_loss = [torch.mean(torch.abs(right_left_disp[i]\
                        - disp_left_est[i]))  for i in range(self.n)]
        lr_right_loss = [torch.mean(torch.abs(left_right_disp[i]\
                         - disp_right_est[i])) for i in range(self.n)]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        # Disparities smoothness
        disp_left_loss  = [torch.mean(torch.abs(\
                           disp_left_smoothness[i])) / 2**i\
                           for i in range(self.n)]
        disp_right_loss = [torch.mean(torch.abs(\
                           disp_right_smoothness[i])) / 2**i\
                           for i in range(self.n)]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)

        loss = image_loss + self.disp_gradient_w * disp_gradient_loss\
               + self.lr_w * lr_loss
        self.image_loss = image_loss
        self.disp_gradient_loss = disp_gradient_loss
        self.lr_loss = lr_loss
        return loss
