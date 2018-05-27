import argparse
import os
import time
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim

# custom modules

from loss import MonodepthLoss
from data_loader import image_transforms, KittiLoader, ImageLoader
import bilinear_sampler_pytorch
import models_resnet

# plot params

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15, 10)


def return_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Monodepth')

    parser.add_argument('data_dir',
                        help='path to the dataset folder. \
                        It should contain subfolders with following structure:\
                        "image_02/data" for left images and \
                        "image_03/data" for right images'
                        )
    parser.add_argument('model_path', help='path to the trained model')
    parser.add_argument('output_directory',
                        help='where save dispairities\
                        for tested images'
                        )
    parser.add_argument('--input_height', type=int, help='input height',
                        default=256)
    parser.add_argument('--input_width', type=int, help='input width',
                        default=512)
    parser.add_argument('--model', default='resnet50_md',
                        help='encoder architecture: ' +
                        'resnet18 or resnet50' + '(default: resnet50)'
                        )
    parser.add_argument('--mode', default='train',
                        help='mode: train or test (default: train)')
    parser.add_argument('--epochs', default=50,
                        help='number of total epochs to run')
    parser.add_argument('--learning_rate', default=1e-4,
                        help='initial learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', default=256,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--adjust_lr', default=True,
                        help='apply learning rate decay or not\
                        (default: True)'
                        )
    parser.add_argument('--tensor_type',
                        default='torch.cuda.FloatTensor',
                        help='choose type for GPU "torch.cuda.FloatTensor" \
                              or type for CPU "torch.FloatTensor"'
                        )
    parser.add_argument('--do_augmentation', default=True,
                        help='do augmentation of images or not')
    parser.add_argument('--augment_parameters', default=[
        0.8,
        1.2,
        0.5,
        2.0,
        0.8,
        1.2,
        ],
            help='lowest and highest values for gamma,\
                        brightness and color respectively'
            )
    parser.add_argument('--print_images', default=False,
                        help='print disparity and image\
                        generated from disparity on every iteration'
                        )
    parser.add_argument('--print_weights', default=False,
                        help='print weights of every layer')
    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR\
        decayed by 10 every 30 epochs"""

    if epoch >= 30 and epoch < 40:
        lr = learning_rate / 2
    elif epoch >= 40:
        lr = learning_rate / 4
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) \
        * m_disp


class Model:

    def __init__(self, args):
        self.args = args
        if args.mode == 'train':
            data_dirs = os.listdir(args.data_dir)
            data_transform = image_transforms(
                    mode=args.mode,
                    tensor_type=args.tensor_type,
                    augment_parameters=args.augment_parameters,
                    do_augmentation=args.do_augmentation)
            train_datasets = [KittiLoader(os.path.join(args.data_dir,
                              data_dir), True,
                              transform=data_transform) for data_dir in
                              data_dirs]
            train_dataset = ConcatDataset(train_datasets)
            self.n_img = train_dataset.__len__()
            print ('Use a dataset with', self.n_img, 'images')
            self.train_loader = DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True)
            self.device = torch.device((
                'cuda:0' if torch.cuda.is_available() and
                args.tensor_type == 'torch.cuda.FloatTensor' else 'cpu'))
            self.loss_function = MonodepthLoss(
                    n=4,
                    SSIM_w=0.85,
                    disp_gradient_w=0.1, lr_w=1,
                    tensor_type=args.tensor_type).to(self.device)
            if args.model == 'resnet50_md':
                self.model = models_resnet.resnet50_md(3)
            elif args.model == 'resnet18_md':
                self.model = models_resnet.resnet18_md(3)
            self.model = self.model.to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=args.learning_rate)
            if args.tensor_type == 'torch.cuda.FloatTensor':
                torch.cuda.synchronize()
        elif args.mode == 'test':
            self.output_directory = args.output_directory

            # loading data

            self.input_height = args.input_height
            self.input_width = args.input_width
            data_transform = image_transforms(mode=args.mode,
                                              tensor_type=args.tensor_type)

            test_dataset = ImageLoader(args.data_dir, False,
                                       transform=data_transform)
            self.num_test_examples = test_dataset.__len__()
            self.test_loader = DataLoader(test_dataset, batch_size=1,
                                          shuffle=False)

            # set up CPU device

            self.device = torch.device('cpu')

            # define model

            if args.model == 'resnet50_md':
                self.model = models_resnet.resnet50_md(3)
            elif args.model == 'resnet18_md':
                self.model = models_resnet.resnet18_md(3)
            self.model.load_state_dict(torch.load(args.model_path))
            self.model = self.model.to(self.device)

    def train(self):

        # Start training

        losses = []
        best_loss = 1e19  # Just a big number

        # Loop over the dataset multiple times

        self.model.train()
        for epoch in range(self.args.epochs):
            if self.args.adjust_lr:
                adjust_learning_rate(self.optimizer, epoch,
                                     self.args.learning_rate)
            running_loss = 0.0
            c_time = time.time()
            for (i, data) in enumerate(self.train_loader, 0):

                # get the inputs

                left = data['left_image'].to(self.device)
                right = data['right_image'].to(self.device)

                # zero the parameter gradients

                self.optimizer.zero_grad()

                # forward + backward + optimize

                disps = self.model(left)
                loss = self.loss_function(disps, [left, right])
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

                # print statistics

                if self.args.print_weights:
                    j = 1
                    for (name, parameter) in self.model.named_parameters():
                        if name.split(sep='.')[-1] == 'weight':
                            plt.subplot(5, 9, j)
                            plt.hist(parameter.data.view(-1))
                            plt.xlim([-1, 1])
                            plt.title(name.split(sep='.')[0])
                            j += 1
                    plt.show()

                if self.args.print_images:
                    print('disp_left_est[0]')
                    plt.imshow(np.squeeze(
                        np.transpose(self.loss_function.disp_left_est[0][0,
                                     :, :, :].cpu().detach().numpy(),
                                     (1, 2, 0))))
                    plt.show()
                    print('left_est[0]')
                    plt.imshow(np.transpose(self.loss_function.left_est[0][0,
                               :, :, :].cpu().detach().numpy(), (1, 2,
                               0)))
                    plt.show()
                    print('disp_right_est[0]')
                    plt.imshow(np.squeeze(
                        np.transpose(self.loss_function.disp_right_est[0][0,
                                     :, :, :].cpu().detach().numpy(),
                                     (1, 2, 0))))
                    plt.show()
                    print('right_est[0]')
                    plt.imshow(np.transpose(self.loss_function.right_est[0][0,
                               :, :, :].cpu().detach().numpy(), (1, 2,
                               0)))
                    plt.show()

                running_loss += loss.item()

            # Estimate loss per image

            running_loss /= self.n_img / self.args.batch_size
            print (
                'Epoch:',
                epoch + 1,
                'loss:',
                running_loss,
                'time:',
                round(time.time() - c_time, 3),
                's',
                )
            if running_loss < best_loss:
                self.save(self.args.model_path[:-4] + '_cpt.pth')
                best_loss = running_loss
                print('Model_saved')
            running_loss = 0.0

        print ('Finished Training. Best loss:', best_loss)
        self.save(self.args.model_path)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def test(self):
        self.model.eval()

        # start testing

        disparities = np.zeros((self.num_test_examples,
                               self.input_height, self.input_width),
                               dtype=np.float32)
        disparities_pp = np.zeros((self.num_test_examples,
                                  self.input_height, self.input_width),
                                  dtype=np.float32)
        with torch.no_grad():

            for (i, data) in enumerate(self.test_loader, 0):

                # get the inputs

                left = data.squeeze()
                left = left.to(self.device)

                # forward

                disps = self.model(left)
                disp = disps[0][:, 0, :, :].unsqueeze(1)
                disparities[i] = disp[0].squeeze()
                disparities_pp[i] = \
                    post_process_disparity(disp.squeeze().numpy())

        np.save(self.output_directory + '/disparities.npy', disparities)
        np.save(self.output_directory + '/disparities_pp.npy',
                disparities_pp)
        print('Finished Testing')


def main(args):
    args = return_arguments()
    if args.mode == 'train':
        model = Model(args)
        model.train()
    elif args.mode == 'test':
        model_test = Model(args)
        model_test.test()


if __name__ == '__main__':
    main()

