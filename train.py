"""
 @Time    : 2023/5/13 14:39
 @Author  : Youcan Xu
 @E-mail  : youcanxv@163.com
 @Project : Code reproduction of Image Compressed Sensing Using Non-local Neural Network
 @File    : train.py
 @Function: train NLCSNet
"""
import torch
import torch.utils.data as Data
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from NL_CSNet import NL_CSNet
from torch import nn
import time
import os
from tensorboardX import SummaryWriter
import argparse
from tqdm import tqdm
import datetime
from data_utils import TrainDatasetFromFolder,checkdir
import torchvision.transforms as transforms
from torch.autograd import Variable
ckpt_path = './ckpt'
exp_name = 'Net'
checkdir(ckpt_path)
file = open("./ckpt/net", "a")
checkdir(os.path.join(ckpt_path, exp_name))
vis_path = os.path.join(ckpt_path, exp_name, 'log')
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')
writer = SummaryWriter(log_dir=vis_path, comment=exp_name)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--crop_size', default=96, type=int, help='training images crop size')
    parser.add_argument('--block_size', default=32, type=int, help='CS block size')
    parser.add_argument('--pre_epochs', default=200, type=int, help='pre train epoch number')
    parser.add_argument('--num_epochs', default=300, type=int, help='train epoch number')
    parser.add_argument('--train_save', type=str,default='NET')
    parser.add_argument('--batchSize', default=8, type=int, help='train batch size')
    parser.add_argument('--sub_rate', default=0.5, type=float, help='sampling sub rate')

    parser.add_argument('--loadEpoch', default=0, type=int, help='load epoch number')
    parser.add_argument('--generatorWeights', type=str, default='', help="path to CSNet weights (to continue training)")

    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    BLOCK_SIZE = opt.block_size
    NUM_EPOCHS = opt.num_epochs
    PRE_EPOCHS = opt.pre_epochs
    LOAD_EPOCH = 25
    train_set = TrainDatasetFromFolder("/user50/youcan/SegMaR-main/data/train/Imgs", crop_size=CROP_SIZE, blocksize=BLOCK_SIZE)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    net = NL_CSNet(BLOCK_SIZE, opt.sub_rate)
    mse_loss = nn.MSELoss()
    if opt.generatorWeights != '':
        net.load_state_dict(torch.load(opt.generatorWeights))
        LOAD_EPOCH = opt.loadEpoch
    if torch.cuda.is_available():
        net.cuda()
        mse_loss.cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.0004, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    curr_iter =0
    start_time = time.time()
    for epoch in range(LOAD_EPOCH, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'g_loss': 0, }
        net.train()
        for data, target in train_bar:
            batch_size = data.size(0)
            if batch_size <= 0:
                continue
            running_results['batch_sizes'] += batch_size

            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img, mat_meas, mat_feas = net(z)

            # transpose
            Maps_meas = torch.clone(mat_meas)
            for i in range(0, Maps_meas.shape[0]):
                map_t = Maps_meas[i, :, :].clone()
                Maps_meas[i,:,:] = torch.t(map_t)
            Maps_feas = torch.clone(mat_feas)
            for i in range(0, Maps_feas.shape[0]):
                map_t = Maps_feas[i, :, :].clone()
                Maps_feas[i,:,:] = torch.t(map_t)

            optimizer.zero_grad()
            loss1 = mse_loss(fake_img, real_img)
            loss2 = mse_loss(mat_meas, Maps_meas)
            loss3 = mse_loss(mat_feas, Maps_feas)
            # g_loss = mse_loss(fake_img, real_img) + 0.001*mse_loss(mat_meas, Maps_meas) + 0.001*mse_loss(mat_feas, Maps_feas)
            g_loss = loss1 + 0.001 * loss2 + 0.001 * loss3
            g_loss.backward()
            optimizer.step()
            scheduler.step()
            running_results['g_loss'] += g_loss.item() * batch_size
            # ---- train visualization ----
            if curr_iter % 10 == 0:
                writer.add_scalar('loss', g_loss, curr_iter)
                writer.add_scalar('loss_1', loss1, curr_iter)
                writer.add_scalar('loss_2', loss2, curr_iter)
                writer.add_scalar('loss_2', loss3, curr_iter)
            log = '[%d] Loss_G: %.4f lr: %.7f' % (
                epoch, running_results['g_loss'] / running_results['batch_sizes'], optimizer.param_groups[0]['lr'])
            train_bar.set_description(desc=log)
            open(log_path, 'a').write(log + '\n')
            curr_iter += 1

        # for saving model
        save_path = 'checkpoints/{}/'.format(opt.train_save)
        os.makedirs(save_path, exist_ok=True)
        if epoch % 5 == 0:
            torch.save(net.state_dict(), save_path + 'net_epoch_%d_%6f.pth' % (epoch, running_results['g_loss']/running_results['batch_sizes']))
            print('[Saving Snapshot:]', save_path + 'Net-%d.pth' % epoch)
            file.write('[Saving Snapshot:]' + save_path+ 'Net-%d.pth' % epoch + '\n')
    writer.close()
    print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
    print(exp_name)
    print("Optimization Have Done!")
    file.close()


