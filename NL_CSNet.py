"""
 @Time    : 2023/5/12 14:39
 @Author  : Youcan Xu
 @E-mail  : youcanxv@163.com
 @Project : Code reproduction of Image Compressed Sensing Using Non-local Neural Network
 @File    : NL_CSNet.py
 @Function: structure of NL_CSNet
"""
from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.autograd import Variable
from non_local_embedded_gaussian import NONLocalBlock2D

#-- Reshape + Concat layer --#
class Reshape_Concat_Adap(torch.autograd.Function):
    blocksize = 0
    def __init__(self, block_size):
        Reshape_Concat_Adap.blocksize = block_size
    @staticmethod
    def forward(ctx, input_, ):
        """重新局部一维变成二维图像"""
        ctx.save_for_backward(input_)
        data = torch.clone(input_.data)
        b_ = data.shape[0]
        c_ = data.shape[1]
        w_ = data.shape[2]
        h_ = data.shape[3]
        output = torch.zeros((b_, int(c_ / Reshape_Concat_Adap.blocksize / Reshape_Concat_Adap.blocksize),
                              int(w_ * Reshape_Concat_Adap.blocksize), int(h_ * Reshape_Concat_Adap.blocksize))).to(
            device)
        #-- 图像拼接 --#
        for i in range(0, w_):
            for j in range(0, h_):
                data_temp = data[:, :, i, j]
                data_temp = data_temp.view((b_, int(c_ / Reshape_Concat_Adap.blocksize / Reshape_Concat_Adap.blocksize),
                                            Reshape_Concat_Adap.blocksize, Reshape_Concat_Adap.blocksize))
                output[:, :, i * Reshape_Concat_Adap.blocksize:(i + 1) * Reshape_Concat_Adap.blocksize,
                j * Reshape_Concat_Adap.blocksize:(j + 1) * Reshape_Concat_Adap.blocksize] += data_temp
        return output
    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        input_ = torch.clone(inp.data)
        grad_input = torch.clone(grad_output.data)

        b_ = input_.shape[0]
        c_ = input_.shape[1]
        w_ = input_.shape[2]
        h_ = input_.shape[3]

        output = torch.zeros((b_, c_, w_, h_)).to(device)
        output = output.view(b_, c_, w_, h_)
        for i in range(0, w_):
            for j in range(0, h_):
                data_temp = grad_input[:, :, i * Reshape_Concat_Adap.blocksize:(i + 1) * Reshape_Concat_Adap.blocksize,
                            j * Reshape_Concat_Adap.blocksize:(j + 1) * Reshape_Concat_Adap.blocksize]
                # data_temp = torch.zeros(data_t.shape).cuda() + data_t
                data_temp = data_temp.contiguous()
                data_temp = data_temp.view((b_, c_, 1, 1))
                output[:, :, i, j] += torch.squeeze(data_temp)

        return Variable(output)


def My_Reshape_Adap(input, blocksize):
    return Reshape_Concat_Adap(blocksize).apply(input)
#-- The residualblock for reconstruction network --#
class ResidualBlock(nn.Module):
    def __init__(self, channels, has_BN=False):
        super(ResidualBlock, self).__init__()
        self.has_BN = has_BN
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        if has_BN:
            self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        if has_BN:
            self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        if self.has_BN:
            residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        if self.has_BN:
            residual = self.bn2(residual)
        return x + residual

#-- code of Upsample in proposed non-local subnetwork --#
class Upsample(nn.Module):
    def __init__(self,factor=2,dr=3,inplanes=64,outplanes=1):
        """
        输入特征图，对其做上采样
        :param factor:上采样系数 应该为 16, 4, 1
        :param dr: 3
        :param inplanes: 输入通道数
        """
        super(Upsample, self).__init__()
        self.pixelShuffle = nn.PixelShuffle(factor)#上采样
        self.dl = dr
        denseNetBlocks = []
        for i in range(dr):
            denseNetBlocks.append(ResidualBlock(inplanes, has_BN=False))#图中未画出有BatchNormal层
        self.denseNetBlocks = nn.ModuleList(denseNetBlocks)
        outplanes = (dr+1)*inplanes
        assert 2*outplanes % factor**2 == 0 ,'channel wrong!' #通道数应该能被factor**2 整除
        self.aggregation = nn.Conv2d(outplanes,2*outplanes,1,1)
        self.conv = nn.Conv2d(2*outplanes//(factor**2),1,1,1)
    def forward(self,x):
        """
        :param x: (B,C,H,W),factor=4  例如输入(B,64,2,2)
        :return: (B,4*C*2//factor**2,H*factor,W*factor) 输出(B,32,8,8)
        """
        features = []
        features.append(x)
        for i in range(self.dl):
            feature_add = torch.zeros_like(x)
            for feature in features:
                feature_add = feature + feature_add
            x = self.denseNetBlocks[i](feature_add)
            features.append(x)
        feature_add=torch.cat(features,dim=1)
        out = self.aggregation(feature_add)
        out = self.pixelShuffle(out)
        return self.conv(out)
class Downsample(nn.Module):
    def __init__(self,factor=2,dl=3,inplanes=16,outplanes=32):
        """
        :param factor:对应论文中的downsample 系数 应该为 16, 4, 1
        :param dl:Fig4 中Downsample模块中的 Residual Block 个数
        :param inplanes:#上一个水平分支的channel数
        :param outplanes:#下一个分支
        """
        super(Downsample, self).__init__()
        ConNum = int(math.log2(factor)) #论文中 downsample 操作是用stride=2 的卷积完成的
        #--DownsampleBlock--#
        downsample_convs = []
        downsample_convs.append(nn.Conv2d(inplanes,outplanes,kernel_size=3,stride=2,padding=1))
        for i in range(ConNum-1):
            downsample_convs.append(nn.Conv2d(outplanes,outplanes,kernel_size=3,stride=2,padding=1))
        self.downSample = nn.Sequential(*downsample_convs)
        #--denseNet block--#
        self.dl = dl
        denseNetBlocks = []
        for i in range(dl):
            denseNetBlocks.append(ResidualBlock(outplanes, has_BN=False))#图中未画出有BatchNormal层
        self.denseNetBlocks = nn.ModuleList(denseNetBlocks)
        self.aggregation = nn.Conv2d((dl+1)*outplanes,outplanes,1,1)
    def forward(self,x):
        """
        :param x:上一个水平分支的输入 以(B,inplanes,128,128)举例
        :return: (B,outplanes,128/factor,128/factor) 下一个水平分支的输入
        """
        x = self.downSample(x) #下采样 factor倍  (B,outplanes,8,8)
        features = []
        features.append(x)
        for i in range(self.dl):
            feature_add = torch.zeros_like(x)
            for feature in features:
                feature_add = feature + feature_add
            x = self.denseNetBlocks[i](feature_add)
            features.append(x)
        feature_add=torch.cat(features,dim=1)
        return self.aggregation(feature_add)

class MSNL(nn.Module):
    def __init__(self, blocksize=32, inplanes=1, outplanes=16, HbranchNums=3,
                 VbranchNums=4, dl=3, SB=3, SN=3,sub_sample=False):
        """
        :param inplanes:
        :param outplanes:
        :param HbranchNums:  水平分支子模块Non-local submodule
        :param VbranchNums: 垂直模块 ，包含Downspame,UpSample
        """
        super(MSNL, self).__init__()
        self.blocksize = blocksize
        self.convIn = nn.Conv2d(inplanes,outplanes,kernel_size=3,stride=1,padding=1)#对应模型开始Conv
        self.convOut = nn.Conv2d(16,1,kernel_size=3,stride=1,padding=1)#对应模型结束Conv
        channel_inHbranches_=[16,32,64]
        factors_inVbranches =[16,4]
        #论文中出现concatentation,的地方有歧义 我认为应该是相加，而不是按通道拼接
        #--水平分支子模块Non-local submodule--#
        self.NL_feas_list=[]
        for i in range(HbranchNums):
            Hbranch_Module = nn.ModuleList(
                [NONLocalBlock2D(in_channels=channel_inHbranches_[i],sub_sample=sub_sample) for j in range(SN)]
            )
            self.NL_feas_list.append(Hbranch_Module)
        #--垂直模块 ，包含Downspame,UpSample--#
        self.Mul_Scale_modules = []
        for i in range(2):#factor 为一的应该就是主分支  #参考fig4
            Vbranch_Module = nn.ModuleList(
                [Downsample(factors_inVbranches[i], inplanes=channel_inHbranches_[i],
                            outplanes=channel_inHbranches_[i + 1]) for j in range(2)] +
                [Upsample(factor=factors_inVbranches[i], inplanes=channel_inHbranches_[i+1])
                 for j in range(2)]
            )
            self.Mul_Scale_modules.append(Vbranch_Module)

        self.nl_feas = NONLocalBlock2D(in_channels=64)
    def forward(self,x):
        """
        :param x: initial reconstruction. (B,1,128,128)
        :return: deep reconstruction. (B,1,128,128)
        """
        affinity_matrixs = []
        #--第一个垂直分支下采样--#
        x1 = self.convIn(x) #(B,16,128,128)
        x2 = self.Mul_Scale_modules[0][0](x1) #(B,32,8,8)
        x3 = self.Mul_Scale_modules[1][0](x2)#(B,32,2,2)
        # --第一列 Non-local处理--#
        x1, m1 = self.NL_feas_list[0][0](x1)
        x2, m2 = self.NL_feas_list[1][0](x2)
        x3, m3 = self.NL_feas_list[2][0](x3)
        affinity_matrixs.extend([m1,m2,m3])
        # --第二个垂直分支下采样--#
        x2 = self.Mul_Scale_modules[0][1](x1) + x2
        x3 = self.Mul_Scale_modules[1][1](x2) + x3
        # --第二列 Non-local处理--#
        x1, m1 = self.NL_feas_list[0][1](x1)
        x2, m2 = self.NL_feas_list[1][1](x2)
        x3, m3 = self.NL_feas_list[2][1](x3)
        affinity_matrixs.extend([m1, m2, m3])
        # --第一个垂直分支上采样--#
        x2 = self.Mul_Scale_modules[1][2](x3) + x2
        x1 = self.Mul_Scale_modules[0][2](x2) + x1
        # --第三列 Non-local处理--#
        x1,m1 = self.NL_feas_list[0][2](x1)
        x2,m2 = self.NL_feas_list[1][2](x2)
        x3,m3 = self.NL_feas_list[2][2](x3)
        affinity_matrixs.extend([m1, m2, m3])
        # --第二个垂直分支上采样--#
        x2 = self.Mul_Scale_modules[1][3](x3) + x2
        x1 = self.Mul_Scale_modules[0][3](x2) + x1
        out = self.convOut(x1)

        return out, affinity_matrixs

#--  code of NL-CSNet 代码复现  --#
class NL_CSNet(nn.Module):
    def __init__(self, blocksize=32, subrate=0.1,subsample=True):
        super(NL_CSNet, self).__init__()
        self.blocksize = blocksize
        #--for sampling-- 用卷积代替测量矩阵类似于Ax--#
        self.sampling = nn.Conv2d(1, int(np.round(blocksize * blocksize * subrate)), blocksize, stride=blocksize,
                                  padding=0, bias=False)
        #-- non-local subnetwork in measurement domain --#
        self.nl_meas = NONLocalBlock2D(in_channels=int(np.round(blocksize * blocksize * subrate)))
        #-- MS-NLNet -- #
        self.MS_Nl = MSNL(blocksize=blocksize,sub_sample=subsample)
        #-- upsampling --#
        self.upsampling = nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), blocksize * blocksize, 1, stride=1,
                                    padding=0)
        #-- reconstruction network --#
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.PReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        """
        :param x: (B,1,H,W) 这里以(B,1,128,128)举例
        :return:
        """
        #-- image Sampling --#
        x = self.sampling(x)  # (B,102,4,4)  102 相当于y = Ax + η中的 y
        #-- Non-local in Measurement domain --#
        x, mat_meas = self.nl_meas(x)  # x(Batch,102,4,4),mat_meas(Batch,16,16)
        x = self.upsampling(x)  # (Batch,1024,4,4)

        # --得到initial reconstruction--#
        x = My_Reshape_Adap(x, self.blocksize)  # Reshape + Concat
        block1 = self.conv1(x)
        block2 = self.conv2(block1)
        block3 = self.conv3(block2)

        #-- deep reconstruction --#
        block3, mat_feas_list = self.MS_Nl(block3)
        block4 = self.conv4(block3)
        block5 = self.conv5(block4)
        return block5, mat_meas, mat_feas_list  #(Batch,1,128,128),(Batch,1,4,4),list:保存每个相似度矩阵
if __name__ == '__main__':
    import torch
    from thop import profile

    img = torch.randn(1, 1, 128, 128)
    net = NL_CSNet()
    out = net(img)
    print(out)


