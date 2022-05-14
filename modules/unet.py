import torch.nn as nn


class Unet(nn.Module):
    def __init__(self, hid_ds, sizes, stride, upsample_type='transpose'):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        for i in range(len(hid_ds)-1):
            self.downsample_layers.append(
                UnetDownsample(hid_ds[i], hid_ds[i+1],
                               sizes[i], sizes[i+1],
                               stride))
        sizes.reverse()
        hid_ds.reverse()
        for i in range(len(hid_ds)-1):
            self.upsample_layers.append(
                UnetUpsample(
                    hid_ds[i], hid_ds[i+1],
                    sizes[i], sizes[i+1], 
                    stride, upsample_type))
        
    def forward(self, x):
        img_zs = []
        for i, layer in enumerate(self.upsample_layers):
            x = layer(x)
            img_zs.append(x)
        img_zs.reverse()
        
        for i, layer in enumerate(self.downsample_layers):
            x = layer(x) + img_zs[i]
        return x
    
class UnetDownsample(nn.Module):
    def __init__(self, in_d, out_d, in_size, out_size, stride):
        super().__init__()
        '''
        Given input size, output size & stride 
        kernel size is automatically selected
        '''
        # -[(output - 1)*stride]-input = kernel
        k_size_h = -int(((out_size[0] - 1)*stride)-in_size[0])
        k_size_w = -int(((out_size[1] - 1)*stride)-in_size[1])
        self.layer = nn.Sequential(
            nn.Conv2d(in_d, out_d, 
                      (k_size_h, k_size_w), 
                      stride=stride),
            nn.ReLU(),
            nn.BatchNorm2d(out_d),
            nn.Dropout2d(0.3))
        
    def forward(self, x):
        return self.layer(x)
        

class UnetUpsample(nn.Module):
    def __init__(self, in_d, out_d, in_size, out_size, stride, type='transpose'):
        super().__init__()
        '''
        Given input size, output size & stride 
        kernel size is automatically selected
        '''
        assert type in ['transpose', 'upsample'], \
            'UnetUpsample layers supports transpose or upsample only'
        # -([(output - 1)*stride]-input)-2*padding = kernel
        k_size_h = -int(((in_size[0] - 1)*stride)-out_size[0])
        k_size_w = -int(((in_size[1] - 1)*stride)-out_size[1])
        if type == 'transpose':        
            self.layer = nn.Sequential(
                nn.ConvTranspose2d(in_d, out_d, 
                                   (k_size_h, k_size_w), 
                                   stride=stride),
                nn.ReLU(),
                nn.BatchNorm2d(out_d),
                nn.Dropout2d(0.3))
        elif type == 'upsample':
            # TODO
            assert 1 == 0,'TODO fix implementation'
            self.layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_d, out_d, 1),
                nn.ReLU(),
                nn.BatchNorm2d(out_d),
                nn.Dropout2d(0.3))
        
    def forward(self, x):
        return self.layer(x)
        
        
hid_ds = [3, 3,3,3,3,3 ] 
hid_ds.reverse()
for i in range(len(hid_ds)-1):
    print(hid_ds[i+1])
sizes = [ [64,64],[32, 32], [16, 16], [8,8], [4,4], [2,2]]
net = Unet(hid_ds, sizes, 2, 'upsample')
import torch
x = torch.randn((4, 3, 32, 32))
out = net(x)
print(out.shape)

# import torch
# net = UnetUpsample(3, 3, [128,128],[256,256], 1)
# x = torch.randn((3, 3, 128,128))
# out = net(x)
# print(out.shape)
