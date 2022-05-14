import torch
import torch.nn as nn
import torch.nn.functional as F

class Wavenet(nn.Module):
    def __init__(self,
                n_blocks,
                n_layers_per_block,
                in_channels,
                out_channels,
                dilation_channels,
                residual_channels,
                skip_channels,
                kernel_size,
                conditional,
                conditional_channels):
        super().__init__()
        
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_channels, residual_channels, 1),
            nn.ReLU(0.2),
            nn.Conv1d(residual_channels, residual_channels, 1),
            nn.ReLU(0.2))
        self.out_conv = nn.Sequential(
            nn.Conv1d(skip_channels, out_channels, 1),
            nn.ReLU(0.2),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.ReLU(0.2))
       
        self.res_blocks = nn.ModuleList()
        for i in range(n_blocks):
            dilation = 1
            self.res_layers = nn.ModuleList()
            
            for i in range(n_layers_per_block):
                self.res_layers.append(
                    ResidualLayer(
                        dilation,
                        dilation_channels,
                        residual_channels,
                        skip_channels,
                        kernel_size,
                        conditional,
                        conditional_channels,
                    ))
            
                dilation *= 2
                
            self.res_blocks.append(self.res_layers)            
        
    def forward(self, input, condition=None):
        '''
        Condition is additional information such as
        previous head poses 
        input (eg: head_poses): [BS, N, L]
        condition (eg: audio): [BS, N, L]
        '''
        x = self.in_conv(input)
        skip = 0
        for res_block in self.res_blocks:
            for res_layer in res_block:
                x, new_skip = res_layer(x, condition)
                skip += new_skip
            
        out = self.out_conv(skip)
        return out
        
class ResidualLayer(nn.Module):
    def __init__(self,
                 dilation,
                 dilation_channels,
                 residual_channels,
                 skip_channels,
                 kernel_size,
                 conditional,
                 conditional_channels,
                 ):
        super().__init__()
        '''
        ######################### Current Residual Block ##########################
        #     |-----------------------*residual*--------------------|             #
        #     |                                                     |             # 
        #     |        |-- dilated conv -- tanh --|                 |             #
        # -> -|-- pad--|                          * ---- |-- 1x1 -- + --> *input* #
        #              |-- dilated conv -- sigm --|      |                        #
        #                                               1x1                       # 
        #                                                |                        # 
        # ---------------------------------------------> + -------------> *skip*  #
        ###########################################################################
        '''
        self.conditional = conditional
        self.padding = (int((kernel_size-1) * dilation), 0)
        # Dilated convolutions
        self.filter_conv = nn.Conv1d(
                    in_channels=residual_channels,
                    out_channels=dilation_channels,
                    kernel_size=kernel_size,
                    dilation=dilation)
        self.gate_conv = nn.Conv1d(
                    in_channels=residual_channels,
                    out_channels=dilation_channels,
                    kernel_size=kernel_size,
                    dilation=dilation)
        
        self.residual_conv = nn.Conv1d(
                    in_channels=dilation_channels,
                    out_channels=residual_channels,
                    kernel_size=1)
        self.skip_conv = nn.Conv1d(
                    in_channels=dilation_channels,
                    out_channels=skip_channels,
                    kernel_size=1)
        
        if self.conditional == True:
            self.conditional_filter_conv = nn.Conv1d(
                        in_channels=conditional_channels,
                        out_channels=dilation_channels,
                        kernel_size=1)
            self.conditional_gate_conv = nn.Conv1d(
                        in_channels=conditional_channels,
                        out_channels=dilation_channels,
                        kernel_size=1)
            
    def forward(self, input, conditional=None):
        input_pad = F.pad(input, self.padding)
        
        filter_out = self.filter_conv(input_pad)
        gate_out = self.gate_conv(input_pad)
        if conditional is not None and self.conditional == True:
            conditional_filter_out = self.conditional_filter_conv(conditional)
            conditional_gate_out = self.conditional_gate_conv(conditional)  
            
            filter_out = filter_out + conditional_filter_out
            gate_out = gate_out + conditional_gate_out
        
        filter_out = torch.tanh(filter_out)
        gate_out = torch.sigmoid(gate_out)
        
        x = filter_out * gate_out
        residual = self.residual_conv(x) + input
        skip = self.skip_conv(x)
        
        return residual, skip
    
    
import torch
net = Wavenet(2, 7, 256, 2, 32, 32, 256, 3, True, 128)

audio = torch.randn((8, 1455, 256))
cond = torch.randn((8, 1455 ,128))

out = net(audio.transpose(1,2), cond.transpose(1,2))
print(out.shape)
