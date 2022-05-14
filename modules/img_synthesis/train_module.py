import torch
import pytorch_lightning as ptl
from torch.optim.lr_scheduler import StepLR
from data.dataset import ImgSynthesisDataset
from torch.utils.data import DataLoader, random_split
from modules.img_synthesis.img_generator import ImgSynthesis
from modules.img_synthesis.loss import ImgSynthesisLoss



class ImgSynthesisTrainModule(ptl.LightningModule):
    def __init__(self, model_config, data_config, train_config):
        super().__init__() 
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.model = ImgSynthesis(model_config)
        self.dataset = ImgSynthesisDataset(data_config)
        self.loss = ImgSynthesisLoss()
        
        self.data_config = data_config
        self.train_config = train_config
        # self.train_ds, self.val_ds = random_split(self.dataset,)

    def train_dataloader(self):
        return DataLoader(self.dataset, **self.data_config['train_dl'])

    # def val_dataloader(self):
        # return DataLoader(self.val_ds, **self.data_config['val_dl'])

    def forward(self, batch):
        sample_images, landmarks = batch[0], batch[1]
        return self.model(sample_images, landmarks)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        targets = batch[2]
        loss = self.loss(outputs, targets)
        return {'loss': loss}

    # def validation_step(self, batch, batch_idx):
    #     return {'loss': loss, 'logs':logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            **self.train_config['optimizer'])
        scheduler = StepLR(optimizer, 
                           **self.train_config['scheduler'])
        return {'optimizer':optimizer,
                'scheduler': scheduler}
        
        

# Try with l1 loss function
# Add patch discriminator 
# Add other loss function (4 in total i think refer to paper)