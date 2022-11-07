import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision
import timm

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.transformer = nn.Sequential(
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize([0.0], [1.0])
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
    
    def forward(self, x):

        return self.encoder(self.transformer(x))

class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.transformer = nn.Sequential(
            T.Resize((299, 299)),
        )
        self.inception = timm.create_model('inception_resnet_v2', pretrained=True)
        for layer in self.inception.parameters():
            layer.requires_grad = False
        # print(self.inception._modules)
    
    def forward(self, x):
        x = torch.cat((x, x, x), dim=1)
        # print(x.shape)
        return self.inception(self.transformer(x))

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            torch.nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),      
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            torch.nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            torch.nn.Upsample(scale_factor=2),
        )
    
    def forward(self, x):
        return self.decoder(x)

class Koala(nn.Module):
    def __init__(self):
        super(Koala, self).__init__()
        self.encoder = Encoder()

        self.feature_extractor = Inception()
        self.fusion = nn.Conv2d(1256, 256, kernel_size=1, stride=1, padding=0)
        self.bnorm = nn.BatchNorm2d(256)
        self.decoder = Decoder()

    def forward(self, x):
        z_layer = self.encoder(x)  
        features = self.feature_extractor(x)
        features = features.reshape(-1, 1000, 1, 1)   
        features = features.repeat(1, 1, 28, 28)

        fusion_layer = torch.cat((z_layer, features), dim=1)
        x = self.fusion(fusion_layer)
        x = self.bnorm(x)
        x = self.decoder(x)
        return x
        


def test():
    x = torch.rand((10, 1, 224, 224))
    model = Koala()
    preds = model(x)
    print(preds.shape)


    # assert preds.shape == x.shape


if __name__=="__main__":
    test()