import torch
try:
    import segmentation_models_pytorch as smp
except ImportError:
    print('Error importing segmentation_models_pytorch')
    print('Please install the package using pip install segmentation_models_pytorch==0.3.3')
    raise


class Translator(torch.nn.Module):
    
    def __init__(self, args, translator=None, frozen=False, pretrained=True):
        
        super().__init__()
        
        if(hasattr(args, 'modtr_backbone') is None):
            args.modtr_backbone = 'resnet34'
        if(hasattr(args, 'modtr_encoder_depth') is None):
            args.modtr_encoder_depth = 5
        if(hasattr(args, 'modtr_in_channels') is None):
            args.modtr_in_channels = 3
        if(hasattr(args, 'modtr_out_channels') is None):
            args.modtr_out_channels = 3
        
        self.translator = translator if translator is not None else smp.Unet(encoder_name=args.modtr_backbone,
                                                                                encoder_depth=args.modtr_encoder_depth,
                                                                                encoder_weights=args.modtr_encoder_weights if pretrained else None,
                                                                                in_channels=args.modtr_in_channels,
                                                                                classes=args.modtr_out_channels)
        self.translator.segmentation_head[-1] = torch.nn.Sigmoid()
        
        self.frozen = frozen
        
        if self.frozen:
            self.freeze()
        else:
            self.unfreeze()
            
    
    def freeze(self):
        for param in self.translator.parameters():
            param.requires_grad = False
            
            
    def unfreeze(self):
        for param in self.translator.parameters():
            param.requires_grad = True
            
            
    def get_parameters(self):
        return self.translator.parameters()
    
    
    def forward(self, x):
        return self.translator(x)