from src.detector import Detector
from src.translator import Translator
import torch


class ModTr(torch.nn.Module):
    
    def __init__(self, args=None, translator=None, detector=None):
        
        super().__init__()
        
        self.translator = translator if translator is not None else Translator(frozen=False)
        self.detector = detector if detector is not None else Detector(detector=None, 
                                                                       detector_name=None, 
                                                                       frozen=True, 
                                                                       pretrained=True
                                                                       )
        
        self.detector.freeze()
        self.detector.eval()
                    

    def forward(self, x, modality='ir'):
        
        if(modality == 'ir'):
            x = self.translator(x)
            x = self.detector(x)
        elif(modality == 'rgb'):
            x = self.detector(x)
        
        return x