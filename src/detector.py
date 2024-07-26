import torch
import src.engine.torchvision_detectors as torchvision_detectors

class Detector(torch.nn.Module):
    
    def __init__(self, detector=None, detector_name=None, frozen=False, pretrained=True):
        super().__init__()
                
        if(detector is None or detector_name in torchvision_detectors.get_names()):
            
            self.detector = torchvision_detectors.select_torchvision_detector(detector_name='fasterrcnn' if (detector is None and detector_name is None) else 
                                                                            detector_name,
                                                                            pretrained=pretrained)
        else:
            self.detector = detector
            
        self.frozen = frozen
        
        if self.frozen:
            self.freeze()
        else:
            self.unfreeze()
            
    
    def freeze(self):
        for param in self.detector.parameters():
            param.requires_grad = False
            
            
    def unfreeze(self):
        for param in self.detector.parameters():
            param.requires_grad = True
            
            
    def get_parameters(self):
        return self.detector.parameters()
    
    
    def forward(self, x):
        return self.detector(x)
    
    
    ## You should implement the loss of your detector here
    def calculate_loss(self, outs, targets, train_det, model_name):
        return Detector.calculate_torchvision_detectors_loss(detector=self.detector, 
                                                             outs=outs, 
                                                             targets=targets, 
                                                             train_det=train_det, 
                                                             model_name=model_name)
        


    @staticmethod
    def calculate_torchvision_detectors_loss(detector, outs, targets, train_det, model_name):
        losses_det, detections = torchvision_detectors.calculate_loss(detector=detector, 
                                                                outs=outs, 
                                                                targets=targets, 
                                                                train_det=train_det,
                                                                model_name=model_name)
        return losses_det, detections