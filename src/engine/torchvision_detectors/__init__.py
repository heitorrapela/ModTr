import src.engine.torchvision_detectors.eval_fasterrcnn as eval_fasterrcnn
import src.engine.torchvision_detectors.eval_fcos as eval_fcos
import src.engine.torchvision_detectors.eval_retinanet as eval_retinanet
import src.engine.torchvision_detectors.eval_ssd as eval_ssd
import torchvision

def get_names():
    return ['ssd', 'fasterrcnn', 'retinanet', 'fcos']

def calculate_loss(detector, outs, targets, train_det=False, model_name='fasterrcnn'):

    if('ssd' in model_name):
        losses_det, detections = eval_ssd.eval(detector, 
                                        list(outs), 
                                        targets, 
                                        train_det=train_det, 
                                        model_name=model_name)

    elif('fasterrcnn' in model_name):
        losses_det, detections = eval_fasterrcnn.eval(detector, 
                                                      outs, targets,
                                                      train_det=train_det, 
                                                      model_name=model_name)

    elif('retinanet' in model_name):
        losses_det, detections = eval_retinanet.eval(detector, 
                                                     outs, 
                                                     targets, 
                                                     train_det=train_det, 
                                                     model_name=model_name)

    elif('fcos' in model_name):
        losses_det, detections = eval_fcos.eval(detector, 
                                                outs, 
                                                targets, 
                                                train_det=train_det, 
                                                model_name=model_name)

    return losses_det, detections



def select_torchvision_detector(detector_name='ssd300_vgg16', pretrained=True):
    
    if(detector_name == 'ssd' or detector_name == 'ssd300_vgg16'):
        return torchvision.models.detection.ssd300_vgg16(pretrained=pretrained)

    elif(detector_name == 'ssdlite' or detector_name == 'ssdlite320_mobilenetv3'):
        return torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=pretrained)

    elif(detector_name == 'fasterrcnn' or detector_name == 'fasterrcnn_resnet50_fpn'):
        return torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
        #return torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=pretrained)

    elif(detector_name == 'retinanet' or detector_name == 'retinanet_resnet50_fpn'):
        return torchvision.models.detection.retinanet_resnet50_fpn(pretrained=pretrained)
    
    elif(detector_name == 'fcos' or detector_name == 'fcos_resnet50_fpn'):
        return torchvision.models.detection.fcos_resnet50_fpn(pretrained=pretrained)
    
    else:
        print("Model Name not found (Using fasterrcnn")

    return torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)