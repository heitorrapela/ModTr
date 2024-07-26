import os
from src.engine.config.config import Config
import torch
Config.set_environment()

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from src.engine.utils.utils import Utils
from src.engine.logger.wandb import WandbLogger

from src.engine.losses import losses
from src.engine.metrics import metrics
from src.engine.dataloader.dataloaderPL import MultiModalDataModule
import albumentations as alb
import albumentations.pytorch
from src.modtr import ModTr
from src.translator import Translator
from src.detector import Detector
import numpy as np


# True = Speed-up but not deterministic
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

args = Config.argument_parser()
seed_everything(args.seed)

dataset = args.dataset if args.dataset is not None else Config.Dataset.dataset
Config.set_dataset_path(dataset)

detector = args.detector if args.detector is not None else Config.Detector.name
Config.set_detector(detector, train_det=False, pretrained=args.directly_coco)

Config.set_loss_weights(args)

ext = args.ext if args.ext is not None else Config.Dataset.ext

wandb_logger = WandbLogger(args)

args.modtr_backbone = args.decoder_backbone if args.decoder_backbone is not None else Config.EncoderDecoder.decoder_backbone
args.modtr_encoder_depth = args.decoder_encoder_depth if args.decoder_encoder_depth is not None else Config.EncoderDecoder.decoder_encoder_depth
args.modtr_encoder_weights = args.decoder_encoder_weights if args.decoder_encoder_weights is not None else Config.EncoderDecoder.decoder_encoder_weights
args.modtr_in_channels = args.decoder_in_channels if args.decoder_in_channels is not None else Config.EncoderDecoder.decoder_in_channels
args.modtr_out_channels = args.decoder_out_channels if args.decoder_out_channels is not None else Config.EncoderDecoder.decoder_out_channels


pre_train_path = None
if Config.EncoderDecoder.load_encoder_decoder:
    pre_train_path = args.pre_train_path if args.pre_train_path is not None else Config.EncoderDecoder.encoder_decoder_load_path

LR = 0.0001 if args.lr is None else args.lr


class EncoderDecoderLit(pl.LightningModule):
    def __init__(self, batch_size=4, 
                wandb_logger=None, model_name='resnet34',
                in_channels=1, output_channels=3,  lr=0.0001,
                loss_pixel='mse', loss_perceptual='lpips_alexnet', 
                detector_name='ssd', train_det=False, fuse_data='none', scheduler_on=False, cosine_t_max=1):
        super().__init__()

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        # self.save_hyperparameters()

        self.model_name = model_name
        args.modtr_backbone = self.model_name
        
        self.wandb_logger = wandb_logger
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.lr = lr
        self.batch_size = batch_size
        self.train_det = train_det
        self.fuse_data = fuse_data
        self.optimizer_name = Config.Optimizer.name
        self.segmentation_head = Config.EncoderDecoder.decoder_head
        self.scheduler_on = scheduler_on
        self.cosine_t_max = cosine_t_max


        self.modtr = ModTr(args=args, 
                            translator=Translator(args=args,
                                                translator=None), 
                            detector=Detector(detector=None, 
                                            detector_name=detector_name, 
                                            frozen=True, 
                                            pretrained=True))


        ## Detector
        self.detector_name = detector_name

        self.detector = self.modtr.detector.detector

        self.encoder_decoder = self.modtr.translator

        self.loss_pixel = losses.Reconstruction.select_loss_pixel(loss_pixel=loss_pixel)

        ## Metrics
        self.train_metrics_detection_map_modtr = metrics.Detection().map
        self.valid_metrics_detection_map_modtr = metrics.Detection().map
        self.test_metrics_detection_map_modtr = metrics.Detection().map

        self.train_metrics_detection_map_rgb = metrics.Detection().map
        self.valid_metrics_detection_map_rgb = metrics.Detection().map
        self.test_metrics_detection_map_rgb = metrics.Detection().map

        self.train_metrics_detection_map_ir = metrics.Detection().map
        self.valid_metrics_detection_map_ir = metrics.Detection().map
        self.test_metrics_detection_map_ir = metrics.Detection().map

        self.valid_epoch = 0

        self.best_valid_map_50 = 0.0
        self.best_valid_epoch = 0

       
    def forward_step(self, imgs_rgb, targets_rgb, imgs_ir, targets_ir, batch_idx, step='train'):

        imgs_ir = Utils.batch_images_for_encoder_decoder(imgs=imgs_ir, device=device, ablation_flag=args.ablation_flag)
        imgs_rgb = Utils.batch_images_for_encoder_decoder(imgs=imgs_rgb, device=device, ablation_flag=args.ablation_flag)

        targets_rgb = Utils.batch_targets_for_detector(targets=targets_rgb, device=device, detector_name=self.detector_name)
        targets_ir = Utils.batch_targets_for_detector(targets=targets_ir, device=device, detector_name=self.detector_name)
        
        ## Encoder / Decoder
        imgs_ir_three_channel = Utils.expand_one_channel_to_output_channels(imgs_ir, self.output_channels)
        imgs_translated = self.encoder_decoder(imgs_ir_three_channel)
        imgs_translated = Utils.fusion_data(imgs_translated, imgs_ir_three_channel, fuse_data=self.fuse_data)
        
        loss_pixel_rgb = 0.0 if self.loss_pixel == None else self.loss_pixel(imgs_rgb, imgs_translated) * Config.Losses.hparams_losses_weights['pixel_rgb']
        loss_pixel_ir = 0.0 if self.loss_pixel == None else self.loss_pixel(imgs_ir_three_channel, imgs_translated) * Config.Losses.hparams_losses_weights['pixel_ir']
    
        ## Detector Translated
        train_det = True if (self.train_det == True and step == 'train') else False
        losses_det, detections_trans = Detector.calculate_torchvision_detectors_loss(self.detector, imgs_translated, targets_ir, train_det=train_det, model_name=self.detector_name)

        ## Detector RGB
        _, detections_rgb = Detector.calculate_torchvision_detectors_loss(self.detector, imgs_rgb, targets_rgb, train_det=train_det, model_name=self.detector_name)

        ## Detector IR
        _, detections_ir = Detector.calculate_torchvision_detectors_loss(self.detector, imgs_ir_three_channel, targets_ir, train_det=train_det, model_name=self.detector_name)


        if 'fasterrcnn' in self.detector_name:
            losses_det['classification'] = losses_det['loss_classifier']
            losses_det['bbox_regression'] = losses_det['loss_box_reg']

        losses_det['bbox_regression'] = losses_det['bbox_regression'] * Config.Losses.hparams_losses_weights['det_regression']
        losses_det['classification'] = losses_det['classification'] * Config.Losses.hparams_losses_weights['det_classification']
        
        losses_det['loss_objectness'] = (losses_det['loss_objectness'] *  Config.Losses.hparams_losses_weights['det_objectness']
                                            if 'fasterrcnn' in self.detector_name else 0.0)
        losses_det['loss_rpn_box_reg'] = (losses_det['loss_rpn_box_reg'] * Config.Losses.hparams_losses_weights['det_rpn_box_reg']
                                            if 'fasterrcnn' in self.detector_name else 0.0)
        losses_det['bbox_ctrness'] = (losses_det['bbox_ctrness'] * Config.Losses.hparams_losses_weights['det_bbox_ctrness'] 
                                            if 'fcos' in self.detector_name else 0.0)
        
        loss_det_total = losses_det['bbox_regression'] + losses_det['classification'] + \
                        losses_det['loss_objectness'] + \
                        losses_det['loss_rpn_box_reg'] + losses_det['bbox_ctrness']                               


        ## Total Loss
        total_loss = loss_det_total + loss_pixel_ir

        if step == 'val':

            self.valid_metrics_detection_map_rgb.update(detections_rgb, targets_rgb)
            self.valid_metrics_detection_map_modtr.update(detections_trans, targets_ir)
            self.valid_metrics_detection_map_ir.update(detections_ir, targets_ir)

        # Normalize for plotting
        imgs_translated = Utils.normalize_batch_images(imgs_translated.detach().clone())

        return {
            'loss' : {'total': total_loss, 
                'pixel_rgb': loss_pixel_rgb, 
                'pixel_ir': loss_pixel_ir,
                'det_regression': losses_det['bbox_regression'],
                'det_classification': losses_det['classification'],
                'det_objectness': losses_det['loss_objectness'],
                'det_rpn_box_reg': losses_det['loss_rpn_box_reg'],
                'det_bbox_ctrness': losses_det['bbox_ctrness'],
                'det_total': loss_det_total,
                },
            'output': { # 'det_modtr': output_modtr_det, 
                        # 'det_rgb': output_rgb_det, 
                        # 'det_ir': output_ir_det,
                        'imgs_rgb': imgs_rgb,
                        'imgs_ir': imgs_ir,
                        'imgs_translated': imgs_translated,
                    }
        }

    def training_step(self, train_batch, batch_idx):

        imgs_rgb, targets_rgb, imgs_ir, targets_ir = train_batch

        forward_return =  self.forward_step(imgs_rgb, targets_rgb, imgs_ir, targets_ir, batch_idx, step='train')
        
        self.log('train_loss', forward_return['loss']['total'], 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True, 
                 batch_size=self.batch_size)
        
        self.wandb_logger.training_loss_log(forward_return)
        
        return forward_return['loss']['total']


    def validation_step(self, val_batch, batch_idx):

        imgs_rgb, targets_rgb, imgs_ir, targets_ir = val_batch

        forward_return =  self.forward_step(imgs_rgb, targets_rgb, imgs_ir, targets_ir, batch_idx, step='val')
        
        self.log('val_loss', forward_return['loss']['total'], 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True, 
                 batch_size=self.batch_size)

        self.wandb_logger.valid_loss_log(forward_return)

        return forward_return['loss']['total']


    def on_validation_epoch_end(self):
        
        map_rgb = Utils.filter_dictionary(self.valid_metrics_detection_map_rgb.compute(), {'map_50', 'map_75', 'map', 'map_per_class'})
        map_modtr = Utils.filter_dictionary(self.valid_metrics_detection_map_modtr.compute(), {'map_50', 'map_75', 'map', 'map_per_class'})
        map_ir = Utils.filter_dictionary(self.valid_metrics_detection_map_ir.compute(), {'map_50', 'map_75', 'map', 'map_per_class'})
        
        self.wandb_logger.valid_metrics_log(map_rgb, map_modtr, map_ir, self.valid_epoch)

        if(self.best_valid_map_50 < map_modtr['map_50'] and self.current_epoch > 0):

            self.best_valid_map_50 = map_modtr['map_50']
            self.best_valid_epoch = self.current_epoch
            
            self.wandb_logger.valid_summary_update(map_rgb, map_modtr, map_ir, 
                                                   self.best_valid_epoch, 
                                                   self.trainer.checkpoint_callback.dirpath)

            ckpt_path = os.path.join(
                self.trainer.checkpoint_callback.dirpath, 'best_encoder_decoder_pl.ckpt'
            )
            self.trainer.save_checkpoint(ckpt_path)

        self.log('val_map_50', map_modtr['map_50'], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)

        self.valid_metrics_detection_map_rgb.reset()
        self.valid_metrics_detection_map_modtr.reset()
        self.valid_metrics_detection_map_ir.reset()

        self.valid_epoch += 1


    def test_step(self, test_batch, batch_idx):
        
        imgs_rgb, targets_rgb, imgs_ir, targets_ir = test_batch

        imgs_ir = Utils.batch_images_for_encoder_decoder(imgs=imgs_ir, device=device, ablation_flag=args.ablation_flag)
        imgs_rgb = Utils.batch_images_for_encoder_decoder(imgs=imgs_rgb, device=device, ablation_flag=args.ablation_flag)
        
        targets_rgb = Utils.batch_targets_for_detector(targets=targets_rgb, device=device, detector_name=self.detector_name)
        targets_ir = Utils.batch_targets_for_detector(targets=targets_ir, device=device, detector_name=self.detector_name)

        imgs_ir_three_channel = Utils.expand_one_channel_to_output_channels(imgs_ir, self.output_channels)
        imgs_translated = self.encoder_decoder(imgs_ir_three_channel)
        imgs_translated = Utils.fusion_data(imgs_translated, imgs_ir_three_channel, fuse_data=self.fuse_data)

        imgs_rgb = imgs_rgb.float()

        _, detections_trans = Detector.calculate_torchvision_detectors_loss(self.detector, imgs_translated, targets_ir, train_det=False, model_name=self.detector_name)

        ## Detector RGB
        _, detections_rgb = Detector.calculate_torchvision_detectors_loss(self.detector, imgs_rgb, targets_rgb, train_det=False, model_name=self.detector_name)

        ## Detector IR
        _, detections_ir = Detector.calculate_torchvision_detectors_loss(self.detector, imgs_ir_three_channel, targets_ir, train_det=False, model_name=self.detector_name)


        self.test_metrics_detection_map_rgb.update(detections_rgb, targets_rgb)
        self.test_metrics_detection_map_modtr.update(detections_trans, targets_ir)
        self.test_metrics_detection_map_ir.update(detections_ir, targets_ir)


    def on_test_epoch_end(self):

        map_rgb = Utils.filter_dictionary(self.test_metrics_detection_map_rgb.compute(), {'map_50', 'map_75', 'map', 'map_per_class'})
        map_modtr = Utils.filter_dictionary(self.test_metrics_detection_map_modtr.compute(), {'map_50', 'map_75', 'map', 'map_per_class'})
        map_ir = Utils.filter_dictionary(self.test_metrics_detection_map_ir.compute(), {'map_50', 'map_75', 'map', 'map_per_class'})
    
        self.wandb_logger.test_summary_update(map_rgb, map_modtr, map_ir)


    def configure_optimizers(self):
        
        optimizer = Config().config_optimizer(optimizer=self.optimizer_name,
                                        params=(list(self.encoder_decoder.parameters())),
                                        lr=self.lr)

        sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                        T_max=self.cosine_t_max)

        return {
            "optimizer": optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "train_loss",
                "interval": "step",
                "frequency": 1,
            }
        }

# Set device
device = Config.cuda_or_cpu() if args.device is None else args.device

# Fixed transformations
fixed_transformations = alb.Compose(
    [
    alb.pytorch.ToTensorV2(),
    ]
)

# data augmentation
data_augmentation = alb.Compose(
    [fixed_transformations],  
    bbox_params=alb.BboxParams(format='pascal_voc', label_fields=['labels']), 
    additional_targets={'image1': 'image', 'bboxes1': 'bboxes', 'labels1': 'labels'}, p=1.0
 )

dm = MultiModalDataModule(
                        dataset=dataset,
                        path_images_train_rgb=Config.Dataset.train_path,
                        path_images_train_ir=Config.Dataset.train_path,
                        path_images_test_rgb=Config.Dataset.test_path, 
                        path_images_test_ir=Config.Dataset.test_path,
                        batch_size=args.batch, 
                        num_workers=args.num_workers, 
                        ext=ext,
                        seed=args.seed,
                        split_ratio_train_valid=Config.Dataset.train_valid_split,
                        data_augmentation=data_augmentation,
                        fixed_transformations=fixed_transformations,
                        ablation_flag=args.ablation_flag,
                        )


# Model
model = EncoderDecoderLit(batch_size=args.batch, 
                        wandb_logger=wandb_logger,
                        model_name=args.decoder_backbone, 
                        in_channels=Config.EncoderDecoder.in_channels_encoder,
                        output_channels=Config.EncoderDecoder.out_channels_decoder, 
                        lr=LR,
                        loss_pixel=Config.Losses.pixel, 
                        loss_perceptual=Config.Losses.perceptual,
                        detector_name=Config.Detector.name,
                        train_det=Config.Detector.train_det,
                        fuse_data=args.fuse_data,
                        scheduler_on=Config.Optimizer.scheduler_on,
                        cosine_t_max=args.epochs * len(dm.train_dataloader()),
                        )


if(Config.EncoderDecoder.load_encoder_decoder):
    model = EncoderDecoderLit.load_from_checkpoint(checkpoint_path=Config.EncoderDecoder.encoder_decoder_load_path,
                                                batch_size=args.batch, 
                                                wandb_logger=wandb_logger,
                                                model_name=args.decoder_backbone, 
                                                in_channels=Config.EncoderDecoder.in_channels_encoder,
                                                output_channels=Config.EncoderDecoder.out_channels_decoder,
                                                lr=LR,
                                                loss_pixel=Config.Losses.pixel, 
                                                loss_perceptual=Config.Losses.perceptual,
                                                detector_name=Config.Detector.name,
                                                train_det=Config.Detector.train_det,
                                                fuse_data=args.fuse_data,
                                                scheduler_on=Config.Optimizer.scheduler_on,
                                                strict=False,
                                                cosine_t_max=args.epochs * len(dm.train_dataloader()),
                                            )

# saves best model
checkpoint_best_callback = pl.callbacks.ModelCheckpoint(
    save_top_k=1,
    monitor="val_map_50",
    mode="max",
    dirpath=os.path.join('lightning_logs', 
                         args.wandb_project, 
                         args.wandb_name, "_".join([args.dataset, args.modality, Config.Detector.name])),
    filename="best",
)

# Training
trainer = pl.Trainer(
                    gpus=Config.Environment.N_GPUS,
                    accelerator="gpu",
                    max_epochs=args.epochs,
                    callbacks=[
                            pl.callbacks.RichProgressBar(),
                            pl.callbacks.EarlyStopping(monitor="val_map_50", mode="max", patience=5),
                            checkpoint_best_callback,
                    ],
                    accumulate_grad_batches=2,
                    limit_train_batches=args.limit_train_batches,
                    num_sanity_val_steps=0, # debug
                    enable_model_summary=True,
                    logger=False,
                    )


trainer.fit(model, dm)

trainer.save_checkpoint(os.path.join(trainer.checkpoint_callback.dirpath,
                                    'encoder_decoder_pl.ckpt'))

torch.save(model.detector.state_dict(),  
            os.path.join(trainer.checkpoint_callback.dirpath, 'detector.bin')
)

torch.save(model.encoder_decoder.state_dict(),  
            os.path.join(trainer.checkpoint_callback.dirpath, 'encoder_decoder.bin')
)

trainer.test(model, dm, ckpt_path="best")

wandb_logger.finish(dirpath=trainer.checkpoint_callback.dirpath)