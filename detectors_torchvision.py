import os
from src.engine.config.config import Config
import torch
Config.set_environment()

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from src.engine.utils.utils import Utils
from src.engine.logger.wandb import WandbLogger
import numpy as np

from src.engine.metrics import metrics

from src.engine.dataloader.dataloaderPL import SingleModalDataModule, MultiModalDataModule

import imageio
import skimage
import skimage.transform # lazy import
import torchvision
import wandb
from src.detector import Detector
import albumentations as alb
import albumentations.pytorch


args = Config().argument_parser()
seed_everything(args.seed)

dataset = args.dataset if args.dataset is not None else Config.Dataset.dataset
Config.set_dataset_path(dataset)

detector = args.detector if args.detector is not None else Config.Detector.name
Config.set_detector(detector, train_det=False, pretrained=True if args.directly_coco else False)

Config.set_loss_weights(args)

ext = args.ext if args.ext is not None else Config.Dataset.ext

fine_tuning = args.fine_tuning

fine_tuning_lp = args.fine_tuning_lp

pre_train_path = None if not fine_tuning \
                else args.path

# LR = (0.0001 if not fine_tuning else 0.00001) if args.lr is None else args.lr
LR = 0.0001 if args.lr is None else args.lr


# gradient_clip_val = 0.5
# gradient_clip_algorithm = "value"
# stochastic_weight_avg = True

# Config log to be logged by wandb
config = dict (
    # Hparams
    batch_size=args.batch,
    epochs=args.epochs,
    lr=LR,

    # Optimizer Params
    optimizer_name=Config.Optimizer.name, 
    lr_scheduler_step_size=Config.Optimizer.scheduler_step_size,
    lr_scheduler_gamma=Config.Optimizer.scheduler_gamma, 
    
    # Detector Params
    detector_name=Config.Detector.name,
    detector_input_size=Config.Detector.input_size,
    detector_batch_norm_eps=Config.Detector.batch_norm_eps,
    detector_batch_norm_momentum=Config.Detector.batch_norm_momentum,
    detector_pretrained=Config.Detector.pretrained,
    score_threshold=Config.Detector.score_threshold,

    
    # Train / Valid Split
    train_path=args.train,
    test_path=args.test,
    train_valid_split=Config.Dataset.train_valid_split,

    # Seed
    seed=args.seed,
    modality=args.modality,
)

wandb_logger = wandb.init(project=args.wandb_project, name=args.wandb_name, config=config, tags=args.tags)

class DetectorLit(pl.LightningModule):
    def __init__(self, batch_size=4, wandb_logger=None,
                lr=0.0001, detector_name='ssd', pretrained=True, optimizer_name='adam', modality=None, directly_coco=False, cosine_t_max=1):
        super().__init__()


        self.wandb_logger = wandb_logger
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.detector_name = detector_name
        self.cosine_t_max = cosine_t_max


        self.detector = Detector(detector=None, 
                            detector_name=detector_name, 
                            frozen=False, 
                            pretrained=True)


        ## Metrics
        self.valid_metrics_detection_map = metrics.Detection(class_metrics=True).map
        self.test_metrics_detection_map = metrics.Detection(class_metrics=True).map

        self.train_epoch = 0
        self.valid_epoch = 0

        self.train_media_step = 0
        self.valid_media_step = 0

        self.train_loss_step = 0
        self.valid_loss_step = 0

        self.test_media_step = 0
        self.test_metric_step = 0

        self.best_valid_map_50 = 0.0
        self.best_valid_epoch = 0

        self.wandb_logger.define_metric("train/loss/step")
        self.wandb_logger.define_metric("train/loss/*", step_metric="train/loss/step")

        self.wandb_logger.define_metric("train/media/step")
        self.wandb_logger.define_metric("train/media/*", step_metric="train/media/step")

        self.wandb_logger.define_metric("valid/loss/step")
        self.wandb_logger.define_metric("valid/loss/*", step_metric="valid/loss/step")

        self.wandb_logger.define_metric("valid/media/step")
        self.wandb_logger.define_metric("valid/media/*", step_metric="valid/media/step")

        self.wandb_logger.define_metric("valid/metrics/step")
        self.wandb_logger.define_metric("valid/metrics/*", step_metric="valid/metrics/step")

        self.wandb_logger.define_metric("test/media/step")
        self.wandb_logger.define_metric("test/media/*", step_metric="test/media/step")

        self.wandb_logger.define_metric("test/metrics/step")
        self.wandb_logger.define_metric("test/metrics/*", step_metric="test/metrics/step")
        

    def training_step(self, train_batch, batch_idx):
        
        if(args.modality == 'rgb' or args.modality == 'ir'):
            imgs, targets = train_batch
            
            ## 3-channel IR
            if(args.modality == 'ir' and args.dataset == 'llvip'):
                imgs = [Utils.expand_one_channel_to_output_channels(img, 3).squeeze_(0) for img in imgs]
        else:
            imgs_rgb, targets, imgs_ir, targets_ir = train_batch


        targets = Utils.batch_targets_for_detector(targets=targets, device=device, detector_name=self.detector_name)
        losses_det, detections = Detector.calculate_loss(self.detector, imgs, targets, train_det=True, model_name=self.detector_name)
        # total_loss = sum(loss for loss in losses_det.values())

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
        
        total_loss = losses_det['bbox_regression'] + losses_det['classification'] + \
                        losses_det['loss_objectness'] + \
                        losses_det['loss_rpn_box_reg'] + losses_det['bbox_ctrness']                               


        self.wandb_logger.log({ 'train/loss/det_reg': losses_det['bbox_regression'].item(),
                                'train/loss/det_class': losses_det['classification'].item(),
                                'train/loss/det_obj': losses_det['loss_objectness'].item() if 'fasterrcnn' in self.detector_name else 0.0,
                                'train/loss/det_rpn': losses_det['loss_rpn_box_reg'].item() if 'fasterrcnn' in self.detector_name else 0.0,
                                'train/loss/det_bbox_ctrness': losses_det['bbox_ctrness'].item() if 'fcos' in self.detector_name else 0.0,
                                'train/loss/total': total_loss.item(), 
                                'train/loss/step': self.train_loss_step,                                
        })
        self.train_loss_step += 1

        return total_loss

    def validation_step(self, val_batch, batch_idx):
        
        if(args.modality == 'rgb' or args.modality == 'ir'):
            imgs, targets = val_batch
            ## 3-channel IR
            if(args.modality == 'ir' and args.dataset == 'llvip'):
                imgs = [Utils.expand_one_channel_to_output_channels(img, 3).squeeze_(0) for img in imgs]
        else:
            imgs_rgb, targets, imgs_ir, targets_ir = val_batch
            
            if(args.modality == 'concat'):
                imgs = [Utils.concat_modalities(img_rgb, imgs_ir).squeeze_(0) for (img_rgb, imgs_ir) in zip(imgs_rgb, imgs_ir)]
        
        targets = Utils.batch_targets_for_detector(targets=targets, device=device, detector_name=self.detector_name)
        ## Detector
        losses_det, detections = Detector.calculate_loss(self.detector, imgs, targets, train_det=False, model_name=self.detector_name)
        
        self.valid_metrics_detection_map.update(detections, targets)

        if 'fasterrcnn' in self.detector_name:
            losses_det['classification'] = losses_det['loss_classifier']
            losses_det['bbox_regression'] = losses_det['loss_box_reg']

        total_loss = losses_det['bbox_regression'] + losses_det['classification'] + (losses_det['loss_objectness'] + losses_det['loss_rpn_box_reg'] 
                                                                                    if 'fasterrcnn' in self.detector_name else 0.0
                                                                                ) + (
                                                                                    losses_det['bbox_ctrness'] if 'fcos' in self.detector_name else 0.0
                                                                                )
        
        self.log('val_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)

        self.wandb_logger.log({'valid/loss/det_reg': losses_det['bbox_regression'].item(),
                        'valid/loss/det_class': losses_det['classification'].item(),
                        'valid/loss/det_obj': losses_det['loss_objectness'].item() if 'fasterrcnn' in self.detector_name else 0.0,
                        'valid/loss/det_rpn': losses_det['loss_rpn_box_reg'].item() if 'fasterrcnn' in self.detector_name else 0.0,
                        'valid/loss/det_bbox_ctrness': losses_det['bbox_ctrness'].item() if 'fcos' in self.detector_name else 0.0,
                        'valid/loss/total': total_loss.item(),
                        'valid/loss/step': self.valid_loss_step,
                    })
        self.valid_loss_step += 1

        return total_loss


    def on_validation_epoch_end(self):

        valid_metrics = self.valid_metrics_detection_map.compute()
        valid_map_metrics = Utils.filter_dictionary(valid_metrics, {'map_50', 'map_75', 'map', 'map_per_class'})

        self.wandb_logger.log({
                        'valid/metrics/map': valid_map_metrics, 
                        'valid/metrics/step': self.valid_epoch,
                    })

        if(self.best_valid_map_50 < valid_map_metrics['map_50'] and self.current_epoch > 0):

            self.best_valid_map_50 = valid_map_metrics['map_50']
            self.best_valid_epoch = self.current_epoch


            self.wandb_logger.summary["valid/metrics/map"] = valid_map_metrics
            self.wandb_logger.summary["valid/metrics/best_epoch"] = self.best_valid_epoch
            self.wandb_logger.summary["checkpoint_dirpath"] = self.trainer.checkpoint_callback.dirpath

            ckpt_path = os.path.join(
                self.trainer.checkpoint_callback.dirpath, 'best_encoder_decoder_pl.ckpt'
            )
            self.trainer.save_checkpoint(ckpt_path)
        

        self.log('val_map_50', valid_map_metrics['map_50'], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)

        self.valid_metrics_detection_map.reset()

        self.valid_epoch += 1

    def on_train_epoch_end(self):
        self.train_epoch += 1
        

    def test_step(self, test_batch, batch_idx):
        

        if(args.modality == 'rgb' or args.modality == 'ir'):
            imgs, targets = test_batch
        else:
            imgs_rgb, targets, imgs_ir, targets_ir = test_batch


        targets = Utils.batch_targets_for_detector(targets=targets, device=device, detector_name=self.detector_name)

        if(args.modality == 'ir'):
            imgs = list(imgs)
            
            if(args.dataset == 'llvip'):
                for idx, img in enumerate(imgs):
                    imgs[idx] = Utils.expand_one_channel_to_output_channels(img, 3).squeeze_(0)


        ## Detector
        _, detections = Detector.calculate_loss(self.detector, imgs, targets, train_det=False, model_name=self.detector_name)

        self.test_metrics_detection_map.update(detections, targets)


    def on_test_epoch_end(self):

        test_map_metrics = Utils.filter_dictionary(self.test_metrics_detection_map.compute(), {'map_50', 'map_75', 'map', 'map_per_class'})

        self.wandb_logger.log({
            'test/metrics/map': test_map_metrics, 
            'test/metrics/step': self.test_metric_step,
        })

        self.wandb_logger.summary["test/metrics/map"] = test_map_metrics

        self.test_metrics_detection_map.reset()

        self.test_metric_step += 1 


    def configure_optimizers(self):
        
        optimizer = Config().config_optimizer(optimizer=self.optimizer_name,
                                        params=(list(self.detector.parameters())),
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




if __name__ == "__main__":

    # Set device
    device = Config.cuda_or_cpu()

    # Fixed transformations
    fixed_transformations = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )

    # data augmentation
    data_augmentation = fixed_transformations

    dm = None
    if(args.modality == 'ir' or args.modality == 'rgb'):
        dm = SingleModalDataModule(
                                dataset=dataset,
                                path_images_train=Config.Dataset.train_path,
                                path_images_test=Config.Dataset.test_path, 
                                batch_size=args.batch, 
                                num_workers=args.num_workers, 
                                ext=ext,
                                seed=args.seed,
                                split_ratio_train_valid=Config.Dataset.train_valid_split,
                                modality=args.modality, 
                                data_augmentation=data_augmentation,
                                fixed_transformations=fixed_transformations,
                                ablation_flag=args.ablation_flag,
        )
    elif(args.modality == 'both' or args.modality == 'concat' or args.modality == 'fuse'):
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

    cosine_t_max = args.epochs * len(dm.train_dataloader())


    # Model
    model = DetectorLit(batch_size=args.batch, 
                wandb_logger=wandb_logger,
                lr=LR, 
                detector_name=Config.Detector.name, 
                pretrained=Config.Detector.pretrained,
                optimizer_name=Config.Optimizer.name,
                modality=args.modality,
                directly_coco=args.directly_coco,
                cosine_t_max=cosine_t_max,
                )

    # saves best model
    checkpoint_best_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor="val_map_50",
        mode="max",
        dirpath=os.path.join('lightning_logs', args.wandb_project, args.wandb_name, "_".join([args.dataset, args.modality, Config.Detector.name])),
        filename="best",
    )

    # Save last model
    checkpoint_last_callback = pl.callbacks.ModelCheckpoint(
        save_last=True,
        dirpath=os.path.join('lightning_logs', args.wandb_project, args.wandb_name, "_".join([args.dataset, args.modality, Config.Detector.name])),
        filename="last"
    )

    # Training
    trainer = pl.Trainer(gpus=Config.Environment.N_GPUS,
                        accelerator="gpu",
                        max_epochs=args.epochs,
                        callbacks=[
                            pl.callbacks.RichProgressBar(),
                            pl.callbacks.EarlyStopping(monitor="val_map_50", mode="max", patience=5),
                            checkpoint_best_callback,
                            checkpoint_last_callback
                        ],

                        deterministic=True,
                        accumulate_grad_batches=2,
                        limit_train_batches=args.limit_train_batches,
                        num_sanity_val_steps=0, # debug
                        precision=args.precision, # 32 default
                        enable_model_summary=True,
                        logger=False,
                        )
    

    trainer.fit(model, dm)

    trainer.save_checkpoint(os.path.join(trainer.checkpoint_callback.dirpath,
                                        'detector_pl.ckpt'))

    torch.save(model.detector.state_dict(),  
                os.path.join(trainer.checkpoint_callback.dirpath, 'detector.bin')
    )

    trainer.test(model, dm, ckpt_path="best")

    wandb_logger.summary["checkpoint_dirpath"] = trainer.checkpoint_callback.dirpath

    wandb_logger.finish()