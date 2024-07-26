import wandb

class WandbLogger():
    def __init__(self, args):
        super().__init__()
        
        self.experiment = wandb.init(project=args.wandb_project, 
                              name=args.wandb_name, 
                              tags=args.tags)
        
        splits = ['train', 'valid']
        types = ['loss', 'media', 'metrics']
        
        for split in splits:
            for type in types:                
                self.experiment.define_metric(f"{split}/{type}/step")
                self.experiment.define_metric(f"{split}/{type}/*", step_metric=f"{split}/{type}/step")
        
        self.train_loss_step = 0        
        self.train_media_step = 0
    
        self.valid_loss_step = 0
        self.valid_media_step = 0

    def training_loss_log(self, log_dict):
        self.experiment.log({ 'train/loss/pixel_rgb': log_dict['loss']['pixel_rgb'], 
                                'train/loss/pixel_ir': log_dict['loss']['pixel_ir'],
                                'train/loss/det_reg': log_dict['loss']['det_regression'],
                                'train/loss/det_class': log_dict['loss']['det_classification'],
                                'train/loss/det_objectness': log_dict['loss']['det_objectness'],
                                'train/loss/det_rpn_box_reg': log_dict['loss']['det_rpn_box_reg'],
                                'train/loss/det_bbox_ctrness': log_dict['loss']['det_bbox_ctrness'],
                                'train/loss/det_total': log_dict['loss']['det_total'],
                                'train/loss/total': log_dict['loss']['total'], 
                                'train/loss/step': self.train_loss_step,
        })
        self.train_loss_step = self.train_loss_step + 1

    
    def training_media_log(self, log_dict, log_frequency):
        # if(log_frequency == 1):
                
        self.experiment.log({"train/media/input_ir": [wandb.Image(log_dict['output']['imgs_ir'], caption="train/input_ir")],
                        "train/media/input_rgb": [wandb.Image(log_dict['output']['imgs_rgb'], caption="train/input_rgb")],
                        "train/media/output_modtr": [wandb.Image(log_dict['output']['imgs_translated'], caption="train/output")],
                        "train/media/output_modtr_det": [wandb.Image(log_dict['output']['det_modtr'], caption="train/output_modtr_det")],
                        "train/media/output_rgb_det": [wandb.Image(log_dict['output']['det_rgb'], caption="train/output_rgb_det")],
                        "train/media/output_ir_det": [wandb.Image(log_dict['output']['det_ir'], caption="train/output_ir_det")],
                        "train/media/input_ir_samples": [wandb.Image(im) for im in log_dict['output']['imgs_ir']],
                        "train/media/input_rgb_samples": [wandb.Image(im) for im in log_dict['output']['imgs_rgb']],
                        "train/media/output_modtr_samples": [wandb.Image(im) for im in log_dict['output']['imgs_translated']],
                        "train/media/output_modtr_det_samples": [wandb.Image(im) for im in log_dict['output']['det_modtr']],
                        "train/media/output_rgb_det_samples": [wandb.Image(im) for im in log_dict['output']['det_rgb']],
                        "train/media/output_ir_det_samples": [wandb.Image(im) for im in log_dict['output']['det_ir']],
                        "train/media/step" : self.train_media_step,
                    })
        self.train_media_step = self.train_media_step + 1
    
    
    
    def valid_loss_log(self, log_dict):
        self.experiment.log({ 'valid/loss/pixel_rgb': log_dict['loss']['pixel_rgb'], 
                    'valid/loss/pixel_ir': log_dict['loss']['pixel_ir'],
                    'valid/loss/det_reg': log_dict['loss']['det_regression'],
                    'valid/loss/det_class': log_dict['loss']['det_classification'],
                    'valid/loss/det_objectness': log_dict['loss']['det_objectness'],
                    'valid/loss/det_rpn_box_reg': log_dict['loss']['det_rpn_box_reg'],
                    'valid/loss/det_bbox_ctrness': log_dict['loss']['det_bbox_ctrness'],
                    'valid/loss/det_total': log_dict['loss']['det_total'],
                    'valid/loss/total': log_dict['loss']['total'],
                    'valid/loss/step': self.valid_loss_step,
                })
        self.valid_loss_step =  self.valid_loss_step + 1
    
    
    def valid_media_log(self, log_dict, log_frequency):
        #if(log_frequency == 1):
                
        self.experiment.log({"valid/media/input_ir": [wandb.Image(log_dict['output']['imgs_ir'], caption="valid/input_ir")],
                "valid/media/input_rgb": [wandb.Image(log_dict['output']['imgs_rgb'], caption="valid/input_rgb")],
                "valid/media/output_modtr": [wandb.Image(log_dict['output']['imgs_translated'], caption="valid/output_modtr")], 
                "valid/media/output_modtr_det": [wandb.Image(log_dict['output']['det_modtr'], caption="valid/output_modtr_det")],
                "valid/media/output_rgb_det": [wandb.Image(log_dict['output']['det_rgb'], caption="valid/output_rgb_det")],
                "valid/media/output_ir_det": [wandb.Image(log_dict['output']['det_ir'], caption="valid/output_ir_det")],
                "valid/media/input_ir_samples": [wandb.Image(im) for im in log_dict['output']['imgs_ir']],
                "valid/media/input_rgb_samples": [wandb.Image(im) for im in log_dict['output']['imgs_rgb']],
                "valid/media/output_modtr_samples": [wandb.Image(im) for im in log_dict['output']['imgs_translated']],
                "valid/media/output_modtr_det_samples": [wandb.Image(im) for im in log_dict['output']['det_modtr']],
                "valid/media/output_rgb_det_samples": [wandb.Image(im) for im in log_dict['output']['det_rgb']],
                "valid/media/output_ir_det_samples": [wandb.Image(im) for im in log_dict['output']['det_ir']],
                "valid/media/step" : self.valid_media_step,
            })
        self.valid_media_step = self.valid_media_step + 1


    def valid_metrics_log(self, map_rgb, map_modtr, map_ir, step):
        self.experiment.log({
                    "valid/metrics/map_rgb": map_rgb,
                    "valid/metrics/map_modtr": map_modtr,
                    "valid/metrics/map_ir": map_ir,
                    "valid/metrics/step": step,
        })
        
        
    def valid_summary_update(self, map_rgb, map_modtr, map_ir, best_epoch, dirpath):
        self.experiment.summary["valid/metrics/map_rgb"] = map_rgb
        self.experiment.summary["valid/metrics/map_modtr"] = map_modtr
        self.experiment.summary["valid/metrics/map_ir"] = map_ir
        self.experiment.summary["valid/metrics/best_epoch"] = best_epoch
        self.experiment.summary["checkpoint_dirpath"] = dirpath


    def test_summary_update(self, map_rgb, map_modtr, map_ir):
        self.experiment.summary["test/metrics/map_rgb"] = map_rgb
        self.experiment.summary["test/metrics/map_modtr"] = map_modtr
        self.experiment.summary["test/metrics/map_ir"] = map_ir

        self.experiment.log({
                        'test/metrics/map_rgb': map_rgb,
                        'test/metrics/map_modtr': map_modtr,
                        'test/metrics/map_ir': map_ir,
                    })


    def valid_metrics_log(self, map_rgb, map_modtr, map_ir, step):
        self.experiment.log({
                    "valid/metrics/map_rgb": map_rgb,
                    "valid/metrics/map_modtr": map_modtr,
                    "valid/metrics/map_ir": map_ir,
                    "valid/metrics/step": step,
        })


    def test_media_log(self, imgs_rgb, imgs_ir, outs): # , log_frequency):
        #if(log_frequency == 1):
            
        self.experiment.log({"test/media/input_ir": [wandb.Image(imgs_ir, caption="test/input_ir")],
                                "test/media/input_rgb": [wandb.Image(imgs_rgb, caption="test/input_rgb")],
                                "test/media/output": [wandb.Image(outs, caption="test/output")],
                                "test/media/input_ir_samples": [wandb.Image(im) for im in imgs_ir],
                                "test/media/input_rgb_samples": [wandb.Image(im) for im in imgs_rgb],
                                "test/media/output_samples": [wandb.Image(im) for im in outs],
                            })


    def finish(self, dirpath):
        self.experiment.summary["checkpoint_dirpath"] = dirpath
        self.experiment.finish()
        
    
if __name__ == "__main__":
    args=None
    WandbLogger(args)