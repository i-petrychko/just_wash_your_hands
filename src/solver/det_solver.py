'''
by lyuwenyu
'''
import time 
import json
import datetime

import torch 

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg 
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler, wandb_logger=self.wandb_logger)

            self.lr_scheduler.step()
        
                

            module = self.ema.module if self.ema else self.model
            val_stats, val_coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            test_stats, test_coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.test_dataloader, base_ds, self.device, self.output_dir
            )

            if self.output_dir:
                # Save latest checkpoint
                latest_path = self.output_dir / 'checkpoint_latest.pth'
                dist.save_on_master(self.state_dict(epoch), latest_path)
                if self.wandb_logger is not None:
                    self.wandb_logger.upload_checkpoint(str(latest_path))
                
                # Save periodic checkpoint
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_path = self.output_dir / f'checkpoint{epoch:04}.pth'
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)
                    
                    if self.wandb_logger is not None:
                        self.wandb_logger.upload_checkpoint(str(checkpoint_path))
                
                # Save best checkpoint based on AP
                if val_stats is not None and 'bbox' in val_coco_evaluator.coco_eval:
                    current_map = val_coco_evaluator.coco_eval['bbox'].stats[0]
                    if not hasattr(self, 'best_map') or current_map > self.best_map:
                        self.best_map = current_map
                        best_path = self.output_dir / 'checkpoint_best.pth'
                        dist.save_on_master(self.state_dict(epoch), best_path)
                        if self.wandb_logger is not None:
                            self.wandb_logger.upload_checkpoint(str(best_path))

            # TODO 
            for k in val_stats.keys():
                if k in best_stat:
                    best_stat['epoch'] = epoch if val_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], val_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = val_stats[k][0]
            print('best_stat: ', best_stat)
            
            # Log metrics to wandb
            if self.wandb_logger is not None:
                # Log training metrics
                wandb_train_stats = {f"Train/{k}": v for k, v in train_stats.items() if k in ['loss', 'loss_vfl', 'loss_bbox', 'loss_giou']}
                self.wandb_logger.log_metrics(wandb_train_stats, step=epoch)

                wandb_lr_stats = {f"LR/{k}": v for k, v in train_stats.items() if k in ['lr']}
                self.wandb_logger.log_metrics(wandb_lr_stats, step=epoch)


                # Log validation metrics
                if val_stats is not None:

                    if 'bbox' in val_coco_evaluator.coco_eval:
                        bbox_eval = val_coco_evaluator.coco_eval['bbox']
                        stats = bbox_eval.stats
                        
                        # Log mAP at different IoU thresholds
                        # Todo investigate metrics
                        wandb_coco_stats = {
                            'Val/mAP@[0.50:0.95]': stats[0],  # Mean Average Precision (IoU=0.50:0.95, averaged over all classes)
                            'Val/mAP@0.50': stats[1],  # AP at IoU=0.50 (PASCAL VOC metric)
                            'Val/mAP@0.75': stats[2],  # AP at IoU=0.75 (strict metric)
                            'Val/mAP_small': stats[3],  # AP for small objects (across all classes)
                            'Val/mAP_medium': stats[4],  # AP for medium objects (across all classes)
                            'Val/mAP_large': stats[5],  # AP for large objects (across all classes)
                            
                            'Val/mAR@1': stats[6],  # Mean Average Recall at max 1 detection per image
                            'Val/mAR@10': stats[7],  # Mean Average Recall at max 10 detections per image
                            'Val/mAR@100': stats[8],  # Mean Average Recall at max 100 detections per image (COCO standard)
                            'Val/mAR_small': stats[9],  # AR for small objects (across all classes)
                            'Val/mAR_medium': stats[10],  # AR for medium objects (across all classes)
                            'Val/mAR_large': stats[11],  # AR for large objects (across all classes)
                        }
                        self.wandb_logger.log_metrics(wandb_coco_stats, step=epoch)
                
                if test_stats is not None:

                    if 'bbox' in test_coco_evaluator.coco_eval:
                        bbox_eval = test_coco_evaluator.coco_eval['bbox']
                        stats = bbox_eval.stats
                        
                        # Log mAP at different IoU thresholds
                        # Todo investigate metrics
                        wandb_coco_stats = {
                            'Test/mAP@[0.50:0.95]': stats[0],  # Mean Average Precision (IoU=0.50:0.95, averaged over all classes)
                            'Test/mAP@0.50': stats[1],  # AP at IoU=0.50 (PASCAL VOC metric)
                            'Test/mAP@0.75': stats[2],  # AP at IoU=0.75 (strict metric)
                            'Test/mAP_small': stats[3],  # AP for small objects (across all classes)
                            'Test/mAP_medium': stats[4],  # AP for medium objects (across all classes)
                            'Test/mAP_large': stats[5],  # AP for large objects (across all classes)
                            
                            'Test/mAR@1': stats[6],  # Mean Average Recall at max 1 detection per image
                            'Test/mAR@10': stats[7],  # Mean Average Recall at max 10 detections per image
                            'Test/mAR@100': stats[8],  # Mean Average Recall at max 100 detections per image (COCO standard)
                            'Test/mAR_small': stats[9],  # AR for small objects (across all classes)
                            'Test/mAR_medium': stats[10],  # AR for medium objects (across all classes)
                            'Test/mAR_large': stats[11],  # AR for large objects (across all classes)
                        }
                        self.wandb_logger.log_metrics(wandb_coco_stats, step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'val_{k}': v for k, v in val_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
            

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if val_coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in val_coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(val_coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)
                                
    
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        self.finish()


    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        val_stats, val_coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)
                
        if self.output_dir:
            dist.save_on_master(val_coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        self.finish()
        
        return
