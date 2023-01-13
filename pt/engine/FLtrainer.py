# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pandas as pd

import os
import time
import logging
import torch
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.evaluation import PascalVOCDetectionEvaluator
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes

from pt.data.build import (
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
    build_detection_targetonly_loader_two_crops,
)
from pt.data.dataset_mapper import DatasetMapperTwoCropSeparate
from pt.engine.hooks import LossEvalHook
#from pt.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from pt.modeling.meta_arch.multi_teacher import MultiTSModel
from pt.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from pt.solver.build import build_lr_scheduler
from detectron2.utils.env import TORCH_VERSION
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.utils.visualizer import Visualizer
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2 import structures 

from detectron2.data import detection_utils
from PIL import Image
from torchvision import transforms
from augment.transforms import paste_to_batch
from detectron2.structures import ImageList
from pt.structures.instances import FreeInstances
from torch import nn
from augment.getters import transforms_views
from torchvision.ops import RoIAlign
import torch.nn.functional as F
import random
import copy
import gc

from pt.engine.trainer import PTrainer
from pt.modeling.meta_arch.ts_ensemble import EnsembleTSModel


# PTrainer
class FLtrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        self.cfg = cfg
        self.source_index = cfg.UNSUPNET.SOURCE_IDX
        # load data
        data_loader = self.build_train_loader(cfg)
        
        
        # build model architecture
        student_model = self.build_model(cfg)
        #  build optimizer
        optimizer = self.build_optimizer(cfg, student_model)
        
        # build teacher model
        model_path_list = cfg.MODEL.TEACHER_PATH
        model_teacher_list = []                              
        for idx in range(len(model_path_list)):
            teacher_model = self.build_model(cfg)
            model_teacher_list.append(teacher_model)     
        self.model_teacher_list = model_teacher_list
        
        
        # parallel
        if comm.get_world_size() > 1:
            student_model = DistributedDataParallel(
                student_model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            student_model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        multi_teacher_model = MultiTSModel(model_teacher_list, student_model)   
        
        # need to change???
        self.checkpointer = DetectionCheckpointer(
            multi_teacher_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )        
        
        ##------load model weight
        self._update_model_weight()
        
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        

        self.register_hooks(self.build_hooks())

        # merlin to save memeory
        def inplace_relu(m):
            classname = m.__class__.__name__
            if classname.find('ReLU') != -1:
                m.inplace = True

        for i, model_teacher in enumerate(model_teacher_list):
            self.model_teacher_list[i].apply(inplace_relu)
            
        self.model.apply(inplace_relu)
        

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        if cfg.TEST.EVALUATOR == "COCOeval":
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        if cfg.TEST.EVALUATOR == "VOCeval":
            return PascalVOCDetectionEvaluator(dataset_name)
        else:
            raise ValueError("Unknown test evaluator.")

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_targetonly_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    #print("iter:{} device:{}".format(self.iter, self.cfg.MODEL.DEVICE))
                    self.before_step()
                    with torch.autograd.set_detect_anomaly(True):
                        self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, proposal_type="roih"):
        if proposal_type == "rpn":
            # ------------ >all -----------
            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = FreeInstances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits

            # ------------ <all -----------
            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.pseudo_boxes = new_boxes

        elif proposal_type == "roih":

            # ------------ >all -----------
            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = FreeInstances(image_shape)

            # create box
            # new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor
            # new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            # new_proposal_inst.gt_boxes = new_boxes
            # new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes
            # new_proposal_inst.scores = proposal_bbox_inst.scores

            # ------------ <all -----------
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor
            pseudo_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.pseudo_boxes = pseudo_boxes
            new_proposal_inst.scores_logists = proposal_bbox_inst.scores_logists
            if proposal_bbox_inst.has('boxes_sigma'):
                new_proposal_inst.boxes_sigma = proposal_bbox_inst.boxes_sigma

        return new_proposal_inst

    def process_pseudo_label2(self,proposals_roih):
        list_instances = []
        for proposal_bbox_inst in proposals_roih:

            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = FreeInstances(image_shape)

            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor
            pseudo_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.pseudo_boxes = pseudo_boxes
            new_proposal_inst.scores_logists = proposal_bbox_inst.scores_logists
            if proposal_bbox_inst.has('boxes_sigma'):
                new_proposal_inst.boxes_sigma = proposal_bbox_inst.boxes_sigma
            list_instances.append(new_proposal_inst)
        return list_instances
    
    
    def process_pseudo_label(
            self, proposals_rpn_unsup_k, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # all
            if psedo_label_method == "all":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data
            

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[PTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start

        
        ##--------------old if BURNUP
        record_dict = {}
        #  generate the pseudo-label using teacher model
        # note that we do not convert to eval mode, as 1) there is no gradient computed in
        # teacher model and 2) batch norm layers are not updated as well
        # ---------------------------------------------------------------------------- #
        # labeling for target
        # ---------------------------------------------------------------------------- #
        
        num_teacher = len(self.model_teacher_list)
        
        with torch.no_grad():
            
            roih_list = []
            for tea_i in range(num_teacher):
                ( _, proposals_rpn_unsup_k, proposals_roih_unsup_k, _,) = self.model_teacher_list[tea_i](unlabel_data_k, branch="unsup_data_weak")
                roih_list.append(proposals_roih_unsup_k)
            batch_size = len(roih_list[0])
            mt_src = self.get_match_array_nogt(roih_list)
            
            pesudo_proposals_roih_combined = []
            for batch_idx in range(batch_size):
                keep_index = self.bb_ensemble(mt_src[batch_idx],self.source_index)
                pesudo_proposals_roih_combined.append(roih_list[self.source_index][batch_idx][keep_index])
                    
            pesudo_proposals_roih_combined_final = self.process_pseudo_label2(pesudo_proposals_roih_combined)
            

        #  Pseudo-labeling
        joint_proposal_dict = {}
#         pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
#             proposals_roih_unsup_k, "roih", "all"
#         )
        joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_combined_final

        #  add pseudo-label to unlabeled data
        unlabel_data_q = self.remove_label(unlabel_data_q)
        unlabel_data_k = self.remove_label(unlabel_data_k)

        unlabel_data_q = self.add_label(
            unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
        )
        unlabel_data_k = self.add_label(
            unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
        )
        

        unlabel_data_q = self.resize(unlabel_data_q)        
        #print(unlabel_data_q)
        
        all_unlabel_data = unlabel_data_q + unlabel_data_k

        
        # target domain unsupervised loss
        # --------------------------
        record_all_unlabel_data, _, _, _ = self.model(
            unlabel_data_q, branch="unsupervised", danchor=True
        )
        new_record_all_unlabel_data = {}
        for key in record_all_unlabel_data.keys():
            new_record_all_unlabel_data[key + "_unsup"] = record_all_unlabel_data[
                key
            ]
        record_dict.update(new_record_all_unlabel_data)
        

        # weight losses
        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss":                
                if key.split('_')[-1] == "unsup":  # unsupervised loss
                    loss_dict[key] = record_dict[key] * self.cfg.UNSUPNET.TARGET_UNSUP_LOSS_WEIGHT
                else:
                    raise NotImplementedError

        losses = sum(loss_dict.values())
        
        
        ##--------------old if BURNUP

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.clip_gradient(self.model, 10.)
        self.optimizer.step()

        del record_dict
        del loss_dict
        del losses
        torch.cuda.empty_cache()
        gc.collect()

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] if k in x.keys() else 0.0 for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                        student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())
            
    def get_match_array_nogt(self,proposals_roih):
        source_num = len(proposals_roih)
        batch_size = len(proposals_roih[0])

        batch_match_array= []
        for data_idx in range(batch_size):
            match_array_source = []
            for i, source_prediction_n in enumerate(proposals_roih):
                match_array_source_n =[]
                # others
                #pairwise_sa_sb = []        
                for j in range(source_num):
                    if j!=i:
                        sourcen_n_match_other = structures.pairwise_iou(source_prediction_n[data_idx].get('pred_boxes'),proposals_roih[j][data_idx].get('pred_boxes'))
                        #pairwise_sa_sb.append(sourcen_n_match_other)
                        match_array_source_n.append(self.get_match_array(sourcen_n_match_other))
                match_array_source.append(match_array_source_n)
            batch_match_array.append(match_array_source)

        return  batch_match_array

    def get_match_array(self,pairwise_iou_results):
        match_array=[None]*len(pairwise_iou_results)
        for i in range(len(pairwise_iou_results)):
            if torch.sum(pairwise_iou_results[i])==0:
                match_array[i] = False
                #print("{} no match index".format(i))
            else:
                match_array[i] = True
        return match_array
    def bb_ensemble(self,mt_src,src_idx):
        source_num = len(mt_src[0])
        df_src = pd.DataFrame()    
        src_array = np.array(mt_src[src_idx]).T
        df_src = pd.DataFrame(src_array)
        df_src['summary'] = df_src.sum(axis=1)
        keep_index = df_src.index[df_src.summary==source_num]
        return keep_index
    
    
    def load_TSmodel(self,cfg, model_path):
        Trainer =PTrainer
        model = Trainer.build_model(cfg)
        model_teacher = Trainer.build_model(cfg)
        ensem_ts_model = EnsembleTSModel(model_teacher, model)    
        DetectionCheckpointer(ensem_ts_model).resume_or_load(model_path, resume=False)
        return ensem_ts_model


            
    @torch.no_grad()
    def _update_model_weight(self):
        cfg = self.cfg
        #-----load model weight
        model_path_list = cfg.MODEL.TEACHER_PATH
        student_model_path = cfg.MODEL.STUDENT_PATH
        model_list=[]
        for model_teacher_path in model_path_list:
            print("load teacher model:{} ".format(model_teacher_path))
            model_with_weight = self.load_TSmodel(cfg, model_teacher_path)
            model_list.append(model_with_weight)

        print("load student model:{} ".format(student_model_path))
        student_initial_backbone = self.load_TSmodel(cfg, student_model_path)
        
        
        
        #---- load teacher
        for i, model in enumerate(model_list):
            new_teacher_dict = OrderedDict()
    
            source_model_dict = model.modelStudent.state_dict()

            for key, value in source_model_dict.items():
                if key in self.model_teacher_list[i].state_dict().keys():
                    new_teacher_dict[key] = value    
            self.model_teacher_list[i].load_state_dict(new_teacher_dict)

        #----------load student
        new_student_dict = OrderedDict()
        pseudo_model_dict = student_initial_backbone.modelStudent.state_dict()
        for key, value in pseudo_model_dict.items():    
            if key in self.model.state_dict().keys():
                new_student_dict[key] = value
        self.model.load_state_dict(new_student_dict)    



            
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def resume_or_load(self, resume=False):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        if resume:
            checkpoint = self.checkpointer.load(
                self.cfg.MODEL.WEIGHTS, checkpointables=['model', 'optimizer', 'scheduler']
                # self.cfg.MODEL.WEIGHTS, checkpointables=['model', 'optimizer', 'scheduler', 'iteration']
            )
        else:
            checkpoint = self.checkpointer.load(
                self.cfg.MODEL.WEIGHTS, checkpointables=[]
            )
        if resume:
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

#         def test_and_save_results_teacher():
#             self._last_eval_results_teacher = self.test(self.cfg, self.model_teacher)
#             return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_student))
        #ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def preprocess_image_no_normal(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].cuda() for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def resize(self, data):
        data = copy.deepcopy(data)
        bs = len(data)
        for i in range(bs):
            
            #print(data[i]['file_name'])
            img = data[i]['image']
            h, w = img.shape[-2], img.shape[-1]
            ratio = random.uniform(0.5, 1.0)
            d_h, d_w = int(h * ratio), int(w * ratio)
            x1 = int((w - d_w) / 2)
            y1 = int((h - d_h) / 2)
            bg = torch.zeros_like(img)
            try:
                bg += self.model.pixel_mean.cpu().int()
            except:
                bg += self.model.module.pixel_mean.cpu().int()
            bg[:, y1:y1 + d_h, x1:x1 + d_w] = F.interpolate(img.unsqueeze(0).float(),
                                                            size=(d_h, d_w),
                                                            align_corners=False,
                                                            mode='bilinear').squeeze(0)
            data[i]['image'] = bg
            if data[i]['instances'].has('gt_boxes'):
                data[i]['instances'].gt_boxes.tensor *= ratio
                data[i]['instances'].gt_boxes.tensor[:, 0] += x1
                data[i]['instances'].gt_boxes.tensor[:, 2] += x1
                data[i]['instances'].gt_boxes.tensor[:, 1] += y1
                data[i]['instances'].gt_boxes.tensor[:, 3] += y1

            if data[i]['instances'].has('pseudo_boxes'):
                data[i]['instances'].pseudo_boxes.tensor *= ratio
                data[i]['instances'].pseudo_boxes.tensor[:, 0] += x1
                data[i]['instances'].pseudo_boxes.tensor[:, 2] += x1
                data[i]['instances'].pseudo_boxes.tensor[:, 1] += y1
                data[i]['instances'].pseudo_boxes.tensor[:, 3] += y1
        return data

    def clip_gradient(self, model, clip_norm):
        """Computes a gradient clipping coefficient based on gradient norm."""
        totalnorm = 0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                modulenorm = p.grad.norm()
                totalnorm += modulenorm ** 2
        totalnorm = torch.sqrt(totalnorm).item()
        norm = (clip_norm / max(totalnorm, clip_norm))
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.mul_(norm)
