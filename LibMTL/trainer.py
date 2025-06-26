import torch, os
import torch.nn as nn
import numpy as np

from LibMTL._record import _PerformanceMeter
from LibMTL.utils import count_parameters
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method
import logging
import sys
from tqdm import tqdm
from PIL import Image
from prettytable import PrettyTable
from . import eva
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

COLOR = {0:(13,38,89), 1:(67,111,134),2: (243, 6, 8), 3: (18, 242, 22), 4: (0, 3, 243), 5: (101, 81, 159), 6: (196, 122, 162), 7: (248, 156, 10)}

MTL_DIR = os.path.dirname(os.path.abspath(__file__))

class Trainer(nn.Module):
    r'''A Multi-Task Learning Trainer.

    This is a unified and extensible training framework for multi-task learning. 

    Args:
        task_dict (dict): A dictionary of name-information pairs of type (:class:`str`, :class:`dict`). \
                            The sub-dictionary for each task has four entries whose keywords are named **metrics**, \
                            **metrics_fn**, **loss_fn**, **weight** and each of them corresponds to a :class:`list`.
                            The list of **metrics** has ``m`` strings, repersenting the name of ``m`` metrics \
                            for this task. The list of **metrics_fn** has two elements, i.e., the updating and score \
                            functions, meaning how to update thoes objectives in the training process and obtain the final \
                            scores, respectively. The list of **loss_fn** has ``m`` loss functions corresponding to each \
                            metric. The list of **weight** has ``m`` binary integers corresponding to each \
                            metric, where ``1`` means the higher the score is, the better the performance, \
                            ``0`` means the opposite.                           
        weighting (class): A weighting strategy class based on :class:`LibMTL.weighting.abstract_weighting.AbsWeighting`.
        architecture (class): An architecture class based on :class:`LibMTL.architecture.abstract_arch.AbsArchitecture`.
        encoder_class (class): A neural network class.
        decoders (dict): A dictionary of name-decoder pairs of type (:class:`str`, :class:`torch.nn.Module`).
        rep_grad (bool): If ``True``, the gradient of the representation for each task can be computed.
        multi_input (bool): Is ``True`` if each task has its own input data, otherwise is ``False``. 
        optim_param (dict): A dictionary of configurations for the optimizier.
        scheduler_param (dict): A dictionary of configurations for learning rate scheduler. \
                                 Set it to ``None`` if you do not use a learning rate scheduler.
        kwargs (dict): A dictionary of hyperparameters of weighting and architecture methods.

    .. note::
            It is recommended to use :func:`LibMTL.config.prepare_args` to return the dictionaries of ``optim_param``, \
            ``scheduler_param``, and ``kwargs``.

    Examples::
        
        import torch.nn as nn
        from LibMTL import Trainer
        from LibMTL.loss import CE_loss_fn
        from LibMTL.metrics import acc_update_fun, acc_score_fun
        from LibMTL.weighting import EW
        from LibMTL.architecture import HPS
        from LibMTL.model import ResNet18
        from LibMTL.config import prepare_args

        task_dict = {'A': {'metrics': ['Acc'],
                           'metrics_fn': [acc_update_fun, acc_score_fun],
                           'loss_fn': [CE_loss_fn],
                           'weight': [1]}}
        
        decoders = {'A': nn.Linear(512, 31)}
        
        # You can use command-line arguments and return configurations by ``prepare_args``.
        # kwargs, optim_param, scheduler_param = prepare_args(params)
        optim_param = {'optim': 'adam', 'lr': 1e-3, 'weight_decay': 1e-4}
        scheduler_param = {'scheduler': 'step'}
        kwargs = {'weight_args': {}, 'arch_args': {}}

        trainer = Trainer(task_dict=task_dict,
                          weighting=EW,
                          architecture=HPS,
                          encoder_class=ResNet18,
                          decoders=decoders,
                          rep_grad=False,
                          multi_input=False,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          **kwargs)

    '''
    def __init__(self, task_dict, weighting, architecture, encoder_class, decoders, 
                 rep_grad, multi_input, optim_param, scheduler_param, 
                 save_path=None, load_path=None, **kwargs):
        super(Trainer, self).__init__()
        
        self.device = torch.device('cuda:3')
        self.kwargs = kwargs
        self.task_dict = task_dict
        self.task_num = len(task_dict)
        self.task_name = list(task_dict.keys())
        self.rep_grad = rep_grad
        self.multi_input = multi_input
        self.scheduler_param = scheduler_param
        self.save_path = save_path
        self.load_path = load_path

        self._prepare_model(weighting, architecture, encoder_class, decoders)
        self._prepare_optimizer(optim_param, scheduler_param)
        
        self.meter = _PerformanceMeter(self.task_dict, self.multi_input)
        self.architecture = architecture
        self.weighting = weighting

    def _prepare_model(self, weighting, architecture, encoder_class, decoders):

        weighting = weighting_method.__dict__[weighting] 
        architecture = architecture_method.__dict__[architecture]
        
        class MTLmodel(architecture, weighting):
            def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, kwargs):
                super(MTLmodel, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
                self.init_param()
                
        self.model = MTLmodel(task_name=self.task_name, 
                              encoder_class=encoder_class, 
                              decoders=decoders, 
                              rep_grad=self.rep_grad, 
                              multi_input=self.multi_input,
                              device=self.device,
                              kwargs=self.kwargs['arch_args']).to(self.device)
        if self.load_path is not None:
            if os.path.isdir(self.load_path):
                self.load_path = os.path.join(self.load_path, 'best.pt')
            self.model.load_state_dict(torch.load(self.load_path), strict=False)
            print('Load Model from - {}'.format(self.load_path))
        count_parameters(self.model)
        
    def _prepare_optimizer(self, optim_param, scheduler_param):
        optim_dict = {
                'sgd': torch.optim.SGD,
                'adam': torch.optim.Adam,
                'adagrad': torch.optim.Adagrad,
                'rmsprop': torch.optim.RMSprop,
            }
        scheduler_dict = {
                'exp': torch.optim.lr_scheduler.ExponentialLR,
                'step': torch.optim.lr_scheduler.StepLR,
                'cos': torch.optim.lr_scheduler.CosineAnnealingLR,
                'reduce': torch.optim.lr_scheduler.ReduceLROnPlateau,
            }
        optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}
        self.optimizer = optim_dict[optim_param['optim']](self.model.parameters(), **optim_arg)
        if scheduler_param is not None:
            scheduler_arg = {k: v for k, v in scheduler_param.items() if k != 'scheduler'}
            self.scheduler = scheduler_dict[scheduler_param['scheduler']](self.optimizer, **scheduler_arg)
        else:
            self.scheduler = None

    def _process_data(self, loader):
        try:
            data, label = next(loader[1])
        except:
            loader[1] = iter(loader[0])
            data, label = next(loader[1])
        data = data.to(self.device, non_blocking=True)
        if not self.multi_input:
            for task in self.task_name:
                label[task] = label[task].to(self.device, non_blocking=True)
        else:
            label = label.to(self.device, non_blocking=True)
        return data, label
    
    def process_preds(self, preds, task_name=None):
        r'''The processing of prediction for each task. 

        - The default is no processing. If necessary, you can rewrite this function. 
        - If ``multi_input`` is ``True``, ``task_name`` is valid and ``preds`` with type :class:`torch.Tensor` is the prediction of this task.
        - otherwise, ``task_name`` is invalid and ``preds`` is a :class:`dict` of name-prediction pairs of all tasks.

        Args:
            preds (dict or torch.Tensor): The prediction of ``task_name`` or all tasks.
            task_name (str): The string of task name.
        '''
        return preds

    def _compute_loss(self, preds, gts, task_name=None):
        if not self.multi_input:
            train_losses = torch.zeros(self.task_num).to(self.device)
            for tn, task in enumerate(self.task_name):
                train_losses[tn] = self.meter.losses[task]._update_loss(preds[task], gts[task])
        else:
            train_losses = self.meter.losses[task_name]._update_loss(preds, gts)
        return train_losses
        
    def _prepare_dataloaders(self, dataloaders):
        if not self.multi_input:
            loader = [dataloaders, iter(dataloaders)]
            return loader, len(dataloaders)
        else:
            loader = {}
            batch_num = []
            for task in self.task_name:
                loader[task] = [dataloaders[task], iter(dataloaders[task])]
                batch_num.append(len(dataloaders[task]))
            return loader, batch_num

    def train(self, train_dataloaders, test_dataloaders, epochs, 
              val_dataloaders=None, return_weight=False, cls_w=1, seg_w=1, seg_only=False, encoder='resnetd50'):
        r'''The training process of multi-task learning.

        Args:
            train_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for training. \
                            If ``multi_input`` is ``True``, it is a dictionary of name-dataloader pairs. \
                            Otherwise, it is a single dataloader which returns data and a dictionary \
                            of name-label pairs in each iteration.

            test_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for the validation or testing. \
                            The same structure with ``train_dataloaders``.
            epochs (int): The total training epochs.
            return_weight (bool): if ``True``, the loss weights will be returned.
        '''
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch

        if seg_only:
            logging.basicConfig(
                filename=MTL_DIR + "/model_out/mtl_seg_only_{}.txt".format(encoder), level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        else:
            logging.basicConfig(filename=MTL_DIR + "/model_out/mtl_{}_{}_libmtl_cls_w={}_seg_w={}_{}.txt".format(self.architecture, self.weighting, cls_w, seg_w, encoder),
                                level=logging.INFO,
                                format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        iterator = tqdm(range(epochs), ncols=70)
        iter_num = 0

        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        for ep, epoch in enumerate(iterator):
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')
            logging.info('[INFO] Epoch {}/{} is training ...'.format(ep, epochs))
            for batch_index in range(train_batch):
                iter_num += 1
                if not self.multi_input:
                    train_inputs, train_gts = self._process_data(train_loader)
                    train_preds = self.model(train_inputs)
                    train_preds = self.process_preds(train_preds)
                    train_losses = self._compute_loss(train_preds, train_gts)
                    self.meter.update(train_preds, train_gts)
                else:
                    train_losses = torch.zeros(self.task_num).to(self.device)
                    for tn, task in enumerate(self.task_name):
                        train_input, train_gt = self._process_data(train_loader[task])
                        train_pred = self.model(train_input, task)
                        train_pred = train_pred[task]
                        train_pred = self.process_preds(train_pred, task)
                        train_losses[tn] = self._compute_loss(train_pred, train_gt, task)
                        self.meter.update(train_pred, train_gt, task)

                self.optimizer.zero_grad(set_to_none=False)
                w = self.model.backward(train_losses, **self.kwargs['weight_args'])
                if w is not None:
                    self.batch_weight[:, epoch, batch_index] = w
                self.optimizer.step()

                if seg_only:
                    logging.info(
                        'iteration {} : loss_{} : {} '.format(iter_num, self.task_name[0], train_losses[0]))
                else:
                    logging.info(
                        'iteration {} : loss_{} : {}  loss_{} : {}'.format(iter_num, self.task_name[0], train_losses[0],
                                                                           self.task_name[1], train_losses[1]))
            
            self.meter.record_time('end')
            self.meter.get_score()
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            self.meter.display(epoch=epoch, mode='train')
            self.meter.reinit()
            
            if val_dataloaders is not None:
                self.meter.has_val = True
                val_improvement = self.test(val_dataloaders, epoch, mode='val', return_improvement=True)
            self.test(test_dataloaders, epoch, mode='test')
            if self.scheduler is not None:
                if self.scheduler_param['scheduler'] == 'reduce' and val_dataloaders is not None:
                    self.scheduler.step(val_improvement)
                else:
                    self.scheduler.step()
            if self.save_path is not None and self.meter.best_result['epoch'] == epoch:
                if seg_only:
                    torch.save(self.model.state_dict(),
                                          os.path.join(self.save_path, 'mtl_seg_only_best_{}.pt'.format(encoder)))
                    print('Save Model {} to {}'.format(epoch, os.path.join(self.save_path, 'mtl_seg_only_best_{}.pt'.format(encoder))))
                else:
                    torch.save(self.model.state_dict(),
                               os.path.join(self.save_path, 'mtl_{}_{}_best_cls_w={}_seg_w={}_{}.pt'.format(self.architecture, self.weighting, cls_w, seg_w, encoder)))
                    print('Save Model {} to {}'.format(epoch, os.path.join(self.save_path,
                                                                           'mtl_{}_{}_best_cls_w={}_seg_{}_{}.pt'.format(self.architecture,
                                                                                                      self.weighting, cls_w, seg_w, encoder))))
        self.meter.display_best_result()
        if return_weight:
            return self.batch_weight


    def test(self, test_dataloaders, epoch=None, mode='test', return_improvement=False):
        r'''The test process of multi-task learning.

        Args:
            test_dataloaders (dict or torch.utils.data.DataLoader): If ``multi_input`` is ``True``, \
                            it is a dictionary of name-dataloader pairs. Otherwise, it is a single \
                            dataloader which returns data and a dictionary of name-label pairs in each iteration.
            epoch (int, default=None): The current epoch. 
        '''
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)
        
        self.model.eval()
        self.meter.record_time('begin')
        with torch.no_grad():
            if not self.multi_input:
                for batch_index in range(test_batch):
                    test_inputs, test_gts = self._process_data(test_loader)
                    test_preds = self.model(test_inputs)
                    test_preds = self.process_preds(test_preds)
                    test_losses = self._compute_loss(test_preds, test_gts)
                    self.meter.update(test_preds, test_gts)
            else:
                for tn, task in enumerate(self.task_name):
                    for batch_index in range(test_batch[tn]):
                        test_input, test_gt = self._process_data(test_loader[task])
                        test_pred = self.model(test_input, task)
                        test_pred = test_pred[task]
                        test_pred = self.process_preds(test_pred)
                        test_loss = self._compute_loss(test_pred, test_gt, task)
                        self.meter.update(test_pred, test_gt, task)
        self.meter.record_time('end')
        self.meter.get_score()
        self.meter.display(epoch=epoch, mode=mode)
        improvement = self.meter.improvement
        self.meter.reinit()
        if return_improvement:
            return improvement


    def mtl_test(self, test_dataloaders, cls_w=1, seg_w=1, cls_loss_w=1.0, seg_only=False, encoder='resnetd50', vis=False):
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)

        eva_classification = eva.Evaluator(3)#tr
        eva_seg = eva.Evaluator(7)#tr

        self.model.eval()
        if seg_only:
            path_out = './mnt/data1_hdd/wgk/libmtllast/checkpoint_MTLESD/test_mtl_seg_only_best_{}.pt'.format(encoder)
            os.makedirs(path_out, exist_ok=True)
            key = self.model.load_state_dict(torch.load('/mnt/data1_hdd/wgk/libmtllast/checkpoint_MTLESD/mtl_seg_only_best_{}.pt'.format(encoder)))
        else:
            path_out = './work_dirs/test_{}_{}_cls_w={}_seg_w={}_{}'.format(self.architecture, self.weighting, cls_w, seg_w, encoder)
            os.makedirs(path_out, exist_ok=True)
            # key = self.model.load_state_dict(
            #     torch.load(
            #         './work_dirs/mtl_{}_{}_best_cls_w={}_seg_w={}_{}.pt'.format(self.architecture, self.weighting, cls_w,
            #                                                                  seg_w, encoder), map_location=self.device))
            
            key = self.model.load_state_dict(
                torch.load(
                    '/mnt/data1_hdd/wgk/libmtllast/checkpoint_MTLESD/mtl_{}_{}_best_cls_w={}_seg_w={}_{}.pt'.format(self.architecture, self.weighting, cls_w,
                                                                             seg_w, encoder), map_location=self.device))
        print(key)
        
        ssim_values = []
        hausdorff_distances = []

        for batch_index in tqdm(range(test_batch)):
            img, labels = self._process_data(test_loader)
            label_seg = labels['segmentation']
            label_classification = labels['classification']
            files = labels['path']
            with torch.no_grad():
                img = img.to(self.device)
                result = self.model(img)
                result = self.process_preds(result)

            result_seg = result['segmentation']
            
            result_seg = torch.softmax(result_seg, dim=1).cpu().numpy()
            result_seg = np.argmax(result_seg, axis=1)
            if not seg_only:
                result_classification = result['classification']
                result_classification = torch.softmax(result_classification, dim=1).cpu().numpy()
                result_classification = np.argmax(result_classification, axis=1)
                eva_classification.add_batch(label_classification.cpu().numpy(), result_classification)
            for i, pre in enumerate(label_classification.cpu().numpy()):
                gt = label_seg.cpu().numpy()[i]
                eva_seg.add_batch(gt, result_seg[i])

                # if '/2/' in files[0][i]:
                #     plt.imshow(result_seg[i])
                #     plt.text(0, 0.5, files[0][i])
                #     plt.show()
                if vis:
                    pre_c = result_classification[i] + 1
                    os.makedirs(path_out + '/' + str(pre_c), exist_ok=True)
                    name = files[0][i].replace('\\', '/').split('/')[-1]

                    # 假设 result_seg[i] 是预测的分割图，类别为 0~7
                    seg = result_seg[i].astype(np.uint8)  # 确保是uint8

                    # resize到(1280, 1024) 注意 OpenCV resize 的参数是 (宽, 高)
                    seg_resized = cv2.resize(seg, (1280, 1024), interpolation=cv2.INTER_NEAREST)

                    # 保存为单通道灰度图（类别索引0~7）
                    Image.fromarray(seg_resized).save(path_out + '/' + str(pre_c) + '/' + name)
            for i in range(label_seg.shape[0]):
                gt = label_seg.cpu().numpy()[i]
                
                ssim_value = eva_seg.SSIM(gt.astype(np.uint8), result_seg[i].astype(np.uint8))
                if not np.isnan(ssim_value):
                    ssim_values.append(ssim_value)
                
                hd = eva_seg.Hausdorff_Distance(gt.astype(np.uint8), result_seg[i].astype(np.uint8))
                hausdorff_distances.append(hd)
                
                eva_seg.add_batch(gt, result_seg[i])

        if not seg_only:
            table = PrettyTable(['classification', 'Acc', 'Recall', 'Precision', 'F1'])

            for i in range(len(eva_classification.Dice())):
                table.add_row([str(i + 1),
                               str(eva_classification.Pixel_Accuracy_Class()[i]),
                               str(eva_classification.Recall()[i]),
                               str(eva_classification.Precision()[i]),
                               str(eva_classification.F1()[i]),
                               ])
            table.add_row(['mean',
                           str(eva_classification.Pixel_Accuracy_Class().mean()),
                           str(eva_classification.Recall().mean()),
                           str(eva_classification.Precision().mean()),
                           str(eva_classification.F1().mean()),
                           ])
            print(table)

        table = PrettyTable(['segmentation', 'Dice', 'Recall', 'Precision', 'IoU'])
        e = eva_seg
        for i in range(len(e.Dice())):
            n = i
            if i > 0:
                n += 1
            table.add_row([str(n),
                           str(e.Dice()[i]),
                           str(e.Recall()[i]),
                           str(e.Precision()[i]),
                           str(e.Intersection_over_Union()[i]),
                           ])
        table.add_row(['mean',
                       str(e.Dice().mean()),
                       str(e.Recall().mean()),
                       str(e.Precision().mean()),
                       str(e.Intersection_over_Union().mean()),
                       ])
        print(table)
        
        print("Image Segmentation SSIM Metrics:", np.mean(ssim_values))

        print("Image Segmentation Hausdorff Distance Metrics:", np.mean(hausdorff_distances))