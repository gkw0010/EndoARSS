import torch, argparse
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from aspp import DeepLabHead, DeepLabClassHead
from create_dataset import DS

from LibMTL import Trainer
from LibMTL.model import *
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.metrics import AccMetric
from LibMTL.loss import CELoss
from LibMTL.architecture import AbsArchitecture, HPS, MMoE

def parse_args(parser):
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--train_bs', default=32, type=int, help='batch size for training')
    parser.add_argument('--test_bs', default=16, type=int, help='batch size for test')
    parser.add_argument('--epochs', default=150, type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='/mnt/data1_hdd/wgk/libmtllast/datasets/DS', type=str, help='dataset path')
    parser.add_argument('--cls_w', default=1, type=int, help='loss weight of classification')
    parser.add_argument('--seg_w', default=1, type=int, help='loss weight of segmentation')
    parser.add_argument('--cls_loss_w', default=1, type=float, help='loss weight of cls')
    parser.add_argument('--seg_only', action='store_true', default=False, help='only segmentation')
    parser.add_argument('--encoder', default='resnetd50', type=str, help='class of encoder')
    parser.add_argument('--vis', action='store_true', default=False, help='visualize')
    return parser.parse_args()
    
def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)

    # prepare dataloaders
    ds_train_set = DS(root=params.dataset_path, mode='train', augmentation=params.aug, img_size=int(params.img_size[0]), return_name=True)
    ds_test_set = DS(root=params.dataset_path, mode='test' if params.mode == 'train' else 'test', augmentation=False, img_size=int(params.img_size[0]),
                     return_name=True if params.mode == 'test' else False)
    # print(len(ds_train_set))
    from torch.utils.data import random_split
    import torch

    # ds_train_set, ds_test_set = random_split(ds_train_set, lengths=[1519, 380], generator=torch.Generator().manual_seed(42))
    
    
    ds_train_loader = torch.utils.data.DataLoader(
        dataset=ds_train_set,
        batch_size=params.train_bs,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True)
    
    ds_test_loader = torch.utils.data.DataLoader(
        dataset=ds_test_set,
        batch_size=params.test_bs,
        shuffle=False,
        num_workers=2,
        pin_memory=True)

    # define encoder and decoders
    def encoder_class():
        if params.encoder == 'resnet18':
            return resnet18(pretrained=True)
        elif params.encoder == 'resnet34':
            return resnet34(pretrained=True)
        elif params.encoder == 'resnet50':
            return resnet50(pretrained=True)
        elif params.encoder == 'resnet101':
            return resnet101(pretrained=True)
        elif params.encoder == 'resnetd18':
            return resnet_dilated('resnet18')
        elif params.encoder == 'resnetd34':
            return resnet_dilated('resnet34')
        elif params.encoder == 'resnetd50':
            return resnet_dilated('resnet50')
        elif params.encoder == 'unet':
            return UNet(3, 512)
        # elif params.encoder == 'swinunet':
        #     return Seg_Encoder()
        elif params.encoder == 'dinonet':
            return Dinov2WithLoRA(
                        backbone_size = "base", 
                        r=4, 
                        image_shape=(224, 224), 
                        lora_type="dvlora",  #
                        pretrained_path=params.pretrained_path,
                        residual_block_indexes=[],
                        include_cls_token=True,
                        use_cls_token=False,
                        use_bn=False
                    )
        else:
            print('[ERR] No Supported Encoder!!!, Default Encoder Is Used!!!')
            return resnet_dilated('resnet50')
        # return Seg_Encoder()

    # define Channel
    num_of_channel = 2048
    if params.encoder == 'resnet18' or params.encoder == 'resnet34' or params.encoder == 'resnetd18' or params.encoder == 'resnetd34':
        num_of_channel = 512
    elif params.encoder == 'resnet50' or params.encoder == 'resnetd50':
        num_of_channel = 2048
    elif params.encoder == 'swinunet':
        num_of_channel = 384
    elif params.encoder == 'dinonet':
        num_of_channel = 256

    # define tasks
    task_dict = {'segmentation': {'metrics':['mIoU', 'pixAcc'], 
                              'metrics_fn': SegMetric(7),#tr
                              'loss_fn': SegLoss(),
                              'weight': [int(params.seg_w), int(params.seg_w)]},}
    if not params.seg_only:
        task_dict.update({
            'classification': {'metrics': ['Acc'],
                               'metrics_fn': ClassMetric(3),#tr
                               'loss_fn': CELoss(),
                               'weight': [int(params.cls_w)]}
        })

        decoders = nn.ModuleDict({'segmentation': DeepLabHead(num_of_channel,7),
                                  'classification': DeepLabClassHead(num_of_channel, 3)})
    else:
        decoders = nn.ModuleDict({'segmentation': DeepLabHead(num_of_channel, 7)})
    # decoders = nn.ModuleDict({'segmentation': Seg_Decoder(int(params.img_size[0]))})
    
    class NYUtrainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class, 
                     decoders, rep_grad, multi_input, optim_param, scheduler_param, **kwargs):
            super(NYUtrainer, self).__init__(task_dict=task_dict, 
                                            weighting=weighting, 
                                            architecture=architecture, 
                                            encoder_class=encoder_class, 
                                            decoders=decoders,
                                            rep_grad=rep_grad,
                                            multi_input=multi_input,
                                            optim_param=optim_param,
                                            scheduler_param=scheduler_param,
                                            **kwargs)
            self.seg_only = kwargs['seg_only']

        def process_preds(self, preds):
            img_size = (int(params.img_size[0]), int(params.img_size[1]))
            preds['segmentation'] = F.interpolate(preds['segmentation'], img_size, mode='bilinear', align_corners=True)
            if not self.seg_only:
                preds['classification'] = preds['classification']
            return preds
     
    NYUmodel = NYUtrainer(task_dict=task_dict, 
                          weighting=params.weighting, 
                          architecture=params.arch, 
                          encoder_class=encoder_class, 
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          save_path=params.save_path,
                          load_path=params.load_path,
                          seg_only=params.seg_only,
                          **kwargs)
    if params.mode == 'train':
        NYUmodel.train(ds_train_loader, ds_test_loader, params.epochs, cls_w=int(params.cls_w), seg_w=int(params.seg_w), seg_only=params.seg_only, encoder=params.encoder)
    elif params.mode == 'test':
        NYUmodel.mtl_test(ds_test_loader, cls_w=int(params.cls_w), seg_w=int(params.seg_w), seg_only=params.seg_only, encoder=params.encoder, vis=params.vis)
    else:
        raise ValueError
    
if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    
    params.save_path = '/mnt/data1_hdd/wgk/libmtllast/checkpoint_MTLESD'
    
    params.pretrained_path = '/mnt/data1_hdd/wgk/libmtllast/pretrain_weights'
    
    params.img_size = (244, 244)
    
    params.arch = 'MOLA' # 'PLE'  #'DSelect_k' # 'PLE' # 'MOLA'
    
    #params.seg_only = True
    
    params.seg_w = 1
    
    params.cls_w = 1

    params.lr = 0.0011
    
    params.weighting = 'DB_MTL' # 'GradVac' # 'GradNorm'
    
    params.encoder = 'dinonet' # 'resnet50' # 'dinonet' # 'resnet18'
    
    #############################
    ## DSelect_k Parameters Configuration ##
    params.img_size = (244, 244)
    params.num_experts = [2]
    params.kgamma = 1.0
    params.num_nonzeros = 2
    #############################
    
    # #############################
    # ## PLE Parameters Configuration ##
    # params.img_size = (244, 244, 3)
    # params.num_experts = [1, 2, 2]
    # params.multi_input = False 
    # ############################# 
    
    #############################
    # MMoE Parameters Configuration ##
    params.img_size = (244, 244)
    params.num_experts = [2]
    params.multi_input = False
    ############################# 
    
    params.mode = 'test' #'train / test'  
    #params.vis = True
    # set device
    #set_device("1")
    
    # set random seed
    set_random_seed(42)
    
    main(params)
    
    # Date: 2024/08/30
    # 1. TODO 新增HD、SSIM 图片分割评估指标 (OK)
    # 2. TODO 新增Dinov2 适配多任务学习架构 (ok)
    
