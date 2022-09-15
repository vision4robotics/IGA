import torch
from .base_model import BaseModel
from . import networks



from data_utils import normalize


cls_thres = 0.7

class GreyboxattackModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):


        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L1', type=float, default=1000, help='weight for L1 loss')

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L2', 'cls', 'reg']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.visual_names = ['search_clean_vis','search_adv_vis']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(3, 3, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionL2 = torch.nn.MSELoss()
            self.init_weight_L2 = opt.lambda_L1
            self.init_weight_cls = 1
            self.init_weight_reg = 10
            self.cls_margin = -4
            self.side_margin1 = -5
            self.side_margin2 = -5
            self.weight_L2 = self.init_weight_L2
            self.weight_cls = self.init_weight_cls
            self.weight_reg = self.init_weight_reg
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
        # '''siamrpn++'''
        # self.siam = SiamRPNPP()

    def set_input(self, input):

        self.template = input[0].squeeze(0).cuda()  # pytorch tensor, shape=(1,3,127,127)

        self.search_clean255 = input[1].squeeze(0).cuda() # pytorch tensor, shape=(N,3,255,255) [0,255]
        self.search_clean1 = normalize(self.search_clean255)
        self.num_search = self.search_clean1.size(0)
        # print('clean image shape:',self.init_frame_clean.size())


    def forward(self,target_sz=(255,255)):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        '''resize to (512,512)'''
        search512_clean_1 = torch.nn.functional.interpolate(self.search_clean1, size=(512, 512), mode='bilinear')
        search512_clean_1_255 = torch.nn.functional.interpolate(self.search_clean1, size=(255, 255), mode='bilinear')


        mmm, fff = self.netG(search512_clean_1, search512_clean_1_255)

        search512_adv1 = search512_clean_1_255 + fff  # Residual form: G(A)+A
        self.search_adv1_middle = mmm
        '''Then crop back to (255,255)'''
        self.search_adv1 = torch.nn.functional.interpolate(search512_adv1, size=target_sz, mode='bilinear')
        self.search_adv255 = self.search_adv1 * 127.5 + 127.5
        '''for visualization'''
        self.search_clean_vis = self.search_clean1[0:1]
        self.search_adv_vis = self.search_adv1[0:1]

    def transform(self,patch_clean1,target_sz=(255,255)):
        '''resize to (512,512)'''
        patch512_clean1 = torch.nn.functional.interpolate(patch_clean1, size=(512, 512), mode='bilinear')
        patch512_adv1 = patch512_clean1 + self.netG(patch512_clean1)  # Residual form: G(A)+A
        patch_adv1 = torch.nn.functional.interpolate(patch512_adv1, size=target_sz, mode='bilinear')
        patch_adv255 = patch_adv1 * 127.5 + 127.5
        return patch_adv255

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # Second, G(A) = B
        self.loss_G_L2_middle = self.criterionL2(self.search_adv1_middle, self.search_clean1) * self.weight_L2
        self.loss_G_L2 = self.criterionL2(self.search_adv1, self.search_clean1) * self.weight_L2
        attention_mask = (self.score_maps_clean > cls_thres)
        num_attention = int(torch.sum(attention_mask))
        if num_attention > 0:
            score_map_adv_att = self.score_maps_adv[attention_mask]
            reg_adv_att = self.reg_res_adv[2:4,attention_mask]

            self.feature_clean = torch.nn.functional.normalize(self.feature_clean, dim=1, p=2)
            self.feature_adv = torch.nn.functional.normalize(self.feature_adv, dim=1, p=2)
            self.adv_norms = torch.norm(self.feature_adv, dim=1)
            self.clean_norms = torch.norm(self.feature_clean, dim=1)
            self.loss_norm = self.criterionL2(self.clean_norms, self.adv_norms)
            self.loss_angle = ((torch.cosine_similarity(self.feature_clean, self.feature_adv, dim=1) + 1)).sum()
            print("self.loss_norm:",self.loss_norm)
            print("self.loss_angle:",self.loss_angle/1000)
            self.loss_cls = torch.mean(torch.clamp(score_map_adv_att[:, 1] - score_map_adv_att[:, 0], min=self.cls_margin)) * self.weight_cls
            self.loss_reg = (torch.mean(torch.clamp(reg_adv_att[0,:],min=self.side_margin1))+
                             torch.mean(torch.clamp(reg_adv_att[1,:],min=self.side_margin2))) * self.weight_reg
            # combine loss and calculate gradients
            #self.loss_G = self.loss_G_L2 + self.loss_cls + self.loss_reg + self.loss_G_L2_middle
            self.loss_G = self.loss_G_L2  + self.loss_G_L2_middle + self.loss_angle/1000 - self.loss_norm*1000
            print("self.loss_G_L2_middle:", self.loss_G_L2_middle)
            print("self.loss_G_L2:",self.loss_G_L2)
            # print("self.loss_cls:", self.loss_cls)
            # print("self.loss_reg:", self.loss_reg)
        # else:
        #     self.loss_G = self.loss_G_L2
        self.loss_G.backward()

    def optimize_parameters(self):

        # 1. predict with clean template
        with torch.no_grad():
            self.siam.model.template(self.template)
            self.score_maps_clean, self.feature_clean= self.siam.get_heat_map(self.search_clean255,softmax=True)#(5HWN,),with softmax
        # 2. adversarial attack with GAN
        self.forward()  # compute fake image
        # 3. predict with adversarial template
        self.score_maps_adv,self.reg_res_adv, self.feature_adv = self.siam.get_cls_reg(self.search_adv255,softmax=False)#(5HWN,2)without softmax,(5HWN,4)
        '''backward pass'''
        # update G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights