from options.test_options1 import TestOptions
from models import create_model
import os
opt = TestOptions().parse()

opt.model = 'Grey_box_attack'
opt.netG = 'search'

GBA = create_model(opt)
GBA.load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'checkpoints/%s/model.pth'%opt.model)
GBA.setup(opt)
GBA.eval()
