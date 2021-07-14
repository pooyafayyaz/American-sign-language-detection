from Contrastive_Learning.model import *
from torch.optim.lr_scheduler import StepLR
from glob import glob
from motion_detection import get_motion_seq
from natsort import natsorted
import re
from skimage import io
from tqdm import tqdm

mlp_hidden_size = 1000
projection_size = 512
learning_rate = 0.00001

RESULT_PATH = "./infer_result/"
PATH = "Contrastive_Learning/best-model-parameters.pt"
FRAME_PATH = "./frames/"
RAW_VID_PATH = "/home/pooya/Downloads/cropped_bodypatches/"

all_file_names = glob(FRAME_PATH+"*")

use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if use_cuda else 'cpu')   # 'cpu' in this case
print ('device', DEVICE)


model = Contrastive(mlp_hidden_size, projection_size)
model.to(DEVICE)

model.load_state_dict(torch.load(PATH,map_location=DEVICE))
model.eval()

optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)


for files in tqdm(all_file_names):
	os.mkdir(RESULT_PATH+files.split("/")[-1])
	start_vid1 ,end_vid1 = get_motion_seq(RAW_VID_PATH + files.split("/")[-1]+ ".avi")
	dir_list = os.listdir(files)
	for im_file in natsorted(dir_list):
		if( start_vid1<int(re.search(r'\d+', im_file).group())<end_vid1):

			inputs = io.imread(files+"/"+im_file).T
			inputs = torch.from_numpy(inputs[np.newaxis,:] ).float()
			inputs = inputs.to(DEVICE)
			
			optimizer.zero_grad()
			yhat = model(inputs)

			np.savetxt(RESULT_PATH+files.split("/")[-1]+"/"+im_file+'.txt', yhat.detach().numpy())

