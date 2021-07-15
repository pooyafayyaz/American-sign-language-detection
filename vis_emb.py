from glob import glob
from natsort import natsorted
import re
import os.path
from skimage import io
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, BoxSelectTool
from MulticoreTSNE import MulticoreTSNE as TSNE


INFER_PATH = "./infer_result/*"

all_file_names = glob(INFER_PATH+"*")


if(not os.path.isfile("result.npy")):
	data_array = []
	for files in tqdm(all_file_names):
		all_img_names = glob(files+"/*")
		for img in all_img_names:
			data = np.loadtxt(img,dtype=float)
			data_array.append({"name": img.split("/")[-1], "data" : data})
	np.save("result.npy", data_array)	
else:
	data_array =  np.load("result.npy",allow_pickle=True)	

df = pd.DataFrame(data_array.tolist(),columns=["name","data"])
data = np.array(df.data.tolist(),dtype=float)
print(data.shape)

pca = PCA(n_components=10)
X_pca = pca.fit_transform(data) 

print(X_pca.shape)



tsne = TSNE(n_jobs=4)
tsne_results = tsne.fit_transform(X_pca)

tool_list = [HoverTool(), BoxSelectTool()]
p = figure(plot_width=2000, plot_height=900,tools=tool_list)
p.circle(tsne_results[:,0],tsne_results[:,1])



show(p)
