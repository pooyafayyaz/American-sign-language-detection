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
from cuml.manifold import TSNE
from bokeh.plotting import figure,show
from bokeh.models import HoverTool,BoxSelectTool
from bokeh.plotting import figure, ColumnDataSource



INFER_PATH = "./infer_result/*"
IMG_path = "./frames/"

all_file_names = glob(INFER_PATH+"*")

im_array = []
if(not os.path.isfile("result.npy")):
	data_array = []
	for files in tqdm(all_file_names):
		all_img_names = glob(files+"/*")
		for img in all_img_names:
			data = np.loadtxt(img,dtype=float)
			data_array.append({"name": img.split("/")[-1], "data" : data})
			im_array.append(cv2.imread(IMG_path+img.split("/")[2]+"/"+img.split("/")[3][:-4] ))

	np.save("result.npy", data_array)
else:
	data_array =  np.load("result.npy",allow_pickle=True)

df = pd.DataFrame(data_array.tolist(),columns=["name","data"])
data = np.array(df.data.tolist(),dtype=float)
print(data.shape)


#
# pca = PCA(n_components=10)
# X_pca = pca.fit_transform(data)
#
# print(X_pca.shape)

tsne = TSNE(n_jobs=4)
tsne_results = tsne.fit_transform(data)

hover_html = """<div>
  <div>
    <img
      src="@image_urls" height="64"
      style="float: left; margin: 0px 15px 15px 0px; image-rendering: pixelated;"
      border="2"
      ></img>
  </div>
  <div>
    <span style="font-size: 17px;">@source_text</span>
  </div>
</div>"""

urls = encode_images(im_array)

df['image_urls'] = urls
df['x'] = tsne_results[:,0]
df['y'] = tsne_results[:,1]


src = ColumnDataSource(data=df)

tool_list = [HoverTool = hover_html,ZoomInTool(), ZoomOutTool()]
p = figure(plot_width=2000, plot_height=900,tools=tool_list)
p.circle(size=4, source=src)


def to_png(image):
    image = image.astype(np.uint8)
    out = BytesIO()
    ia = Image.fromarray(image)
    ia.save(out, format='png')
    return out.getvalue()


def encode_images(images):
    urls=[]
    for im in images:
        png = to_png(im)
        url = 'data:image/png;base64,'
        url += base64.b64encode(png).decode('utf-8')
        urls.append(url)
    return urls



show(p)
