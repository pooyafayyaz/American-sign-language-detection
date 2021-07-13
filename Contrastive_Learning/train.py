from model import *
from dataloader import HandPatchdataset
from torch.optim.lr_scheduler import StepLR
from Loss import ContrastiveLoss

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


num_epochs = 5
batch_size = 10
learning_rate = 0.00001
mlp_hidden_size = 1000
projection_size = 512
num_class = 51
step_size = 30
gamma = 0.1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ('device', device)

input_path = "../result.csv"

train_dataset = HandPatchdataset(input_path, "../frames/")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)


model = Contrastive(mlp_hidden_size, projection_size)
model.to(device)

optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)
n_total_steps = len(train_dataloader)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
criterion = ContrastiveLoss(1)

for epoch in range(num_epochs):
	for i, inputs in enumerate(train_dataloader):
		
		image1_l = inputs[0].to(device).float()
		image1_r = inputs[1].to(device).float()

		image2_l = inputs[2].to(device).float()
		image2_r = inputs[3].to(device).float()

		lable = (1-inputs[4].to(device)).float()
		
		model.train()

		optimizer.zero_grad()
		# compute the model output
		image1_l_out = model(image1_l)
		image1_r_out = model(image1_r)		
		image2_l_out = model(image2_l)		
		image2_r_out = model(image2_r)
		
		# calculate loss
		loss = criterion(image1_l_out, image2_l_out,lable) + criterion(image1_r_out, image1_r_out,lable)
		# credit assignment
		loss.backward()
		# update model weights
		optimizer.step()

		if (i+1) % 100 == 0:
			print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './LSTM.pth'
torch.save(model.state_dict(), PATH)



