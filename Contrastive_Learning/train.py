from model import *
from data_loader import LSTMDataset


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


num_epochs = 500
batch_size = 20
learning_rate = 0.00001
mlp_hidden_size = 1000
projection_size = 512
num_class = 51

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ('device', device)

input_path = "result.csv"

train_dataset = HandPatchdataset(input_path, "./frames/")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6 ,collate_fn=collate_fn)

model = Contrastive(mlp_hidden_size, projection_size)
model.to(device)

optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)
n_total_steps = len(train_loader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(n_total_steps))
criterion = torch.nn.MSELoss()

for epoch in range(num_epochs):
	for i, (inputs, labels) in enumerate(train_dataloader):
		
		print(inputs.shape())
		inputs = inputs.to(device)
		labels = labels.to(device)

		model.train()

		optimizer.zero_grad()
		# compute the model output
		yhat = model(inputs)
		# calculate loss
		loss = criterion(yhat, labels)
		# credit assignment
		loss.backward()
		# update model weights
		optimizer.step()

        if (i+1) % 5 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './LSTM.pth'
torch.save(model.state_dict(), PATH)



