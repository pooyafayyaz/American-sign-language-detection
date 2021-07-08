from model import *
from data_loader import LSTMDataset

num_epochs = 500
batch_size = 20
num_layers = 3
learning_rate = 0.00001
sample_rate = 20
hidden_size = 1000
drop_out = 0.5
num_class = 51

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ('device', device)

input_path = "data.csv"
sample_rate_sk = 20

train_dataset = ASLDataset(input_path, sample_rate)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

model = LSTM(input_len, hidden_size, num_layers, num_class, drop_out)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)
n_total_steps = len(train_loader)

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



