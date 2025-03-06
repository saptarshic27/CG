import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import ot
import scipy
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters

dim = 10
batch_size = 100
degrees_of_freedom = 4
num_epochs = 10
test_size = 2000
import torch.nn as nn

# print(data_chi_square[:5])  # Print first 5 samples to verify

class ChiSquareDataset(Dataset):
    def __init__(self, size, df):
        self.data = np.random.chisquare(df=df, size=(size, dim))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Create synthetic datasets
class R4Dataset(Dataset):
    def __init__(self, size):
        self.data = np.random.randn(size, dim)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, dim)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
def to_device(data, device):
    return torch.tensor(data, dtype=torch.float).to(device)
# Instantiate models
G_XtoY = Generator().to(device)
G_YtoX = Generator().to(device)
D_X = Discriminator().to(device)
D_Y = Discriminator().to(device)


N = 10
n_rep = 5
W1 = torch.empty((N, n_rep)).to(device)
W2 = torch.empty((N, n_rep)).to(device)
R1 = torch.empty((N, n_rep)).to(device)
R2 = torch.empty((N, n_rep)).to(device)

for j in range(N):
    for rep in range(n_rep):
        data_size = 1000*(j+1)
        dataset_X = R4Dataset(data_size)
        dataset_Y = ChiSquareDataset(data_size, degrees_of_freedom)
        dataloader_X = DataLoader(dataset_X, batch_size=batch_size, shuffle=True)
        dataloader_Y = DataLoader(dataset_Y, batch_size=batch_size, shuffle=True)
        G_XtoY = Generator().to(device)
        G_YtoX = Generator().to(device)
        D_X = Discriminator().to(device)
        D_Y = Discriminator().to(device)
        # Loss functions
        criterion_GAN = nn.BCELoss().to(device)
        criterion_cycle = nn.L1Loss().to(device)
        criterion_identity = nn.L1Loss().to(device)

        # Optimizers
        optimizer_G = optim.Adam(list(G_XtoY.parameters()) + list(G_YtoX.parameters()), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D_X = optim.Adam(D_X.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D_Y = optim.Adam(D_Y.parameters(), lr=0.0002, betas=(0.5, 0.999))
        for epoch in range(num_epochs):
            for i, (data_X, data_Y) in enumerate(zip(dataloader_X, dataloader_Y)):
                real_X = data_X.float().to(device)
                real_Y = data_Y.float().to(device)

                # ----------------------
                #  Train Discriminators
                # ----------------------
                # Real loss
                real_loss_D_X = criterion_GAN(D_X(real_X), torch.ones_like(D_X(real_X)))
                real_loss_D_Y = criterion_GAN(D_Y(real_Y), torch.ones_like(D_Y(real_Y)))
        
                # Fake loss
                fake_X = G_YtoX(real_Y)
                fake_Y = G_XtoY(real_X)
                fake_loss_D_X = criterion_GAN(D_X(fake_X.detach()), torch.zeros_like(D_X(fake_X)))
                fake_loss_D_Y = criterion_GAN(D_Y(fake_Y.detach()), torch.zeros_like(D_Y(fake_Y)))
        
                # Total loss
                loss_D_X = (real_loss_D_X + fake_loss_D_X) * 0.5
                loss_D_Y = (real_loss_D_Y + fake_loss_D_Y) * 0.5

                # Backpropagation
                optimizer_D_X.zero_grad()
                loss_D_X.backward()
                optimizer_D_X.step()

                optimizer_D_Y.zero_grad()
                loss_D_Y.backward()
                optimizer_D_Y.step()

                # ------------------
                #  Train Generators
                # ------------------
                # GAN loss
                loss_GAN_XtoY = criterion_GAN(D_Y(fake_Y), torch.ones_like(D_Y(fake_Y)))
                loss_GAN_YtoX = criterion_GAN(D_X(fake_X), torch.ones_like(D_X(fake_X)))
        
                  # Cycle loss
                recovered_X = G_YtoX(fake_Y)
                recovered_Y = G_XtoY(fake_X)
                loss_cycle_X = criterion_cycle(recovered_X, real_X)
                loss_cycle_Y = criterion_cycle(recovered_Y, real_Y)

                # Total loss
                loss_G = (loss_GAN_XtoY + loss_GAN_YtoX + 10.0 * (loss_cycle_X + loss_cycle_Y)) #+ 5.0 * (loss_identity_X + loss_identity_Y))

                # Backpropagation
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

                # ----------------------
                #  Testing
                # ----------------------


          # Generate chi-square distributed test data
        test_data_X = R4Dataset(test_size)
        test_data_Y = ChiSquareDataset(test_size, degrees_of_freedom)

        test_X = to_device(test_data_X, device)
        test_Y = to_device(test_data_Y, device)

        fake_Y = G_XtoY(test_X)
        fake_X = G_YtoX(test_Y)
        recovered_X = G_YtoX(fake_Y)
        recovered_Y = G_XtoY(fake_X)

        fake_X_np = fake_X.detach().cpu().numpy()  # Convert to numpy array
        fake_Y_np = fake_Y.detach().cpu().numpy()  # Convert to numpy array



        A1 = scipy.spatial.distance_matrix(test_data_X, fake_X_np)
        a = np.ones(test_size)/test_size
        b = np.ones(test_size)/test_size
        W1[j,rep] = ot.emd2(a,b,A1)
        A2 = scipy.spatial.distance_matrix(test_data_Y, fake_Y_np)
        W2[j,rep] = ot.emd2(a,b,A2)
        R1[j,rep] = criterion_cycle(recovered_X, test_X).detach()
        R2[j,rep] = criterion_cycle(recovered_Y, test_Y).detach()

        print(f"j =  {j}, rep = {rep}")


torch.save(W1, 'W1_10_50.pt')
torch.save(R1, 'R1_10_50.pt')
torch.save(R2, 'R2_10_50.pt')
torch.save(W2, 'W2_10_50.pt')  
