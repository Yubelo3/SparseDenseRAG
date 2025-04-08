from generator import RAGGenerator
from dataset import NQOpenDataset, NQOpenDatasetFactory
from tqdm import tqdm
from logger import TBWriter
from torch.utils.data import DataLoader
import torch

DEVICE = "cuda"
N_EPOCH = 20
BATCH_SIZE = 32
LR = 0.005
SAVE_EVERY = 10


generator = RAGGenerator(DEVICE)
dataset_factory = NQOpenDatasetFactory(device=DEVICE)
trainset = dataset_factory.get_trainset()
trainloader = DataLoader(trainset, BATCH_SIZE, shuffle=True)
optimizer = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.9, 0.999))
writer = TBWriter("train_generator")

bar = tqdm(range(1, N_EPOCH+1))
n_step = 0
for epoch in bar:
    batches, sum_loss = 0, 0.0
    for x in trainloader:
        optimizer.zero_grad()
        loss = generator.get_loss(x["question"], x["context"], x["answer"])
        sum_loss += loss.item()
        bar.set_description(f"step loss: {loss.item():.4f}")
        writer.add_scalar("step loss", loss.item(), n_step)
        n_step += 1
        batches += 1
        loss.backward()
        optimizer.step()
    writer.add_scalar(f"epoch loss", {sum_loss/batches}, epoch-1)
    if epoch % SAVE_EVERY == 0:
        writer.save_ckpt({"generator": generator}, epoch)
