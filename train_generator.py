from generator import RAGGenerator
from dataset import NQOpenDataset, NQOpenDatasetFactory
from tqdm import tqdm
from logger import TBWriter
from torch.utils.data import DataLoader
import torch

DEVICE = "cuda"
N_EPOCH = 10
BATCH_SIZE = 192
LR = 1e-7
SAVE_EVERY = 1
N_DOCS=1


def main():
    generator = RAGGenerator(DEVICE)
    dataset_factory = NQOpenDatasetFactory(device=DEVICE)
    trainset = dataset_factory.get_trainset(N_DOCS)
    valset = dataset_factory.get_valset(N_DOCS)
    trainloader = DataLoader(trainset, BATCH_SIZE,
                             shuffle=True, collate_fn=trainset.collate_fn)
    valloader = DataLoader(valset, BATCH_SIZE, shuffle=False,
                           collate_fn=valset.collate_fn)
    optimizer = torch.optim.Adam(
        generator.parameters(), lr=LR, betas=(0.9, 0.999),weight_decay=0.01)
    writer = TBWriter("train_generator")
    bar = tqdm(range(1, N_EPOCH+1))
    for epoch in bar:
        train_epoch(generator, trainloader, optimizer, writer, epoch, bar)
        val_epoch(generator, valloader, writer, epoch)


def train_epoch(
    generator:RAGGenerator,
    trainloader,
    optimizer,
    writer,
    current_epoch,
    progress_bar,
):
    generator.train()
    batches, sum_loss = 0, 0.0
    for x in trainloader:
        optimizer.zero_grad()
        loss = generator.get_loss(x["question"], x["context"], x["answer"])
        sum_loss += loss.item()
        progress_bar.set_description(f"step loss: {loss.item():.4f}")
        writer.add_scalar("step loss", loss.item())
        writer.step()
        batches += 1
        loss.backward()
        optimizer.step()
    writer.add_scalar("epoch loss", sum_loss/batches, current_epoch-1)
    if current_epoch % SAVE_EVERY == 0:
        writer.save_ckpt({"generator": generator}, current_epoch)


def val_epoch(
    generator:RAGGenerator,
    valloader,
    writer,
    current_epoch,
):
    generator.eval()
    batches, sum_loss = 0, 0.0
    for x in valloader:
        with torch.no_grad():
            loss = generator.get_loss(x["question"], x["context"], x["answer"])
        sum_loss += loss.item()
        batches += 1
    writer.add_scalar("val loss", sum_loss/batches, current_epoch-1)


if __name__ == "__main__":
    main()
