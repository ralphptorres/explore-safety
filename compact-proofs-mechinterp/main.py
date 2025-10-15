import marimo

__generated_with = "0.17.2"
app = marimo.App(width="full")

with app.setup:
    import torch as t
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import os
    import time
    from torch.utils.data import Dataset, DataLoader
    import random, numpy
    from dataclasses import dataclass
    from jaxtyping import Float, Int
    from torch import Tensor
    from typing import Optional, Callable, Union, List, Tuple
    import copy
    from tqdm import tqdm


@app.class_definition
@dataclass
class Parameters:
    n_ctx: int = 2
    d_vocab: int = 2048
    d_model: int = 128
    num_epoch: int = 2
    batch_size: int = 1024
    subset_percentage: float = 5
    lr: float = 0.001


@app.function
def set_seed(seed: int = 57) -> None:
    numpy.random.seed(seed)
    random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


@app.function
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in: {elapsed_time:.6f} seconds")
        return result, elapsed_time

    return wrapper


@app.class_definition
class TrainingDataMax(Dataset):
    def __init__(self, params):
        set_seed(57)
        self.n = params.d_vocab
        self.n_ctx = params.n_ctx

    def __getitem__(self, idx):
        inputs = [random.randint(0, self.n - 1) for i in range(self.n_ctx)]
        return inputs + [max(inputs)]

    def __len__(self):
        return self.n**self.n_ctx


@app.function
def training_step(
    model,
    optimizer,
    batch: Tuple[
        Int[Tensor, "batch_size"],
        Int[Tensor, "batch_size"],
        Int[Tensor, "batch_size"],
    ],
    params: Parameters,
):
    criterion = t.nn.CrossEntropyLoss()
    inputs, labels = t.stack(batch[:-1], dim=1), batch[-1]
    inputs_one_hot = F.one_hot(inputs, params.d_vocab).float()
    outputs = model(inputs_one_hot)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss


@app.function
def train(model, params):
    loss_history = []

    subset_cardinality = int(
        len(TrainingDataMax(params=Parameters)) * (params.subset_percentage / 100)
    )
    remaining_cardinality = len(TrainingDataMax(params=params)) - subset_cardinality

    train_data, _ = t.utils.data.random_split(
        TrainingDataMax(params=params), [subset_cardinality, remaining_cardinality]
    )

    dataloader = DataLoader(train_data, batch_size=params.batch_size, shuffle=True)

    optimizer = t.optim.AdamW(
        model.parameters(),
        lr=params.lr,
    )

    for epoch in tqdm(range(params.num_epoch)):
        for i, batch in enumerate(dataloader):
            loss = training_step(
                model=model, optimizer=optimizer, batch=batch, params=params
            )

            loss_history.append(loss.detach().item())

    return loss_history


@app.class_definition
class MLP(t.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.n_ctx = params.n_ctx

        self.embedding = t.nn.Linear(params.d_vocab, params.d_model, bias=False)
        self.linear = t.nn.Linear(params.d_model, params.d_model, bias=False)
        self.unembedding = t.nn.Linear(params.d_model, params.d_vocab, bias=False)

    def g(self, x):
        return self.unembedding((self.linear(x)))

    def forward(self, a):
        return self.g(self.embedding(a.sum(dim=1)))


@app.cell
def _():
    params = Parameters()
    performance = []
    train_data = TrainingDataMax(params=Parameters)
    return (params,)


@app.cell
def _(params):
    model = MLP(params=Parameters)
    model = MLP(params=params)
    loss_history = train(model=model, params=params)
    return (loss_history,)


@app.cell
def _(loss_history):
    plt.plot(loss_history)
    plt.title("Loss Curve")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("fig/loss-1.svg")
    plt.show()
    return


@app.cell
def _(loss_history):
    loss_history[-5:]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
