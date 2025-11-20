import pyarrow.parquet as pq
from dataclasses import dataclass
from PIL import Image
import io
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
import time
rng = jax.random.key(42)

# grab the dataset
def get_datasets():
    train_ds = pq.read_table("mnist/train-00000-of-00001.parquet")
    test_ds = pq.read_table("mnist/test-00000-of-00001.parquet")
    def load_dataset(ds):
        labels = ds["label"].to_numpy()
        inputs = []
        for img_bytes in ds["image"].to_pylist():
            img_data = Image.open(io.BytesIO(img_bytes["bytes"]))
            inputs.append(np.asarray(img_data).flatten() / 255.0)
        return np.array(inputs), labels

    return load_dataset(train_ds), load_dataset(test_ds)

(train_inputs, train_labels), (test_inputs, test_labels) = get_datasets()

train_inputs = jnp.array(train_inputs)
train_labels = jnp.array(train_labels)
test_inputs = jnp.array(test_inputs)
test_labels = jnp.array(test_labels)

# setup some hyperparameters
batch_size = 256
num_epochs = 5
input_shape = 28 * 28
hidden_dim = input_shape // 2
num_layers = 1
num_classes = 10
lr = 2e-3
beta = 0.95
lmbda = 0.001
ns_steps = 2
### define the model
# first we need to use a different random key for each weight init call
rng, key = jax.random.split(rng)
init_weight = lambda in_dim, out_dim, key: jax.random.normal(key, (in_dim, out_dim), dtype=jnp.float32) * jnp.sqrt(2 / in_dim)

@jax.tree_util.register_dataclass
@dataclass
class Model:
    w_0: Array
    layers: tuple[Array, ...]
    w_out: Array

    def forward(self, x):
        x = jax.nn.relu(jnp.dot(x, self.w_0))
        for layer in self.layers:
            x = jax.nn.relu(jnp.dot(x, layer))
        logits = jnp.dot(x, self.w_out)
        return logits

def init_model(input_shape, hidden_dim, num_layers, rng, key):
    w_0 =  init_weight(input_shape, hidden_dim, key)
    layers = []
    for i in range(num_layers):
        rng, key = jax.random.split(rng)
        layers.append(init_weight(hidden_dim, hidden_dim, key))
    rng, key = jax.random.split(rng)
    w_out = init_weight(hidden_dim, num_classes, key)
    return Model(
        w_0=w_0,
        layers=tuple(layers),
        w_out=w_out
    )

model = init_model(input_shape, hidden_dim, num_layers, rng, key)# muon momentum
mu = jax.tree.map(lambda p: jnp.zeros_like(p), model)

# simple cross entropy loss
def calculate_loss(x, y, model):
    logits = model.forward(x)
    logp = jax.nn.log_softmax(logits, axis=-1)
    y_onehot = jax.nn.one_hot(y, num_classes)
    loss = -jnp.mean(jnp.sum(logp * y_onehot, axis=-1))
    return loss

grad_fn = jax.value_and_grad(calculate_loss, argnums=2)

@jax.jit
def train_step(model, mu, x, y, step):
    loss, grads = grad_fn(x, y, model)
    ### muon update
    # original momentum update
    mu = jax.tree.map(lambda m, grad: m * beta + (1 - beta) * grad, mu, grads)
    # nesterov update
    update = jax.tree.map(lambda m, grad: grad * (1 - beta) + beta * m, mu, grads)
    # newton-shulz iteration
    def newton_shulz(G, ns_steps):
        a, b, c = (3.4445, -4.7750,  2.0315)
        transposed = G.shape[-2] > G.shape[-1]
        G = G.astype(jnp.bfloat16)
        if transposed:
            G = G.T
        G = G / (jnp.linalg.norm(G, keepdims=True) + 1e-7)

        def ns_iter(i, X):
            A = X @ X.mT
            B = b * A + c * (A @ A)
            return a * X + B @ X

        G = jax.lax.fori_loop(0, ns_steps, ns_iter, G, unroll=True)
        if transposed:
            G = G.T
        G = G.astype(jnp.float32)
        return G

    update = jax.tree.map(lambda u, g: newton_shulz(u, ns_steps) * (max(1, g.shape[-2] / g.shape[-1]))**0.5, update, grads)
    model = jax.tree.map(lambda p, u: p * (1 - lr * lmbda) - u * lr, model, update)
    return model, mu, loss

num_steps = len(train_inputs) // batch_size
step = 1
train_start = time.perf_counter()
for epoch in range(num_epochs):
    epoch_loss = 0
    d0 = time.perf_counter()
    for i in range(num_steps):
        x = train_inputs[i * batch_size : (i+1) * batch_size]
        y = train_labels[i * batch_size : (i+1) * batch_size]
        model, mu, loss = train_step(model, mu, x, y, step)
        epoch_loss += loss
        step += 1
    dt = time.perf_counter() - d0
    print(f"Epoch: {epoch} | loss: {epoch_loss / num_steps:.3f} | dt: {dt:.2f}s")

total_train_time = time.perf_counter() - train_start
print(f"total train time: {total_train_time:.2f}s")

# evaluate the model
correct_predictions = 0
num_eval_steps = len(test_inputs) // batch_size
for i in range(num_eval_steps):
    idxs = slice(i * batch_size, min(len(test_inputs), (i + 1) * batch_size))
    x = test_inputs[idxs]
    y = test_labels[idxs]
    
    logits = model.forward(x)
    pred = jnp.argmax(logits, axis=-1)
    correct_predictions += jnp.sum(pred == y)

print(f"Accuracy: {(100 * correct_predictions / len(test_inputs)):.2f}%")
