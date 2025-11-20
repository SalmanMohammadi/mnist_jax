import pyarrow.parquet as pq
import os
from dataclasses import dataclass
from PIL import Image
import io
import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import Mesh, AxisType,NamedSharding, PartitionSpec as P
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
num_epochs = 15
input_shape = 28 * 28
hidden_dim = input_shape // 16
num_layers = 5
num_classes = 10
lr = 2e-3
beta_1 = 0.9
beta_2 = 0.999
eps = 1e-8
lmbda = 0.001

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
            x = x + jax.nn.relu(jnp.dot(x, layer))
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

model = init_model(input_shape, hidden_dim, num_layers, rng, key)
# first moment adamw params
m_1 = jax.tree.map(lambda p: jnp.zeros_like(p), model)
# second moment
m_2 = jax.tree.map(lambda p: jnp.zeros_like(p), model)

### distributed_setup
world_size = jax.device_count()
mesh = jax.make_mesh((world_size,), ("b",))
jax.set_mesh(mesh)
in_specs = (
    jax.tree.map(lambda _: P(), model),
    jax.tree.map(lambda _: P(), m_1),
    jax.tree.map(lambda _: P(), m_2),
    P("b", None),
    P("b"),
    P()
)
out_specs = (
    jax.tree.map(lambda _: P(), model),
    jax.tree.map(lambda _: P(), m_1),
    jax.tree.map(lambda _: P(), m_2),
    P(),
)

# replicate model and optimizer parameters
model = jax.device_put(model, device=NamedSharding(mesh, P()))
m_1 = jax.device_put(m_1, device=NamedSharding(mesh, P()))
m_2 = jax.device_put(m_2, device=NamedSharding(mesh, P()))

# simple cross entropy loss
def calculate_loss(x, y, model):
    logits = model.forward(x)
    logp = jax.nn.log_softmax(logits, axis=-1)
    y_onehot = jax.nn.one_hot(y, num_classes)
    loss = -jnp.mean(jnp.sum(logp * y_onehot, axis=-1))
    return loss

grad_fn = jax.jit(jax.value_and_grad(calculate_loss, argnums=2))

@jax.jit
@jax.shard_map(mesh=mesh, in_specs=in_specs, out_specs=out_specs)
def train_step(model, m_1, m_2, x, y, step):
    loss, grads = grad_fn(x, y, model)
    grads = jax.lax.pmean(grads, "b")
    loss = jax.lax.pmean(loss, "b")
    # print(f"loss: {loss}")
    # exit()
    # adam update
    m_1 = jax.tree.map(lambda m, grad: m * beta_1 + (1 - beta_1) * grad, m_1, grads)
    m_2 = jax.tree.map(lambda v, grad: v * beta_2 + (1 - beta_2) * jnp.square(grad), m_2, grads)
    # bias updates
    m_1_ = jax.tree.map(lambda m: m / (1 - (beta_1 ** step)), m_1)
    m_2_ = jax.tree.map(lambda v: v / (1 - (beta_2 ** step)), m_2)
    model = jax.tree.map(lambda p, m, v: p - (lr * m / (jnp.sqrt(v) + eps)) - (lr * lmbda * p), model, m_1_, m_2_)
    return model, m_1, m_2, loss

num_steps = len(train_inputs) // batch_size
step = 1
train_start = time.perf_counter()
for epoch in range(num_epochs):
    epoch_loss = 0
    d0 = time.perf_counter()
    for i in range(num_steps):
        x = train_inputs[i * batch_size : (i+1) * batch_size]
        y = train_labels[i * batch_size : (i+1) * batch_size]
        model, m_1, m_2, loss = train_step(model, m_1, m_2, x, y, step)
        epoch_loss += loss
        step += 1
    dt = time.perf_counter() - d0
    print(f"Epoch: {epoch} | loss: {epoch_loss / num_steps:.3f} | dt: {dt:.2f}s")

final_loss = epoch_loss / num_steps
print(f"Final loss {final_loss:.5f}, expected: 0.06100")
total_train_time = time.perf_counter() - train_start
print(f"total train time: {total_train_time:.2f}s baseline single device: ~3.3s")

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
