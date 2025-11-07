import pyarrow.parquet as pq
from PIL import Image
import io
import jax
import jax.numpy as jnp
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

# setup some hyperparameters
batch_size = 256
num_epochs = 16
input_shape = 28 * 28
hidden_dim = input_shape // 16
num_layers = 2
num_classes = 10
lr = 5e-3
beta = 0.99

### define the model
# first we need to use a different random key for each weight init call
rng, key = jax.random.split(rng)
init_weight = lambda in_dim, out_dim, key: jax.random.normal(key, (in_dim, out_dim), dtype=jnp.float32) * np.sqrt(1 / in_dim)
model = {
    "w_0": init_weight(input_shape, hidden_dim, key)
}
for i in range(1, num_layers):
    rng, key = jax.random.split(rng)
    model[f"w_{i}"] = init_weight(hidden_dim, hidden_dim, key)
rng, key = jax.random.split(rng)
model["w_out"] = init_weight(hidden_dim, num_classes, key)
# also initialise velocity params for SGD with momentum
velocity = jax.tree.map(lambda p: jnp.zeros_like(p), model)

def forward(x, model):
    x = jax.nn.relu(jnp.dot(x, model["w_0"]))
    for i in range(1, num_layers):
        x = jax.nn.relu(jnp.dot(x, model[f"w_{i}"]))
    logits = jnp.dot(x, model["w_out"])
    return logits

# simple cross entropy loss
def calculate_loss(x, y, model):
    logits = forward(x, model)
    logp = jax.nn.log_softmax(logits, axis=-1)
    y_onehot = jax.nn.one_hot(y, num_classes)
    loss = -jnp.mean(jnp.sum(logp * y_onehot, axis=-1))
    return loss

grad_fn = jax.jit(jax.value_and_grad(calculate_loss, argnums=2))

num_steps = len(train_inputs) // batch_size
for epoch in range(num_epochs):
    epoch_loss = 0
    d0 = time.perf_counter()
    for i in range(num_steps):
        x = train_inputs[i * batch_size : (i+1) * batch_size]
        y = train_labels[i * batch_size : (i+1) * batch_size]
        loss, grads = grad_fn(x, y, model)
        # momentum update
        velocity = jax.tree.map(lambda v, grad: v * beta + grad, velocity, grads)
        model = jax.tree.map(lambda param, v: param - lr * v, model, velocity)
        epoch_loss += loss
    dt = time.perf_counter() - d0
    print(f"Epoch: {epoch} | loss: {epoch_loss / num_steps:.3f} | dt: {dt:.2f}s")

# evaluate the model
correct_predictions = 0
num_eval_steps = len(test_inputs) // batch_size
for i in range(num_eval_steps):
    idxs = slice(i * batch_size, min(len(test_inputs), (i + 1) * batch_size))
    x = test_inputs[idxs]
    y = test_labels[idxs]
    
    logits = forward(x, model)
    pred = jnp.argmax(logits, axis=-1)
    correct_predictions += jnp.sum(pred == y)

print(f"Accuracy: {(100 * correct_predictions / len(test_inputs)):.2f}%")
