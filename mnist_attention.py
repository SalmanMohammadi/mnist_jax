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
batch_size = 128
num_epochs = 100
input_shape = 28 * 28
hidden_dim = 128
num_layers = 2
num_heads = 2
head_dim = 128
num_classes = 10
lr = 2e-3
beta = 0.9

### define the model
# first we need to use a different random key for each weight init call
rng, key = jax.random.split(rng)
init_weight = lambda in_dim, out_dim, key: jax.random.normal(key, (in_dim, out_dim), dtype=jnp.float32) * np.sqrt(1 / in_dim)
# in this script let's use a single attn head
model = {
    "w_0": init_weight(16, hidden_dim, key)
}
for i in range(num_layers):
    current_layer = {}
    for layer in ['k', 'q', 'v']:
        rng, key = jax.random.split(rng)
        current_layer[f"{layer}_proj"] = init_weight(hidden_dim, num_heads * head_dim, key)
    current_layer["o_proj"] = init_weight(num_heads * head_dim, hidden_dim, key)
    model[f"layer_{i}"] = current_layer
    
rng, key = jax.random.split(rng)
model["w_1"] = init_weight(hidden_dim, num_classes, key)

# also initialise velocity params for SGD with momentum
velocity = jax.tree.map(lambda p: jnp.zeros_like(p), model)

def forward(x, model):
    x = x.reshape(batch_size, 28, 28).reshape(batch_size, 7, 4, 7, 4)
    x = x.transpose(0, 1, 3, 2, 4).reshape(batch_size, 49, 16)
    x = jax.nn.relu(jnp.einsum("bsp,ph->bsh", x, model["w_0"]))
    
    for i in range(num_layers):
        cur_layer = model[f"layer_{i}"]
        q = jnp.einsum("bsh,hd->bsd", x, cur_layer["q_proj"])
        k = jnp.einsum("bsh,hd->bsd", x, cur_layer["k_proj"])
        v = jnp.einsum("bsh,hd->bsd", x, cur_layer["v_proj"])

        scores = jnp.einsum("bqd,bkd->bqk", q, k) / jnp.sqrt(head_dim)
        scores = jax.nn.softmax(scores) 
        attn = jnp.einsum("bqk,bvd->bqd", scores, v) 
        attn_out = jnp.einsum("bsd,dh->bsh", attn, cur_layer["o_proj"])

        mean = jnp.mean(attn_out, axis=-1, keepdims=True)
        std = jnp.std(attn_out, axis=-1, keepdims=True)
        attn_out = (attn_out - mean) / (std + 1e-5)        
        x = x + attn_out
    x = jnp.mean(x, axis=1)
    logits = jnp.einsum("bh,hc->bc", x, model["w_1"])    
    return logits



def evaluate():
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

# simple cross entropy loss
def calculate_loss(x, y, model):
    logits = forward(x, model)
    logp = jax.nn.log_softmax(logits, axis=-1)
    y_onehot = jax.nn.one_hot(y, num_classes)
    loss = -jnp.mean(jnp.sum(logp * y_onehot, axis=-1))
    return loss

grad_fn = jax.jit(jax.value_and_grad(calculate_loss, argnums=2))
#grad_fn = jax.value_and_grad(calculate_loss, argnums=2)
num_steps = len(train_inputs) // batch_size
losses = []
for epoch in range(num_epochs):
    epoch_loss = 0
    d0 = time.perf_counter()
    idxs = np.random.permutation(len(train_inputs))
    for i in range(num_steps):
        batch_idxs = idxs[i * batch_size: (i+1) * batch_size]
        x = train_inputs[batch_idxs]
        y = train_labels[batch_idxs]
        loss, grads = grad_fn(x, y, model)
        # momentum update
        velocity = jax.tree.map(lambda v, grad: v * beta + grad, velocity, grads)
        model = jax.tree.map(lambda param, v: param - lr * v, model, velocity)
        epoch_loss += loss
    dt = time.perf_counter() - d0
    print(f"Epoch: {epoch} | loss: {epoch_loss / num_steps:.3f} | dt: {dt:.2f}s")
evaluate()
