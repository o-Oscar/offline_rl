import offline_rl.model.graph_model as gm

batch = gm.Dimention("batch", -1)
img_features = gm.Dimention("img_features", 7)
img_width = gm.Dimention("img_width", 8)
img_height = gm.Dimention("img_height", 8)
pos_embedding = gm.Dimention("pos_embedding", 16)
t_embedding = gm.Dimention("t_embedding", 8)
computation_dim = gm.Dimention("computation_dim", 32)

noisy = gm.Input("noisy", [batch, img_features, img_width, img_height])
input = gm.Input("input", [batch, img_features, img_width, img_height])
t_input = gm.Input("t", [t_embedding])
pos_input = gm.Input("pos", [pos_embedding, img_width, img_height])

t = gm.BlowUp(t_input, [batch, t_embedding, img_width, img_height])
pos = gm.BlowUp(pos_input, [batch, pos_embedding, img_width, img_height])

full_input = gm.Concatenate([noisy, input, t, pos], 1)
projected_input = gm.Linear(full_input, 1, computation_dim)
layers = 3

# print(attention.numeric_shape)


# first block
attention = gm.SelfAttention(projected_input, [0], 1)
feed_forward = gm.ReLU(gm.Linear(attention, 1, computation_dim))


# remaining blocks
for i in range(layers - 1):
    block_input = gm.Concatenate([full_input, feed_forward], 1)
    projected_block = gm.Linear(block_input, 1, computation_dim)
    attention = gm.SelfAttention(projected_block, [0], 1)
    feed_forward = gm.ReLU(gm.Linear(attention, 1, computation_dim))


output = gm.Linear(feed_forward, 1, img_features)

import torch as th

batch.value = 10
inputs = {
    key.name: th.zeros(key.numeric_shape) for key in [noisy, input, t_input, pos_input]
}
print("output shape :", output(**inputs).shape)
exit()
