"""

On a envie de connecter des fils

On veut pouvoir faire une architecture.

Du coup on a des inputs, des outputs


"""


batch = Dimention(-1, "batch")
img_features = Dimention(7, "img_features")
img_width = Dimention(8, "img_width")
img_height = Dimention(8, "img_height")
pos_embedding = Dimention(16, "pos_embedding")
t_embedding = Dimention(8, "t_embedding")
computation_dim = Dimention(16, "computation_dim")

noisy = Input(batch, img_features, img_width, img_height)
input = Input(batch, img_features, img_width, img_height)
t = Input(t_embedding)
pos_embedding = Input(pos_embedding, img_width, img_height)

t = BlowUp(batch, img_features, img_width, img_height)
pos_embedding = BlowUp(batch, img_features, img_width, img_height)

full_input = Concatenate([input, t, pos_embedding], 1)

layers = 3
for i in range(layers):
    block_input = Concatenate([noisy, full_input], 1)
    attention = Attention(block_input)
    




model = ouptut.trace()
