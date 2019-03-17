from __future__ import print_function
import json
import random
import torch
# with open('./data/train.json', encoding='utf-8') as f:
#     train = json.load(f)
#
# print(len(train))
# for k in train.keys():
#     print(len(train[k]))
#     exit(777)

# -----
# x = torch.empty(5, 3)
# print(x)
# x = torch.tensor([5.5, 3])
# print(x.size())

# ------------
# with open('./dev_v2.1.json', encoding='utf-8') as f:
#     data = json.load(f)
# print(data['answers'])

# -----
# whole_division = json.load(open('./data/train.json'))
# relations = whole_division.keys()
# for i in relations:
#     rel = i
#     break
# a = whole_division[rel][0]
# print(a)
# print(relations)
# sampled_relation = random.sample(relations, 5)
# print(sampled_relation)
# target = random.choice(range(len(sampled_relation)))
# print(target)

# -----
# import numpy as np
# import random
# a = np.arange(10)
# print(a)
# random.shuffle(a)
# print(a)

# from bert_pytorch.pytorch_pretrained_bert.tokenization import *
# text = 'We bring the best of Google to innovative nonprofits that are committed to creating a world that works for everyone.'
# tokens = tokenn



# a = ['Trump', 'is', 'president']
# print(' '.join(a))

# a = [1, 3, 4]
# b = [8, 9, 'p']
# c = []
# d = []
# c = c + a
# d.append(a)
# print(c)
# print(d)
# a = ['a', 'b']
# c = c + a
# d.append(a)
# print(c)
# print(d)
#
# print([0 for i in c])

# --- grad test ---

x = torch.randn(2, 3, requires_grad=True)
y = x + 2
z = y+2

optimizer = torch.optim.Adam({x}, lr=0.0001)

loss = torch.sum((z-0))
loss.backward()
print(x.grad)
loss.backward()
print(x.grad)
loss.backward()
print('-------')
x.register_hook(lambda grad: grad/2)
print(x.grad)
loss.backward()
print(x.grad)
optimizer.zero_grad()
print(x.grad)

