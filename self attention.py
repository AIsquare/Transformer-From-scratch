import torch

sentence = 'A quick brown fox, jumps over the lazy dog'
dc = {s:i for i,s in enumerate(sorted(sentence.replace(',','').split()))}
print(dc)


sentence_int = torch.tensor([dc[s] for s in sentence.replace(',','').split()])
print(sentence_int)

torch.manual_seed(1223)
embed = torch.nn.Embedding(9,16)
embed_sentence = embed(sentence_int).detach()
print(embed_sentence)

d = embed_sentence.shape[1]
print(d)

d_q, d_k, d_v = 24,24,28

W_query = torch.nn.Parameter(torch.rand(d_q, d))
W_key = torch.nn.Parameter(torch.rand(d_k, d))
W_value = torch.nn.Parameter(torch.rand(d_v, d))

#Computing the Unnormalized Attention Weights
x_2 = embed_sentence[1]
query_2 = W_query.matmul(x_2)
key_2 = W_key.matmul(x_2)
value_2 = W_value.matmul(x_2)

print(query_2.shape)
print(key_2.shape)
print(value_2.shape)

keys = W_key.matmul(embed_sentence.T).T
values = W_value.matmul(embed_sentence.T).T

print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

#Query x Key
omega_24 = query_2.dot(keys[4])
print(omega_24)

omega_2 = query_2.matmul(keys.T)
print(omega_2)

#Computing the Attention Scores

import torch.nn.functional as F

attention_weights_2 = F.softmax(omega_2 / d_k**0.5, dim=0)
print(attention_weights_2)

context_vector_2 = attention_weights_2.matmul(values)

print(context_vector_2.shape)
print(context_vector_2)

context_vector_2 = attention_weights_2.matmul(values)

print(context_vector_2.shape)
print(context_vector_2)

#Multi-Headed Attention
'''Illustrating this in code, suppose we have 3 attention heads, so we now extend the d′×d
 dimensional weight matrices so 3×d′×d
:'''
h =3
multihead_W_query = torch.nn.Parameter(torch.rand(h,d_q,d))
multihead_W_key = torch.nn.Parameter(torch.rand(h, d_k, d))
multihead_W_value = torch.nn.Parameter(torch.rand(h, d_v, d))

multihead_query_2 = multihead_W_query.matmul(x_2)
print(multihead_query_2.shape)

multihead_key_2 = multihead_W_key.matmul(x_2)
multihead_value_2 = multihead_W_value.matmul(x_2)

'''Now, these key and value elements are specific to the query element.
 But, similar to earlier, we will also need the value and keys for the 
 other sequence elements in order to compute the attention scores for the 
 query. We can do this is by expanding the input sequence embeddings to 
 size 3, i.e., the number of attention heads:'''
stacked_inputs = embed_sentence.T.repeat(3, 1, 1)
print(stacked_inputs.shape)

#Now, we can compute all the keys and values using via torch.bmm() ( batch matrix multiplication):
multihead_keys = torch.bmm(multihead_W_key, stacked_inputs)
multihead_values = torch.bmm(multihead_W_value, stacked_inputs)
print('multihead_W_key',multihead_W_key.shape)
print("multihead_keys.shape:", multihead_keys.shape)
print("multihead_values.shape:", multihead_values.shape)

'''We now have tensors that represent the three attention 
heads in their first dimension. The third and second dimensions 
refer to the number of words and the embedding size, respectively.
 To make the values and keys more intuitive to interpret, we will swap 
 the second and third dimensions, resulting in tensors with the same 
 dimensional structure as the original input sequence, embedded_sentence:'''

multihead_keys = multihead_keys.permute(0, 2, 1)
multihead_values = multihead_values.permute(0, 2, 1)
print("multihead_keys.shape:", multihead_keys.shape)
print("multihead_values.shape:", multihead_values.shape)

#Cross-Attetntion
'''What is cross-attention, and how does it differ from self-attention?

In self-attention, we work with the same input sequence. In cross-attention, we mix or combine two different input sequences. In the case of the original transformer architecture above, that’s the sequence returned by the encoder module on the left and the input sequence being processed by the decoder part on the right.

Note that in cross-attention, the two input sequences x1
 and x2
 can have different numbers of elements. However, their embedding dimensions must match.'''

#(Note that the queries usually come from the decoder, and the keys and values usually come from the encoder.)

#The only part that changes in cross attention is that we now have a second input sequence, for example,
#a second sentence with 8 instead of 6 input elements. Here, suppose this is a sentence with 8 tokens.
embed_sentence_2 = torch.rand(8, 16) # 2nd input sequence

keys = W_key.matmul(embed_sentence_2.T).T
values = W_value.matmul(embed_sentence_2.T).T

print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

'''cross-attention is useful when we go
 from an input sentence to an output sentence in the context of language translation. 
 The input sentence represents one input sequence, and the translation represent the second 
 input sequence (the two sentences can different numbers of words).'''