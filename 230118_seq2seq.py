# coding: utf-8
import sys
sys.path.append('..')
from common.time_layers import *
from seq2seq import Seq2seq, Encoder

# class PeekyDecoder:
#     def __init__(self, vocab_size, wordvec_size, hidden_size):
#         V, D, H = vocab_size, wordvec_size, hidden_size
#         rn = np.random.randn
#
#         embed_W = (rn(V, D) / 100).astype('f')
#         lstm_Wx = (rn(H + D, 4 * H) / np.sqrt(H + D)).astype('f')
#         lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
#         lstm_b = np.zeros(4 * H).astype('f')
#         affine_W = (rn(H + H, V) / np.sqrt(H + H)).astype('f')
#         affine_b = np.zeros(V).astype('f')
#
#         self.embed = TimeEmbedding(embed_W)
#         self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
#         self.affine = TimeAffine(affine_W, affine_b)
#
#         self.params, self.grads = [], []
#         for layer in (self.embed, self.lstm, self.affine):
#             self.params += layer.params
#             self.grads += layer.grads
#         self.cache = None
#
#     def forward(self, xs, h):
#         N, T = xs.shape
#         N, H = h.shape
#
#         self.lstm.set_state(h)
#
#         out = self.embed.forward(xs)
#         hs = np.repeat(h, T, axis=0).reshape(N, T, H)
#         out = np.concatenate((hs, out), axis=2)
#
#         out = self.lstm.forward(out)
#         out = np.concatenate((hs, out), axis=2)
#
#         score = self.affine.forward(out)
#         self.cache = H
#         return score
#
#     def backward(self, dscore):
#         H = self.cache
#
#         dout = self.affine.backward(dscore)
#         dout, dhs0 = dout[:, :, H:], dout[:, :, :H]
#         dout = self.lstm.backward(dout)
#         dembed, dhs1 = dout[:, :, H:], dout[:, :, :H]
#         self.embed.backward(dembed)
#
#         dhs = dhs0 + dhs1
#         dh = self.lstm.dh + np.sum(dhs, axis=1)
#         return dh
#
#     def generate(self, h, start_id, sample_size):
#         sampled = []
#         char_id = start_id
#         self.lstm.set_state(h)
#
#         H = h.shape[1]
#         peeky_h = h.reshape(1, 1, H)
#         for _ in range(sample_size):
#             x = np.array([char_id]).reshape((1, 1))
#             out = self.embed.forward(x)
#
#             out = np.concatenate((peeky_h, out), axis=2)
#             out = self.lstm.forward(out)
#             out = np.concatenate((peeky_h, out), axis=2)
#             score = self.affine.forward(out)
#
#             char_id = np.argmax(score.flatten())
#             sampled.append(char_id)
#
#         return sampled
#
#
# class PeekySeq2seq(Seq2seq):
#     def __init__(self, vocab_size, wordvec_size, hidden_size):
#         V, D, H = vocab_size, wordvec_size, hidden_size
#         self.encoder = Encoder(V, D, H)
#         self.decoder = PeekyDecoder(V, D, H)
#         self.softmax = TimeSoftmaxWithLoss()
#
#         self.params = self.encoder.params + self.decoder.params
#         self.grads = self.encoder.grads + self.decoder.grads

import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq


# 데이터셋 읽기
(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
char_to_id, id_to_char = sequence.get_vocab()

# 입력 반전 여부 설정
is_reverse =  True
if is_reverse:
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

# 하이퍼 파라메터 설정
vocab_size = len(char_to_id)
wordvec_size = 16
hideen_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

# model = Seq2seq(vocab_size, wordvec_size, hideen_size)
model = PeekySeq2seq(vocab_size, wordvec_size, hideen_size)

optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct,
                                    id_to_char, verbose, is_reverse)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('검증 정확도 %.3f%%' % (acc * 100))

# 그래프 그리기
x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker='o')
plt.xlabel('에폭')
plt.ylabel('정확도')
plt.ylim(0, 1.0)
plt.show()
