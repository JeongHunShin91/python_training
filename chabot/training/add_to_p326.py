# p326 코드 밑에 아래 코드 추가


def generate(self, h, start_id, sample_size):
    sampled = []
    char_id = start_id
    self.lstm.set_state(h)

    H = h.shape[1]
    peeky_h = h.reshape(1, 1, H)
    for _ in range(sample_size):
        x = np.array([char_id]).reshape((1, 1))
        out = self.embed.forward(x)

        out = np.concatenate((peeky_h, out), axis=2)
        out = self.lstm.forward(out)
        out = np.concatenate((peeky_h, out), axis=2)
        score = self.affine.forward(out)

        char_id = np.argmax(score.flatten())
        sampled.append(char_id)

    return sampled
