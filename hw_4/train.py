# Read Dataset
import numpy as np
import json
from collections import Counter
import pickle

# Работа со словарем
word_counts = Counter()

for i in range(1, 17):
    captions = json.load(open(f'data/captions_tokenized_{i}.json'))

    #split descriptions into tokens
    for img_i in range(len(captions)):
        for caption_i in range(len(captions[img_i])):
            sentence = captions[img_i][caption_i]
            captions[img_i][caption_i] = ["#START#"]+sentence.split(' ')+["#END#"]

    for el in captions:
      for elem in el:
        word_counts.update(elem)

vocab  = ['#UNK#', '#START#', '#END#', '#PAD#']
vocab += [k for k, v in word_counts.items() if v >= 5 if k not in vocab]
n_tokens = len(vocab)
print('n_tokens:', n_tokens)
word_to_index = {w: i for i, w in enumerate(vocab)}

json.dump(word_to_index, open("data/word_to_index.json", 'w'))
with open('data/vocab', 'wb') as fp:
    pickle.dump(vocab, fp)
# word_to_index_load = json.load( open( "data/word_to_index.json" ) )
# with open ('data/vocab', 'rb') as fp:
#     vocab_load = pickle.load(fp)
print('dict saved')

eos_ix = word_to_index['#END#']
unk_ix = word_to_index['#UNK#']
pad_ix = word_to_index['#PAD#']

def as_matrix(sequences, max_len=None):
    """ Convert a list of tokens into a matrix with padding """
    max_len = max_len or max(map(len,sequences))

    matrix = np.zeros((len(sequences), max_len), dtype='int32') + pad_ix
    for i,seq in enumerate(sequences):
        row_ix = [word_to_index.get(word, unk_ix) for word in seq[:max_len]]
        matrix[i, :len(row_ix)] = row_ix

    return matrix


# Строим модель и готовим функции

import torch, torch.nn as nn
import torch.nn.functional as F

class CaptionNet(nn.Module):
    def __init__(self, n_tokens=n_tokens, emb_size=128, lstm_units=256, cnn_feature_size=2048):
        """ A recurrent 'head' network for image captioning. See scheme above. """
        super(self.__class__, self).__init__()

        self.con1 = nn.Conv2d(cnn_feature_size, 1024, 6)
        self.con2 = nn.Conv2d(1024, 1024, 3)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # a layer that converts conv features to
        self.cnn_to_h0 = nn.Linear(1024, lstm_units)
        self.cnn_to_c0 = nn.Linear(1024, lstm_units)

        # recurrent part, please create the layers as per scheme above.

        # create embedding for input words. Use the parameters (e.g. emb_size).
        self.emb = nn.Embedding(n_tokens, emb_size) # <YOUR CODE>

        # lstm: create a recurrent core of your network. Use either LSTMCell or just LSTM.
        # In the latter case (nn.LSTM), make sure batch_first=True
        self.lstm = nn.LSTM(emb_size, lstm_units, batch_first=True) # <YOUR CODE>

        # create logits: linear layer that takes lstm hidden state as input and computes one number per token
        self.logits = nn.Linear(lstm_units, n_tokens) # <YOUR CODE>

    def forward(self, image_vectors, captions_ix):
        """
        Apply the network in training mode.
        :param image_vectors: torch tensor containing inception vectors. shape: [batch, cnn_feature_size]
        :param captions_ix: torch tensor containing captions as matrix. shape: [batch, word_i].
            padded with pad_ix
        :returns: logits for next token at each tick, shape: [batch, word_i, n_tokens]
        """
        im = self.con1(image_vectors) # [b, 1024, 3, 3]
        im = im.reshape(im.shape[0], 1024, 3 * 3) # [b, 1024, 9]
        im = im.transpose(1,2) # [b, 9, 1024]

        out = self.transformer_encoder(im) # [b, 9, 1024]
        out = out.transpose(1,2).reshape(out.shape[0], 1024, 3 , 3).contiguous() # [b, 1024, 3, 3]

        out = self.con2(out) # [b, 1024, 1, 1]
        image_vectors = out.reshape(out.shape[0], 1024) # [b, 1024]

        initial_cell = self.cnn_to_c0(image_vectors)
        initial_hid = self.cnn_to_h0(image_vectors)

        # compute embeddings for captions_ix
        captions_emb = self.emb(captions_ix) # <YOUR CODE>

        # apply recurrent layer to captions_emb.
        # 1. initialize lstm state with initial_* from above
        # 2. feed it with captions. Mind the dimension order in docstring
        # 3. compute logits for next token probabilities
        # Note: if you used nn.LSTM, you can just give it (initial_cell[None], initial_hid[None]) as second arg

        # lstm_out should be lstm hidden state sequence of shape [batch, caption_length, lstm_units]
        lstm_out, hid = self.lstm(captions_emb, (initial_cell[None], initial_hid[None])) # <YOUR_CODE>

        # compute logits from lstm_out
        logits = self.logits(lstm_out) # <YOUR_CODE>

        return logits


def compute_loss(network, image_vectors, captions_ix):

    # print()
    """
    :param image_vectors: torch tensor containing inception vectors. shape: [batch, cnn_feature_size]
    :param captions_ix: torch tensor containing captions as matrix. shape: [batch, word_i].
        padded with pad_ix
    :returns: scalar crossentropy loss (neg llh) loss for next captions_ix given previous ones
    """

    # captions for input - all except last cuz we don't know next token for last one.
    captions_ix_inp = captions_ix[:, :-1].contiguous()
    captions_ix_next = captions_ix[:, 1:].contiguous()

    # apply the network, get predictions for captions_ix_next
    logits_for_next = network.forward(image_vectors, captions_ix_inp)


    # compute the loss function between logits_for_next and captions_ix_next
    # Use the mask, Luke: make sure that predicting next tokens after EOS do not contribute to loss
    # you can do that either by multiplying elementwise loss by (captions_ix_next != pad_ix)
    # or by using ignore_index in some losses.

    loss = F.cross_entropy(logits_for_next.permute((0,2,1)), captions_ix_next, ignore_index=pad_ix).unsqueeze(dim=0) # <YOUR CODE>

    return loss


network = CaptionNet(n_tokens)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network.to(device)
opt = torch.optim.Adam(network.parameters())
print(device)


from random import choice

def generate_batch(img_codes, captions, batch_size, device, max_caption_len=None):

    #sample random numbers for image/caption indicies
    random_image_ix = np.random.randint(0, len(img_codes), size=batch_size)

    #get images
    batch_images = img_codes[random_image_ix]

    #5-7 captions for each image
    captions_for_batch_images = captions[random_image_ix]

    #pick one from a set of captions for each image
    batch_captions = list(map(choice,captions_for_batch_images))

    #convert to matrix
    batch_captions_ix = as_matrix(batch_captions,max_len=max_caption_len)

    return torch.tensor(batch_images, dtype=torch.float32, device=device), torch.tensor(batch_captions_ix, dtype=torch.int64, device=device)

# Начинаем обучение

batch_size = 50  # adjust me
n_epochs = 50  # adjust me  Было 100
n_batches_per_epoch = 50  # adjust me
n_validation_batches = 5  # how many batches are used for validation after each epoch

from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import train_test_split

print('begin')

import time
start = time.time()

losses = []
for i in range(1, 17):

    captions = json.load(open(f'data/captions_tokenized_{i}.json'))
    img_codes = np.load(f"data/image_codes_for_attn_{i}.npy")

    #split descriptions into tokens
    for img_i in range(len(captions)):
        for caption_i in range(len(captions[img_i])):
            sentence = captions[img_i][caption_i]
            captions[img_i][caption_i] = ["#START#"]+sentence.split(' ')+["#END#"]

    captions = np.array(captions)
    train_img_codes, val_img_codes, train_captions, val_captions = train_test_split(img_codes, captions,
                                                                                    test_size=0.1,
                                                                                    random_state=42)

    for epoch in range(n_epochs):

        train_loss=0
        network.train(True)
        for _ in range(n_batches_per_epoch):

            loss_t = compute_loss(network, *generate_batch(train_img_codes, train_captions, batch_size, device))
            opt.zero_grad()
            loss_t.backward()
            opt.step()

            train_loss += loss_t.item()
            losses.append(loss_t.item())

        train_loss /= n_batches_per_epoch

        val_loss=0
        network.train(False)
        for _ in range(n_validation_batches):
            loss_t = compute_loss(network, *generate_batch(val_img_codes, val_captions, batch_size, device))
            val_loss += loss_t.item()
        val_loss /= n_validation_batches

        print('Epoch: {}, train loss: {}, val loss: {}, time: {}'.format(epoch, train_loss, val_loss, (time.time() - start) / 60))
    print()
    print(f"{i}-й пул файлов обработан")
    print()
    torch.save(network.state_dict(), 'data/network.pth')

# torch.save(network.state_dict(), 'data/network.pth')
print("Finished!")
