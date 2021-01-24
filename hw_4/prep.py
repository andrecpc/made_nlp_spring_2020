import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# you may need to !pip install pycocotools
from torchvision.datasets import coco
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])
coco_train = coco.CocoCaptions("./train2017/", "./annotations/captions_train2017.json", transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=coco_train, batch_size=32, shuffle=False, num_workers=4)



from torchvision.models.inception import Inception3
from warnings import warn
class BeheadedInception3(Inception3):
    """ Like torchvision.models.inception.Inception3 but the head goes separately """

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        else: warn("Input isn't transformed")
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x_for_attn = x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x_for_capt = x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        return x_for_attn, x_for_capt, x



from torch.utils.model_zoo import load_url
model= BeheadedInception3(transform_input=True)

inception_url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
model.load_state_dict(load_url(inception_url))

# model = nn.DataParallel(model.train(False).cuda())
model.to(device)
model.eval()



# img_batch, capt_batch = next(iter(data_loader))
#
# # captions batch is transposed in our version. Check if the same is true for yours
# capt_batch = list(zip(*capt_batch))
# img_batch = Variable(img_batch, volatile=True).to(device)
# vec_batch_for_attn, vec_batch, logits_batch  = [var.cpu().data.numpy() for var in model(img_batch)]
#
# print("NN shapes")
# print('before_pool:', np.shape(vec_batch_for_attn))
# print('after_pool:', np.shape(vec_batch))
# print('logits:', np.shape(logits_batch))


# class labels
import requests
# LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
# labels = {int(key): value for (key, value) in requests.get(LABELS_URL).json().items()}
#
# for i in range(0, 10, 3):
#     print("#", i)
#     print('Captions')
#     print(capt_batch[i])
#
#     top_ix = (-logits_batch[i]).argsort()
#     print('NN classes')
#     print(list(map(labels.get, top_ix[:5])))


# from tqdm import tqdm
#
# with torch.no_grad():
#
#     vectors_before_pool, vectors, logits, captions = [], [], [], []
#     for img_batch, capt_batch in tqdm(data_loader):
#         capt_batch = list(zip(*capt_batch))
#         img_batch = Variable(img_batch, volatile=True).to(device)
#         vec_batch_for_attn, vec_batch, logits_batch  = [var.cpu().data.numpy() for var in model(img_batch)]
#
#         logits.extend([vec for vec in logits_batch])
#         captions.extend(capt_batch)
#         vectors.extend([vec for vec in vec_batch])
#
#         ## WARNING! If you're low on ram, comment this line.
#         vectors_before_pool.extend([vec for vec in vec_batch_for_attn])
#
#
# from nltk.tokenize import TweetTokenizer
# tokenizer = TweetTokenizer()
#
# captions_tokenized = [[' '.join(filter(len, tokenizer.tokenize(cap.lower())))
#                            for cap in img_captions]
#                                 for img_captions in tqdm(captions)]
#
#
#
# i = 123
# print("Original:\n%s\n\n" % '\n'.join(captions[i]))
# print("Tokenized:\n%s\n\n"% '\n'.join(captions_tokenized[i]))
#
#
# np.save("./data/image_codes.npy", np.asarray(vectors))
# np.save("./data/image_codes_for_attn.npy", np.asarray(vectors_before_pool))
# np.save('./data/image_logits.npy', np.asarray(logits))
#
# import json
# with open('./data/captions.json', 'w') as f_cap:
#     json.dump(captions, f_cap)
# with open('./data/captions_tokenized.json', 'w') as f_cap:
#     json.dump(captions_tokenized, f_cap)

from tqdm import tqdm
from nltk.tokenize import TweetTokenizer
import json

with torch.no_grad():

    size = len(data_loader.dataset.ids)
    i = 1

    vectors_before_pool, vectors, logits, captions = [], [], [], []

    for img_batch, capt_batch in tqdm(data_loader):
        capt_batch = list(zip(*capt_batch))
        img_batch = Variable(img_batch, volatile=True).to(device)
        vec_batch_for_attn, vec_batch, logits_batch  = [var.cpu().data.numpy() for var in model(img_batch)]

        logits.extend([vec for vec in logits_batch])
        captions.extend(capt_batch)
        vectors.extend([vec for vec in vec_batch])
        ## WARNING! If you're low on ram, comment this line.
        vectors_before_pool.extend([vec for vec in vec_batch_for_attn])

        if len(vectors_before_pool) > size / 20:

            tokenizer = TweetTokenizer()
            captions_tokenized = [[' '.join(filter(len, tokenizer.tokenize(cap.lower())))
                                      for cap in img_captions]
                                            for img_captions in tqdm(captions)]

            np.save(f"./data/image_codes_{i}.npy", np.asarray(vectors))
            np.save(f"./data/image_codes_for_attn_{i}.npy", np.asarray(vectors_before_pool))
            np.save(f'./data/image_logits_{i}.npy', np.asarray(logits))
            with open(f'./data/captions_{i}.json', 'w') as f_cap:
                json.dump(captions, f_cap)
            with open(f'./data/captions_tokenized_{i}.json', 'w') as f_cap:
                json.dump(captions_tokenized, f_cap)

            vectors_before_pool, vectors, logits, captions = [], [], [], []
            i += 1


tokenizer = TweetTokenizer()
captions_tokenized = [[' '.join(filter(len, tokenizer.tokenize(cap.lower())))
                          for cap in img_captions]
                                for img_captions in tqdm(captions)]
np.save(f"./data/image_codes_{i}.npy", np.asarray(vectors))
np.save(f"./data/image_codes_for_attn_{i}.npy", np.asarray(vectors_before_pool))
np.save(f'./data/image_logits_{i}.npy', np.asarray(logits))
with open(f'./data/captions_{i}.json', 'w') as f_cap:
    json.dump(captions, f_cap)
with open(f'./data/captions_tokenized_{i}.json', 'w') as f_cap:
    json.dump(captions_tokenized, f_cap)
