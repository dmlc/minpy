import os, json
import numpy as np
import h5py


def load_coco_data(base_dir='cs231n/datasets/coco_captioning',
                   max_train=None,
                   pca_features=True):
  data = {}
  caption_file = os.path.join(base_dir, 'coco2014_captions.h5')
  with h5py.File(caption_file, 'r') as f:
    for k, v in f.iteritems():
      data[k] = np.asarray(v)

  if pca_features:
    train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7_pca.h5')
  else:
    train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7.h5')
  with h5py.File(train_feat_file, 'r') as f:
    data['train_features'] = np.asarray(f['features'])

  if pca_features:
    val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7_pca.h5')
  else:
    val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7.h5')
  with h5py.File(val_feat_file, 'r') as f:
    data['val_features'] = np.asarray(f['features'])

  dict_file = os.path.join(base_dir, 'coco2014_vocab.json')
  with open(dict_file, 'r') as f:
    dict_data = json.load(f)
    for k, v in dict_data.iteritems():
      data[k] = v

  train_url_file = os.path.join(base_dir, 'train2014_urls.txt')
  with open(train_url_file, 'r') as f:
    train_urls = np.asarray([line.strip() for line in f])
  data['train_urls'] = train_urls

  val_url_file = os.path.join(base_dir, 'val2014_urls.txt')
  with open(val_url_file, 'r') as f:
    val_urls = np.asarray([line.strip() for line in f])
  data['val_urls'] = val_urls

  # Maybe subsample the training data
  if max_train is not None:
    num_train = data['train_captions'].shape[0]
    mask = np.random.randint(num_train, size=max_train)
    data['train_captions'] = data['train_captions'][mask]
    data['train_image_idxs'] = data['train_image_idxs'][mask]

  return data


def decode_captions(captions, idx_to_word):
  singleton = False
  if captions.ndim == 1:
    singleton = True
    captions = captions[None]
  decoded = []
  N, T = captions.shape
  for i in xrange(N):
    words = []
    for t in xrange(T):
      word = idx_to_word[captions[i, t]]
      if word != '<NULL>':
        words.append(word)
      if word == '<END>':
        break
    decoded.append(' '.join(words))
  if singleton:
    decoded = decoded[0]
  return decoded

def default_gen_buckets(sentences, batch_size, end_idx=0):
    len_dict = {}
    new_sentences = []
    max_len = -1
    for sentence in sentences:
      if not 0 in sentence:
        curlen = len(sentence)
        new_sentences.append(sentence)
      else:
        curlen = sentence.tolist().index(0)
        new_sentences.append(sentence[0:curlen])
      if curlen > max_len:
        max_len = curlen
      if curlen in len_dict:
        len_dict[curlen] += 1
      else:
        len_dict[curlen] = 1
    print(len_dict)
    print max_len
    allnum = 0
    for k in len_dict.keys():
        allnum += len_dict[k]
    
    assert(allnum == len(sentences))

    tl = 0
    buckets = []
    for l, n in len_dict.items(): # TODO: There are better heuristic ways to do this    
        if n + tl >= batch_size:
            buckets.append(l)
            tl = 0
        else:
            tl += n
    if tl > 0:
        buckets.append(max_len)
    return buckets, new_sentences

class BucketSentenceIter():
    def __init__(self, data, batch_size, end_idx=0, split='train'):
        sentences = data['%s_captions' % split]
        self.split = split
        self.data = data
        
        buckets, sentences = default_gen_buckets(sentences, batch_size)

        buckets.sort()
        self.buckets = buckets
        
        self.data['%s_bucket_captions' % split] = [[] for _ in buckets]
        self.data['%s_bucket_image_idxs' % split] = [[] for _ in buckets]

        self.default_bucket_key = max(buckets)

        for i in xrange(len(sentences)):
            sentence = sentences[i]
            image_idx =  data['%s_image_idxs' % split][i]

            if len(sentence) == 0:
                continue
            for idx, bkt in enumerate(buckets):
                if bkt >= len(sentence):
                    self.data['%s_bucket_captions' % split][idx].append(sentence)
                    self.data['%s_bucket_image_idxs' % split][idx].append(image_idx)
                    break
        # convert data into ndarrays for better speed during training
        bucket_captions = [np.zeros((len(x), buckets[i])) for i, x in enumerate(self.data['%s_bucket_captions' % split])]
        bucket_image_idxs = [np.zeros((len(x), 1)) for i, x in enumerate(self.data['%s_bucket_image_idxs' % split])]

        for i_bucket in range(len(self.buckets)):
            for j in range(len(self.data['%s_bucket_captions' % split][i_bucket])):
                sentence = self.data['%s_bucket_captions' % split][i_bucket][j]
                bucket_captions[i_bucket][j, :len(sentence)] = sentence
                image_idx = self.data['%s_bucket_image_idxs' % split][i_bucket][j]
                bucket_image_idxs[i_bucket][j] = image_idx
        self.data['%s_bucket_captions' % split] = bucket_captions
        self.data['%s_bucket_image_idxs' % split] = bucket_image_idxs

        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        bucket_sizes = [len(x) for x in self.data['%s_bucket_captions' % split]]

        print("Summary of dataset ==================")
        for bkt, size in zip(buckets, bucket_sizes):
            print("bucket of len %3d : %d samples" % (bkt, size))

        self.batch_size = batch_size
        self.make_data_iter_plan()

        '''
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        '''

    def make_data_iter_plan(self, split='train'):
        "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        bucket_n_batches = []
        for i in range(len(self.data['%s_bucket_captions' % split])):
            bucket_n_batches.append(len(self.data['%s_bucket_captions' % split][i]) / self.batch_size)
            self.data['%s_bucket_captions' % split][i] = self.data['%s_bucket_captions' % split][i][:int(bucket_n_batches[i]*self.batch_size)]

        bucket_plan = np.hstack([np.zeros(n, int)+i for i, n in enumerate(bucket_n_batches)])
        np.random.shuffle(bucket_plan)

        bucket_idx_all = [np.random.permutation(len(x)) for x in self.data['%s_bucket_captions' % split]]

        self.bucket_plan = bucket_plan
        self.bucket_idx_all = bucket_idx_all
        self.bucket_curr_idx = [0 for x in self.data['%s_bucket_captions' % split]]

        self.caption_buffer = []
        self.image_idx_buffer = []
        for i_bucket in range(len(self.data['%s_bucket_captions' % split])):
            captions = np.zeros((self.batch_size, self.buckets[i_bucket]))
            image_idx = np.zeros((self.batch_size, 1))
            self.caption_buffer.append(captions)
            self.image_idx_buffer.append(image_idx)

    def get_batch(self):
        for i_bucket in self.bucket_plan:
            captions = self.caption_buffer[i_bucket]
            i_idx = self.bucket_curr_idx[i_bucket]
            idx = self.bucket_idx_all[i_bucket][i_idx:i_idx+self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size
            
            #init_state_names = [x[0] for x in self.init_states]

            captions = self.data['%s_bucket_captions' % self.split][i_bucket][idx]
            image_idxs = self.data['%s_bucket_image_idxs' % self.split][i_bucket][idx]
            image_features = self.data['%s_features' % self.split][image_idxs.astype(int)]
            urls = self.data['%s_urls' % self.split][image_idxs.astype(int)]
            yield captions, image_features, urls
        reset()

    def reset(self):
        self.bucket_curr_idx = [0 for x in self.data]

def sample_coco_minibatch(data, batch_size=100, split='train'):
  split_size = data['%s_captions' % split].shape[0]
  mask = np.random.choice(split_size, batch_size)
  captions = data['%s_captions' % split][mask]
  image_idxs = data['%s_image_idxs' % split][mask]
  image_features = data['%s_features' % split][image_idxs]
  urls = data['%s_urls' % split][image_idxs]
  return captions, image_features, urls

