import pickle
import jieba
import sys
import os
import json
import codecs
import numpy as np
from tqdm import tqdm
from params import Params
from opencc import OpenCC
import unicodedata

class data_loader(object):
    def __init__(self,use_pretrained = None):
        self.w_dict = {"_UNK":0}
        self.w_occurence = 0
        self.w_count = 1
        self.w_unknown_count = 0
        self.append_dict = True
        self.invalid_q = 0

        if use_pretrained:
            self.append_dict = False
            self.w_dict, self.w_count = self.process_glove(Params.glove_dir, self.w_dict, self.w_count, Params.emb_size)
            self.ids2word = {v: k for k, v in self.w_dict.items()}

    def ind2word(self,ids):
        output = []
        for i in ids:
            output.append(str(self.ids2word[i]))
        return " ".join(output)
                
    def process_glove(self, wordvecs, dict_, count, emb_size):
        print("Reading GloVe from: {}".format(wordvecs))
        with codecs.open(wordvecs,"rb","utf-8") as f:
            line = f.readline()
            openCC = OpenCC('s2t')
            i = 0
            for j in range(2100):
                vocab = str(j)
                dict_[vocab] = count
                count += 1
            while line:
                vocab = line.split(" ")
                vocab = vocab[:-1]
                if len(vocab) != emb_size + 1:
                    line = f.readline()
                    continue
                vocab = normalize_text(''.join(vocab[0:-emb_size]))
                vocab = openCC.convert(vocab)
                if len(vocab)> 4:
                    line = f.readline()
                    continue
                if vocab not in dict_:
                    dict_[vocab] = count
                line = f.readline()
                count += 1
                i += 1
                if i % 100 == 0:
                    sys.stdout.write("\rProcessing line %d       "%i)
            print("")
        return dict_, count

    def process_json(self, file_dir, out_train, out_valid, write_ = True):
        if not os.path.exists(out_train):
            os.makedirs(out_train)
        if not os.path.exists(out_valid):
            os.makedirs(out_valid)
        self.data = json.load(codecs.open(file_dir,"rb","utf-8",errors='replace'))
        self.loop(self.data, out_train, out_valid, write_ = write_)
        with codecs.open(Params.input_ + "dictionary.txt","wb","utf-8") as f:
            for key, value in sorted(self.w_dict.items(), key=lambda kv: (kv[1], kv[0])):
                f.write("%s: %s" % (key, value) + "\n")

    def process_json_test(self, file_dir, out_test, write_ = True):
        if not os.path.exists(out_test):
            os.makedirs(out_test)
        self.data = json.load(codecs.open(file_dir,"rb","utf-8",errors='replace'))
        self.loop(self.data, out_test, None, write_ = write_)
    
    def loop(self, data, out_train, out_valid, write_ = True):
        data = data['data']
        num_context=0; num_question=0
        for topic in tqdm(data,total = len(data)):
            for para in topic['paragraphs']:
                write_context = ['trainset','validset']
                random_P = [1-Params.valid_split, Params.valid_split]
                choice = np.random.choice(np.arange(2), p=random_P)
                
                if out_valid:
                    words_c = self.add_to_dict(para['context'])
                else:
                    words_c = self.add_to_dict(para['context'], add = False)
                    p_len_lab = context_len_label(para['context'])
                if len(words_c) >= Params.max_p_len:
                    continue
                num_context += 1

                for qas in para['qas']:
                    question = qas['question']
                    if out_valid:
                        words = self.add_to_dict(question)
                    else:
                        words = self.add_to_dict(question, add = False)
                        id_ = qas['id']
                    if len(words) >= Params.max_q_len:
                        continue
                    num_question += 1
                    if out_valid:
                        ans = qas['answers'][0]
                        (start_i, stop_i) = ans_index(para['context'], ans['text'], ans['answer_start'])
                        if start_i == -1:
                            self.invalid_q += 1
                            continue
                    if write_:
                        if write_context[choice]=='trainset':
                            dir_ = out_train
                        if write_context[choice]=='validset':
                            dir_ = out_valid
                        if out_valid == None:
                            dir_ = out_train
                            write_file_id(id_,dir_ + Params.id_dir)
                            write_file(p_len_lab,dir_ + Params.p_len_lab)
                        if out_valid:
                            write_file([str(start_i),str(stop_i)],dir_ + Params.target_dir)
                        write_file(words,dir_ + Params.q_word_dir)
                        write_file(words_c,dir_ + Params.p_word_dir)
                        
        print('number of topic : ' + str(len(data)))
        print('number of context : ' + str(num_context))
        print('nuber of question : ' + str(num_question) )
        if out_valid: print('invalid_question : ' + str(self.invalid_q))

    def add_to_dict(self, line, add = True):
        splitted_line = jieba.cut(line)
        splitted_line = ' '.join(splitted_line)
        splitted_line = str.split(splitted_line)

        words = []
        for i,word in enumerate(splitted_line):
            if word:
                word = self.w_dict.get(word,self.w_dict["_UNK"])
                words.append(str(word))
                if add:
                    self.w_occurence += 1
                if word == 0:
                    if add:
                        self.w_unknown_count += 1
        return words

def context_len_label(line):
    splitted_line = jieba.cut(line)
    splitted_line = ' '.join(splitted_line)
    splitted_line = str.split(splitted_line)
    
    len_lab = []
    counts = 0
    for i,word in enumerate(splitted_line):
        if word:
            len_lab.append(str(counts))
            counts += len(word)
    len_lab.append(str(counts))
    return len_lab
                        
def ans_index(context, answer, ans_start):
    ans_len = len(answer)
    ans_stop = ans_start + ans_len
    start = -1; stop = -1
    
    add_key =   context[:ans_start] + ' AnsStart ' \
              + context[ans_start:ans_stop] \
              + ' AnsStop ' + context[ans_stop:]
              
    splitted_line = jieba.cut(add_key)
    splitted_line = ' '.join(splitted_line)
    splitted_line = str.split(splitted_line)
    for i,word in enumerate(splitted_line):
        if word=='AnsStart': start = i
        if word=='AnsStop' : stop = i-1 
    return (start,stop)

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def write_file(indices, dir_, separate = "\n"):
    with codecs.open(dir_,"ab","utf-8") as f:
        f.write(" ".join(indices) + separate)
        
def write_file_id(indices, dir_, separate = "\n"):
    with codecs.open(dir_,"ab","utf-8") as f:
        f.write(indices + separate)

def load_glove(dir_, name, vocab_size):
    glove = np.zeros((vocab_size, Params.emb_size),dtype = np.float32)
    with codecs.open(dir_,"rb","utf-8") as f:
        line = f.readline()
        openCC = OpenCC('s2t')
        i = 1
        for j in range(2100):
            name_ = str(j)
            vector = [0.1] * Params.emb_size
            vector = np.asarray(vector, np.float32)
            glove[i] = vector
            i += 1
        while line:
            if i % 100 == 0:
                sys.stdout.write("\rProcessing %d vocabs       "%i)
            vector = line.split(" ")
            vector = vector[:-1]
            if len(vector) != Params.emb_size + 1:
                line = f.readline()
                continue
            name_ = normalize_text(''.join(vector[0:-Params.emb_size]))
            name_ = openCC.convert(name_)
            vector = vector[-Params.emb_size:]
            if len(name_)> 4:
                line = f.readline()
                continue
            if vector:
                try:
                    vector = [float(n) for n in vector]
                except:
                    assert 0
                vector = np.asarray(vector, np.float32)
                try:
                    glove[i] = vector
                except:
                    assert 0
            line = f.readline()
            i += 1
        print("\n")
        glove_map = np.memmap(Params.input_ + name + ".np", dtype='float32', mode='write', shape=(vocab_size, Params.emb_size))
        glove_map[:] = glove
        del glove_map

def main():
    with open(Params.input_ + 'dictionary.pkl','wb') as dictionary:
        loader = data_loader(use_pretrained = True)
        print("Tokenizing testing data.")
        loader.process_json_test(Params.test_file, out_test = Params.test_dir)
        print("Tokenizing training data.")
        loader.process_json(Params.train_file, out_train = Params.train_dir, out_valid = Params.valid_dir)
        pickle.dump(loader, dictionary, pickle.HIGHEST_PROTOCOL)
        print("Tokenizing complete")
    if os.path.isfile(Params.input_ + "glove.np"): exit()
    load_glove(Params.glove_dir,"glove",vocab_size = loader.w_count)
    print("Processing complete")
    print("Unknown word ratio: {} / {}".format(loader.w_unknown_count,loader.w_occurence))
    
if __name__ == "__main__":
    main()

