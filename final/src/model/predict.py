import json
import codecs
import jieba
import numpy as np

READ_ALL = True
READ_CONTEXT = [0,10]

test_file = 'test-v1.1.json'
predict   = './predict/'
model_1   = predict + 'predict_1.csv'
model_2   = predict + 'predict_2.csv'
model_3   = predict + 'predict_3.csv'
model_4   = predict + 'predict_4.csv'
model_5   = predict + 'predict_5.csv'
model_6   = predict + 'predict_6.csv'
model_7   = predict + 'predict_7.csv'
model_8   = predict + 'predict_8.csv'
model_9   = predict + 'predict_9.csv'
model_10  = predict + 'predict_10.csv'
ensemble  = predict + 'ensemble.csv'

def read_predict(data):
    predict = np.delete(np.genfromtxt(data, delimiter=",", dtype=str), 0, 0)
    return predict

def print_ans(context, ans):
    ans = str.split(ans)
    for i in range(len(ans)):
        ans[i] = int(ans[i])
    start = ans[0]
    stop = ans[-1] + 1
    answer = context[start:stop]
    return answer, [start, stop]

def ans_sent(context, question):
    context = str.split(context,'。')
    context = context[:-1]
    question = jieba.cut(question)
    question = ' '.join(question)
    question = str.split(question)
    score = np.zeros(len(context))
    for words in question:
        score_ = np.zeros(len(context))
        for i,line in enumerate(context):
            if words in line: score_[i] += 1
        if sum(score_) != 0:
            score_ = score_ / sum(score_)
        else:
            for word in words:
                for i, line in enumerate(context):
                    if word in line: score_[i] += 1
                if sum(score_) != 0:
                    score_ = score_ / sum(score_)
                else:
                    score_ = score_
        ind_ = np.argmax(score_)
        score[ind_] += np.max(score_)
    ind_ = np.argmax(score)
    ans_line = context[ind_]
    start = 0
    for index, line in enumerate(context):
        if index < ind_: start += len(line) + 1
        if index == ind_: stop = start + len(line); break
    SS = [start, stop]
    return ans_line, SS

def find_SS(SS, SS1, SS2, SS3):
    c1 = 0; c2 = 0; c3 = 0
    if   SS[0] <= SS1[0] and SS[1] >= SS1[1]: c1 = 1
    elif SS[0] <= SS2[0] and SS[1] >= SS2[1]: c2 = 1
    elif SS[0] <= SS3[0] and SS[1] >= SS3[1]: c3 = 1
    return c1, c2, c3

def find_next(context, question, shift):
    connect = ['與','、','和','及','或','以及']
    q_key = ['?','什','何','哪','誰']
    p = jieba.cut(context) ; p = ' '.join(p); p = str.split(p)
    q = jieba.cut(question); q = ' '.join(q); q = str.split(q)
    index_ = q.index('以及')
    score = -1
    Q = False
    for k in q_key:
        if k in q[index_+1]: Q = True
    for sign in connect:
        if not Q: break
        ind = [index for index, word in enumerate(p) if sign in word]
        for ind_ in ind:
            p_line = p[max(ind_-3,0):ind_]
            q_line = question.split('以及')[0][-5:]
            score_ = simular(p_line, q_line)
            if score < score_: score = score_; index = ind_
    try:
        (start_, stop_) = modify(p, q, index, index_)
        start = 0; stop  = 0; space = 0;
        for word in p[:start_]: start += len(word)
        for word in p[:stop_ ]: stop  += len(word)
        for i in range(start):
            if context[i]=='\n': space += 1
        start += shift + space; stop += shift + space
    except:
        start = -1; stop = -1
    return (start, stop)

def simular(p_line, q_line):
    score = 0
    for word in q_line:
        for words in p_line:
            score += (word in words)
    return score

def modify(p, q, p_ind, q_ind):
    key = q[q_ind-1: q_ind][0]
    start = p_ind + 1
    change = False
    for i, word in enumerate(p[p_ind+1:p_ind+7]):
        for k in key:
            if k in word:
                change = True
                start = p_ind+ i + 1
                stop = p_ind + i + 2
                if len(p[start]) < 3: start = max(p_ind + i, p_ind + 1)
        if change: break
    if start == p_ind + 1 and change == False:
        stop = p_ind + 2
        if len(p[start]) < 3: stop = p_ind + 3;
    if start == p_ind + 2: start = p_ind + 1;
    return (start, stop)
    
def read_file(data):
    data = json.load(codecs.open(data,"rb","utf-8",errors='replace'))
    data = data['data']
    num_context=0; num_question=0;
    predict_1  = read_predict(model_1)
    predict_2  = read_predict(model_2)
    predict_3  = read_predict(model_3)
    predict_4  = read_predict(model_4)
    predict_5  = read_predict(model_5)
    predict_6  = read_predict(model_6)
    predict_7  = read_predict(model_7)
    predict_8  = read_predict(model_8)
    predict_9  = read_predict(model_9)
    predict_10 = read_predict(model_10)
    EM1 = 0; EM2 = 0; EM3 = 0; ART = 0
    ens = [-1] * len(predict_1)
    for topic in data:
        for para in topic['paragraphs']:
            context = para['context']
            num_context += 1
            if READ_CONTEXT[0] <= num_context < READ_CONTEXT[1]:
                READ = True
            else:
                READ = False
            if READ and not READ_ALL:
                print('< C' + str(num_context) + ' >\n')
                print(context + '\n')
            Q = 0
            for qas in para['qas']:
                question = qas['question']
                id_ = qas['id']
                Q += 1
                num_question += 1
                ans_1, [start1, stop1] = print_ans(context, predict_1[num_question-1][1])
                ans_2, [start2, stop2] = print_ans(context, predict_2[num_question-1][1])
                ans_3, [start3, stop3] = print_ans(context, predict_3[num_question-1][1])
                ans_4, [start4, stop4] = print_ans(context, predict_4[num_question-1][1])
                ans_5, [start5, stop5] = print_ans(context, predict_5[num_question-1][1])
                ans_6, [start6, stop6] = print_ans(context, predict_6[num_question-1][1])
                ans_7, [start7, stop7] = print_ans(context, predict_7[num_question-1][1])
                ans_8, [start8, stop8] = print_ans(context, predict_8[num_question-1][1])
                ans_9, [start9, stop9] = print_ans(context, predict_9[num_question-1][1])
                ans_10, [start10, stop10] = print_ans(context, predict_10[num_question-1][1])
                ans_line, [start, stop] = ans_sent(para['context'], question)
                if READ and not READ_ALL:
                    print('Q' + str(Q) + ': ' + question)
                    print('M1 : ' + ans_1)
                    print('M2 : ' + ans_2)
                    print('M3 : ' + ans_3)
                    print('M4 : ' + ans_4)
                    print('M5 : ' + ans_5)
                    print('M6 : ' + ans_6)
                    print('M7 : ' + ans_7)
                    print('M8 : ' + ans_8)
                    print('M9 : ' + ans_9)
                    print('M10 : ' + ans_10)
                    print('EMS: ' + ens_)
                    print('M+ : ' + ans_line)
                    print()
                EM1_,EM2_,EM3_ = find_SS([start, stop], [start10, stop10], [start9, stop9], [start8, stop8])
                if EM1_ == 1: ens[num_question-1] = 1; EM1 += EM1_
                if EM2_ == 1: ens[num_question-1] = 2; EM2 += EM2_
                if EM3_ == 1: ens[num_question-1] = 3; EM3 += EM3_
                if '以及' in question:
                    (start_, stop_) = find_next(ans_line, question, start)
                    if start_ != -1: ens[num_question-1] = [start_,stop_]; ART += 1;
            if READ and not READ_ALL:
                print('---------------------------------------------------------------')
    if READ_ALL:
        print('Number of contexts: %d' % num_context)
        print('Number of questions: %d' % num_question)
        print('Number of EM1 questions: %d' % EM1)
        print('Number of EM2 questions: %d' % EM2)
        print('Number of EM3 questions: %d' % EM3)
        print('Number of 以及 questions: %d' % ART)
        with open(ensemble, 'w') as f:
            f.write('id,answer\n')
            for i in range(num_question):
                if ens[i] == 1 or ens[i] == -1:
                    id_ = predict_10[i][0]
                    ans = predict_10[i][1]
                elif ens[i] == 2:
                    id_ = predict_9[i][0]
                    ans = predict_9[i][1]
                elif ens[i] == 3:
                    id_ = predict_8[i][0]
                    ans = predict_8[i][1]
                else:
                    ans = ''
                    id_ = predict_1[i][0]
                    start = ens[i][0]; stop = ens[i][1]
                    for j in range(start, stop):
                        if j != stop - 1: ans += str(j) + ' '
                        else: ans += str(j)
                f.write(id_ + ',' + ans + '\n')

if __name__=='__main__':
    read_file(test_file)
