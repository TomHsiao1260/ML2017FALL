class Params():
    
    input_ = './Input/'
    model_ = './model/model/'
    model_train = './model/model/model_train/'
    predict_dir = model_ + 'predict.csv'
    train_dir = input_ + 'trainset/'
    valid_dir = input_ + 'validset/'
    test_dir  = input_ + 'testset/'

    ## data
    data_size = -1   # -1 to use all data
    num_epochs = 100
    train_prop = 0.9 # Not implemented atm
    valid_split = 0.2   # proportion of validset
    
    ## passage
    train_file = input_ + 'train-v1.1.json'
    test_file = input_ + 'test-v1.1.json'

    ## word embedding
    glove_dir = input_ + 'wiki.zh.vec'

    ## Data dir
    target_dir = "indices.txt"
    q_word_dir = "words_questions.txt"
    p_word_dir = "words_context.txt"
    id_dir     = "id_question.txt"
    p_len_lab  = "label_context.txt"

    ## Training
    mode = "test"     # case-insensitive options: ["train", "test", "debug"]
    dropout = None     # dropout probability, if None, don't use dropout
    zoneout = None     # zoneout probability, if None, don't use zoneout
    optimizer = "adam" # Options: ["adadelta", "adam", "gradientdescent", "adagrad"]
    batch_size = 50    # Size of the mini-batch for training
    save_steps = 15     # Save the model at every 50 steps
    clip = True        # clip gradient norm
    norm = 5.0         # global norm
    
    ## NOTE: Change the hyperparameters of your learning algorithm here
    opt_arg = {'adadelta':{'learning_rate':1, 'rho': 0.95, 'epsilon':1e-6},
                'adam':{'learning_rate':1e-3, 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-8},
                'gradientdescent':{'learning_rate':1},
                'adagrad':{'learning_rate':1}}

    ## Architecture
    SRU = True           # Use SRU cell, if False, use standard GRU cell
    max_p_len = 500      # Maximum number of words in each passage context
    max_q_len = 35       # Maximum number of words in each question context
    vocab_size = 185848  # Number of vocabs in glove
    emb_size = 300       # Embeddings size for words
    attn_size = 64       # RNN cell and attention module size
    num_layers = 3       # Number of layers at question-passage matching
    bias = True          # Use bias term in attention
    num_questions = 1741 # Number of questions in testing dataset
