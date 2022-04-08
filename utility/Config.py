import numpy as np
import tensorflow as tf
class Model_Config(object): 
    def __init__(self):
        self.attention_probs_dropout_prob = 0.1
        self.hidden_act = gelu
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 32  # original bert hidden_size = 768
        self.initializer_range = 0.02
        self.intermediate_size = 64  # 4096
        self.max_position_embeddings = 512
        self.num_attention_heads = 8
        self.num_hidden_layers = 24
        self.type_vocab_size = 2
        self.vocab_size = 10001
        self.seq_length = 11  # lenth of meta path plus 3 ([cls] [mask] [pad])

        self.lr = 0.003


        self.input = './Data/movielens_dataset/bert_mask/train_adjlist.txt'
        self.vocab_file = './Data/movielens_dataset/bert_mask/vocab_recommendation.txt'
        self.do_lower_case = True
        self.random_seed = 12345
        self.output_files = ['output1.txt']
        self.do_whole_word_mask = False
        self.max_seq_length = 11
        self.max_seq_length_test = 11
        self.max_predictions_per_seq = 11
        self.dupe_factor = 5
        self.masked_lm_prob = 0.15
        self.short_seq_prob = 0.1



        self.input_filter = 'train_adjlist_filter_cold_start.txt'
        self.few_shot_number = 3
        self.number_walks = 1
        self.walk_length = 4
        self.walk_length_meta_path = 8
        self.representation_size = 32
        self.graph_format = 'adjlist'
        self.workers = 20
        self.window_size = 10
        self.num_users = 6041
        self.num_items = 3953
        self.verbose = 3 # write train file per 'verbose' epoch
        # self.dupe_factor = 10
        # self.Ks = 20

        self.input_meta_path_files = ['meta_path_train.txt']



        # general Conv_parameters
        self.batch_size = 512
        self.batch_size_reconstruct = 64
        self.epochs = 20
        self.learning_rate = 0.003
        self.aggregator_learning_rate = 0.01
        self.learning_rate_downstream = 0.03
        self.negative_num = 99
        self.checkpoint_path_user_task = './Checkpoint/user_task/'
        self.checkpoint_path_item_task = './Checkpoint/item_task/'
        self.hidden_dim = 256
        self.batch_size_recommender = 512
        self.user_epoch = 8
        self.item_epoch = 8
        self.verbose = 1
        self.second_user_epoch = 10
        self.third_user_epoch = 10
        self.second_item_epoch = 10
        self.third_item_epoch = 10

        # self-attention parameter
        self.dropout_rate = 0
        self.num_heads = 4
        self.d_ff = 4
        self.num_blocks = 2


        self.support_num = 3
        self.kshot_num = 3
        self.kshot_second_num = 3
        self.kshot_third_num = 3



        self.oracle_user_ebd_path = './Data/movielens_dataset/target_32_pretrain_feature/user_feature.npy'
        self.oracle_item_ebd_path = './Data/movielens_dataset/target_32_pretrain_feature/item_feature.npy'
        self.pre_train_item_ebd_path = './Data/movielens_dataset/target_32_pretrain_feature/item_feature.npy'
        self.pre_train_user_ebd_path = './Data/movielens_dataset/target_32_pretrain_feature/user_feature.npy'
        self.original_user_ebd = './Data/movielens_dataset/target_32_pretrain_feature/user_feature.npy'
        self.original_item_ebd = './Data/movielens_dataset/target_32_pretrain_feature/item_feature.npy'
        self.embedding_size = 32




        self.oracle_training_file_user_task = './Data/movielens_dataset/user_task/user_task_train_oracle_rating.csv'
        self.oracle_valid_file_user_task = './Data/movielens_dataset/user_task/user_task_valid_oracle_rating.csv'
        self.oracle_training_file_item_task = './Data/movielens_dataset/item_task/item_task_train_oracle_rating.csv'
        self.oracle_valid_file_item_task = './Data/movielens_dataset/item_task/item_task_valid_oracle_rating.csv'
        self.k = 30

        self.training_plus_test_support_file = './Data/movielens_dataset/training_plus_test_support.csv'


        ################## parser setting ################
        self.data_path = './Data/'
        self.dataset = 'movielens_dataset'
        self.pretrain = 0
        self.verbose = 1
        self.is_norm = 1
        self.epoch = 20
        self.subepopch = 200
        self.embed_size = 32
        self.layer_size = '[160, 160, 160, 160]'
        self.layer_size1 = '[32, 32, 32, 32]'

        self.regs = '[1e-5,1e-5,1e-2]'
        self.lr = 0.003
        self.model_type = 'Lightgcn'
        self.adj_type = 'pre'
        self.alg_type = 'lightgcn'
        self.node_dropout_flag = 0
        self.node_dropout = '[0.1]'
        self.mess_dropout = '[0.1]'
        self.Ks = [20]
        self.save_flag = 0
        self.test_flag = 'part'
        self.report = 0
        self.conv_name = 'lightgcn'
        self.pretrain_user_ebd_path = './Data/movielens_dataset/oracle_embedding/LightGCN-GT/user_feature.npy'
        self.pretrain_item_ebd_path = './Data/movielens_dataset/oracle_embedding/LightGCN-GT/item_feature.npy'

        self.initial_user_ebd_path = './Data/movielens_dataset/initial_embedding/init_user_ebd.npy'
        self.initial_item_ebd_path = './Data/movielens_dataset/initial_embedding/init_item_ebd.npy'

        self.meta_user_ebd_path = './Data/movielens_dataset/meta-embedding/meta_user_ebd.npy'
        self.meta_item_ebd_path = './Data/movielens_dataset/meta-embedding/meta_item_ebd.npy'

        self.current_user_ebd_path = './Data/movielens_dataset/current-embedding/current_user_ebd.npy'
        self.current_item_ebd_path =   './Data/movielens_dataset/current-embedding/current_item_ebd.npy'


        self.few_shot_number = 3
        self.pretext_task_name = 'reconstruction'
        self.user_reconstruct_epoch = 8
        self.item_reconstruct_epoch = 8
        self.checkpoint_path_item_task = './Checkpoint/item_task/'
        self.checkpoint_path_user_task = './Checkpoint/user_task/'
        self.checkpoint_path_downstream = './Checkpoint/downstream/'

        self.pretrain_data = self.load_pretrain_data()


    def load_pretrain_data(self):
        pretrain_data = {}
        pretrain_data['user_embed'] = np.load(self.pretrain_user_ebd_path)
        pretrain_data['item_embed'] = np.load(self.pretrain_item_ebd_path)
        pretrain_data['concat_embed'] = np.concatenate((pretrain_data['user_embed'], pretrain_data['item_embed']), 0)
        # print(pretrain_data['concat_embed'].shape)

        return pretrain_data


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf
