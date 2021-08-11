import numpy as np
from utility.Config import Model_Config
setting = Model_Config()

class Dataset(object):
    '''
    Load the original rating file
    '''

    def __init__(self, data_path):
        self.num_items = setting.num_items
        self.num_users = setting.num_users
        self.batch_size_user = setting.batch_size_user
        self.batch_size_item = setting.batch_size_item
        self.kshot_num = setting.kshot_num
        self.kshot_second_num = setting.kshot_second_num
        self.kshot_third_num = setting.kshot_third_num
        self.padding_number_items = self.num_items
        self.padding_number_users = self.num_users

        self.oracle_uesr_ebd = np.load(setting.oracle_user_ebd_path)
        self.oracle_item_ebd = np.load(setting.oracle_item_ebd_path)

        self.neighbor_dict_user_list, self.neighbor_dict_item_list = self.load_original_rating_file_as_list(
            data_path)
        self.generate_oracle_users_and_items(data_path)


    def load_original_rating_file_as_list(self, filename):
        neighbor_dict_user_list, neighbor_dict_item_list = {}, {}
        with open(filename, 'r') as f:
            line = f.readline()
            while line != "" and line != None:
                arr = line.strip().split(',')
                user, item = int(arr[0]), int(arr[1])
                if user not in neighbor_dict_user_list:
                    neighbor_dict_user_list[user] = []
                    neighbor_dict_user_list[user].append(item)
                else:
                    neighbor_dict_user_list[user].append(item)

                if item not in neighbor_dict_item_list:
                    neighbor_dict_item_list[item] = []
                    neighbor_dict_item_list[item].append(user)
                else:
                    neighbor_dict_item_list[item].append(user)
                line = f.readline()

        # padding, if the number of user and item is not in range(num_items) and range(num_users)
        for user in range(self.num_users):
            if user not in neighbor_dict_user_list.keys():
                neighbor_dict_user_list[user] = []
                neighbor_dict_user_list[user].append(self.num_items)  # padding

        for item in range(self.num_items):
            if item not in neighbor_dict_item_list.keys():
                neighbor_dict_item_list[item] = []
                neighbor_dict_item_list[item].append(self.num_users)

        return neighbor_dict_user_list, neighbor_dict_item_list

    ##########   generate few-shot positive instances      ##########
    '''      for each user, randomly select k-shot items
                 and maxnumber second order 3*k-shot users
    '''

    ##########                                             ##########
    def generate_oracle_users_and_items(self, filename):
        oracle_user_list, oracle_item_list = [], []
        with open(filename, 'r') as f:
            line = f.readline()
            while line != "" and line != None:
                arr = line.strip().split(',')
                user, item = int(arr[0]), int(arr[1])
                oracle_user_list.append(user)
                oracle_item_list.append(item)

                line = f.readline()

            oracle_user_set = set(oracle_user_list)
            oracle_item_set = set(oracle_item_list)
            self.oracle_user_list = list(oracle_user_set)
            self.oracle_item_list = list(oracle_item_set)
            self.oracle_num_users = len(oracle_user_set)
            self.oracle_num_items = len(oracle_item_set)


    '''
    mix-user-task
    '''

    def get_positive_instances_user_task(self, random_seed):
        '''
        uesr-task
        '''
        np.random.seed(random_seed)
        batch_num = self.oracle_num_users // self.batch_size_user + 1
        target_user, k_shot_item, second_order_uesrs, oracle_user_ebd, mask_num_second_order_user, third_order_items, mask_num_third_order_item = [], [], [], [], [], [], []
        for batch in range(batch_num):
            b_target_u, b_k_shot_item, b_2nd_order_u, b_3rd_order_i, b_oracle_u_ebd, b_mask_num_2nd_u, b_mask_num_3rd_i = self._get_positive_batch_user_task(
                batch)

            target_user.append(b_target_u)
            k_shot_item.append(b_k_shot_item)
            second_order_uesrs.append(b_2nd_order_u)
            oracle_user_ebd.append(b_oracle_u_ebd)
            mask_num_second_order_user.append(b_mask_num_2nd_u)

            third_order_items.append(b_3rd_order_i)
            mask_num_third_order_item.append(b_mask_num_3rd_i)

        return target_user, k_shot_item, second_order_uesrs, third_order_items, oracle_user_ebd, mask_num_second_order_user, mask_num_third_order_item

    def _get_positive_batch_user_task(self, i):
        batch_target_user, batch_kshot_item, batch_2nd_user, batch_oracle_user_ebd, mask_num_2nd_user, batch_3rd_item, mask_num_3rd_item = [], [], [], [], [], [], []

        begin_index = i * self.batch_size_user
        end_index = min(begin_index + self.batch_size_user, self.oracle_num_users)
        for per_user_index in range(begin_index, end_index):
            target_user_id = self.oracle_user_list[per_user_index]


            per_oracle_user_ebd = self.oracle_uesr_ebd[target_user_id]
            sample_kshot_item = np.random.choice(self.neighbor_dict_user_list[target_user_id], self.kshot_num,
                                                 replace=False)
            current_second_order_user = []
            for each_kshot_item in sample_kshot_item:

                candidate_second_order_user = self.neighbor_dict_item_list[each_kshot_item]

                if target_user_id in candidate_second_order_user:
                    candidate_second_order_user.remove(target_user_id)

                if len(candidate_second_order_user) == 0:
                    candidate_second_order_user.append(target_user_id)

                if len(candidate_second_order_user) < self.kshot_second_num:
                    temp_second_order_user = list(
                        np.random.choice(candidate_second_order_user, self.kshot_second_num, replace=True))
                else:
                    temp_second_order_user = list(
                        np.random.choice(candidate_second_order_user, self.kshot_second_num, replace=False))

                current_second_order_user += temp_second_order_user
            temp_second_order_user = list(set(current_second_order_user))

            current_third_order_item = []
            for each_kshot_3rd_user in temp_second_order_user:
                candidate_third_order_item = self.neighbor_dict_user_list[each_kshot_3rd_user]
                if len(candidate_third_order_item) < self.kshot_third_num:
                    temp_third_order_item = list(
                        np.random.choice(candidate_third_order_item, self.kshot_third_num, replace=True))
                else:
                    temp_third_order_item = list(
                        np.random.choice(candidate_third_order_item, self.kshot_third_num, replace=False))
                current_third_order_item += temp_third_order_item
            temp_third_order_item = list(set(current_third_order_item))

            batch_target_user.append(target_user_id)
            batch_kshot_item.append(sample_kshot_item)
            batch_2nd_user.append(temp_second_order_user)
            batch_oracle_user_ebd.append(per_oracle_user_ebd)
            mask_num_2nd_user.append(len(temp_second_order_user))

            batch_3rd_item.append(temp_third_order_item)
            mask_num_3rd_item.append(len(temp_third_order_item))

        batch_2nd_user_input = self._add_mask(self.padding_number_users, batch_2nd_user, max(mask_num_2nd_user))
        batch_3rd_item_input = self._add_mask(self.padding_number_items, batch_3rd_item, max(mask_num_3rd_item))

        return batch_target_user, batch_kshot_item, batch_2nd_user_input, batch_3rd_item_input, batch_oracle_user_ebd, mask_num_2nd_user, mask_num_3rd_item

    '''
    mix item-task
    '''

    def get_positive_instances_item_task(self, random_seed):
        np.random.seed(random_seed)
        batch_num = self.oracle_num_items // self.batch_size_item + 1
        target_item, k_shot_user, second_order_items, oracle_item_ebd, mask_num_second_order_item, third_order_users, mask_num_third_order_user = [], [], [], [], [], [], []
        for batch in range(batch_num):
            b_target_i, b_k_shot_user, b_2nd_order_i, b_3rd_order_u, b_oracle_i_ebd, b_mask_num_2nd_i, b_mask_num_3rd_u = self._get_positive_batch_item_task(
                batch)

            target_item.append(b_target_i)
            k_shot_user.append(b_k_shot_user)
            second_order_items.append(b_2nd_order_i)
            oracle_item_ebd.append(b_oracle_i_ebd)
            mask_num_second_order_item.append(b_mask_num_2nd_i)

            third_order_users.append(b_3rd_order_u)
            mask_num_third_order_user.append(b_mask_num_3rd_u)

        return target_item, k_shot_user, second_order_items, third_order_users, oracle_item_ebd, mask_num_second_order_item, mask_num_third_order_user

    def _get_positive_batch_item_task(self, i):
        batch_target_item, batch_kshot_user, batch_2nd_item, batch_oracle_item_ebd, mask_num_2nd_item, batch_3rd_user, mask_num_3rd_user = [], [], [], [], [], [], []

        begin_index = i * self.batch_size_item
        end_index = min(begin_index + self.batch_size_item, self.oracle_num_items)
        for per_item_index in range(begin_index, end_index):
            target_item_id = self.oracle_item_list[per_item_index]

            per_oracle_item_ebd = self.oracle_item_ebd[target_item_id]
            if len(self.neighbor_dict_item_list[target_item_id]) < self.kshot_num:
                sample_kshot_user = np.random.choice(self.neighbor_dict_item_list[target_item_id], self.kshot_num,
                                                     replace=True)
            else:
                sample_kshot_user = np.random.choice(self.neighbor_dict_item_list[target_item_id], self.kshot_num,
                                                     replace=False)

            current_second_order_item = []
            for each_kshot_user in sample_kshot_user:

                candidate_second_order_item = self.neighbor_dict_user_list[each_kshot_user]

                if target_item_id in candidate_second_order_item:
                    candidate_second_order_item.remove(target_item_id)

                if len(candidate_second_order_item) == 0:
                    candidate_second_order_item.append(target_item_id)

                if len(candidate_second_order_item) < self.kshot_second_num:
                    temp_second_order_item = list(
                        np.random.choice(candidate_second_order_item, self.kshot_second_num, replace=True))
                else:
                    temp_second_order_item = list(
                        np.random.choice(candidate_second_order_item, self.kshot_second_num, replace=False))

                current_second_order_item += temp_second_order_item
            temp_second_order_item = list(set(current_second_order_item))

            current_third_order_user = []
            for each_kshot_3rd_item in temp_second_order_item:
                candidate_third_order_user = self.neighbor_dict_item_list[each_kshot_3rd_item]
                if len(candidate_third_order_user) < self.kshot_third_num:
                    temp_third_order_user = list(
                        np.random.choice(candidate_third_order_user, self.kshot_third_num, replace=True))
                else:
                    temp_third_order_user = list(
                        np.random.choice(candidate_third_order_user, self.kshot_third_num, replace=False))
                current_third_order_user += temp_third_order_user
            temp_third_order_user = list(set(current_third_order_user))

            batch_target_item.append(target_item_id)
            batch_kshot_user.append(sample_kshot_user)
            batch_2nd_item.append(temp_second_order_item)
            batch_oracle_item_ebd.append(per_oracle_item_ebd)
            mask_num_2nd_item.append(len(temp_second_order_item))

            batch_3rd_user.append(temp_third_order_user)
            mask_num_3rd_user.append(len(temp_third_order_user))

        batch_2nd_item_input = self._add_mask(self.padding_number_items, batch_2nd_item, max(mask_num_2nd_item))
        batch_3rd_user_input = self._add_mask(self.padding_number_users, batch_3rd_user, max(mask_num_3rd_user))

        return batch_target_item, batch_kshot_user, batch_2nd_item_input, batch_3rd_user_input, batch_oracle_item_ebd, mask_num_2nd_item, mask_num_3rd_user

    def _add_mask(self, feature_mask, features, num_max):

        # uniformalize the length of each batch
        for i in range(len(features)):
            features[i] = features[i] + [feature_mask] * (num_max - len(features[i]))
        return features

    def batch_gen_mix_user_task(self, batches, i):

        return [(batches[r])[i] for r in range(5)]

    def batch_gen_3rd_user_task(self, batches, i):
        return [(batches[r])[i] for r in range(7)]

    def batch_gen_mix_item_task(self, batches, i):

        return [(batches[r])[i] for r in range(5)]

    def batch_gen_3rd_item_task(self, batches, i):
        return [(batches[r])[i] for r in range(7)]


###########################      restore           #######################
'''
input: target is item, input k-shot users
'''


def generate_all_item_dict(all_rating):
    np.random.seed(0)
    all_item_dict = {}
    with open(all_rating, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(',')
            user, item = int(arr[0]), int(arr[1])
            if item not in all_item_dict:
                all_item_dict[item] = []
                all_item_dict[item].append(user)
            else:
                all_item_dict[item].append(user)
            line = f.readline()

    all_support_item_dict = {}

    for each_item in all_item_dict.keys():
        all_support_item_dict[each_item] = []
        if len(all_item_dict[each_item]) >= setting.support_num:
            select_instance = np.random.choice(all_item_dict[each_item], setting.support_num, replace=False)
        else:
            select_instance = np.random.choice(all_item_dict[each_item], setting.support_num, replace=True)
        for user in select_instance:
            all_support_item_dict[each_item].append(user)

    return all_support_item_dict


def generate_meta_all_item_set(all_support_item_dict):
    all_item_id, all_support_set_user = [], []
    for item_key in all_support_item_dict.keys():
        pos_user = all_support_item_dict[item_key]

        all_item_id.append(item_key)
        all_support_set_user.append(pos_user)

    all_item_num_instance = len(all_item_id)

    all_item_id = np.array(all_item_id)
    all_support_set_user = np.array(all_support_set_user)

    return all_item_id, all_support_set_user, all_item_num_instance


'''
input: target is user, input k-shot items
'''


def generate_all_user_dict(all_rating):
    np.random.seed(0)
    all_user_dict = {}
    with open(all_rating, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(',')
            user, item = int(arr[0]), int(arr[1])
            if user not in all_user_dict:
                all_user_dict[user] = []
                all_user_dict[user].append(item)
            else:
                all_user_dict[user].append(item)
            line = f.readline()

    all_support_user_dict = {}
    for each_user in all_user_dict.keys():
        all_support_user_dict[each_user] = []
        select_instance = np.random.choice(all_user_dict[each_user], setting.support_num, replace=False)
        for item in select_instance:
            all_support_user_dict[each_user].append(item)
    return all_support_user_dict


def generate_meta_all_user_set(all_support_user_dict):
    all_user_id, all_support_set_item = [], []
    for user_key in all_support_user_dict.keys():
        pos_item = all_support_user_dict[user_key]
        all_user_id.append(user_key)
        all_support_set_item.append(pos_item)

    all_num_instance = len(all_user_id)
    all_user_id = np.array(all_user_id)
    all_support_set_item = np.array(all_support_set_item)

    return all_user_id, all_support_set_item, all_num_instance


def batch_gen_task(batches, i, batch_size, num_instances):
    '''
    apply to train or valid dataset
    '''
    start_index = i * batch_size
    end_index = min(start_index + batch_size, num_instances)

    return [(batches[r])[start_index:end_index] for r in range(2)]

def batch_gen_task_cc(batches, i, batch_size, num_instances):
    '''
    apply to train or valid dataset
    '''
    start_index = i * batch_size
    end_index = min(start_index + batch_size, num_instances)

    return [(batches[r])[start_index:end_index] for r in range(3)]



'''
load ground truth
'''


def load_target_user_embedding(user_ebd_path):
    user_embedding = np.load(user_ebd_path)

    return user_embedding


def load_target_item_embedding(item_ebd_path):
    item_embedding = np.load(item_ebd_path)

    return item_embedding


'''
user task: personalized recommendation
'''


def generate_user_dict_valid(valid_rating):
    valid_user_id = []
    with open(valid_rating, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(',')
            user, item = int(arr[0]), int(arr[1])
            if user not in valid_user_id:
                valid_user_id.append(user)

            line = f.readline()

    valid_num_instance = len(valid_user_id)

    return valid_user_id, valid_num_instance

def generate_user_dict_valid_meta(valid_rating):
    valid_user_dict = {}
    with open(valid_rating, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(',')
            user, item = int(arr[0]), int(arr[1])
            if user not in valid_user_dict:
                valid_user_dict[user] = []
                valid_user_dict[user].append(item)
            else:
                valid_user_dict[user].append(item)

            line = f.readline()

    np.random.seed(0)
    valid_support_user_dict = {}
    for valid_user in valid_user_dict.keys():
        valid_support_user_dict[valid_user] = []
        select_instance = np.random.choice(valid_user_dict[valid_user], setting.support_num, replace=False)
        for item in select_instance:
            valid_support_user_dict[valid_user].append(item)

    return valid_support_user_dict

def generate_user_dict_train(train_rating):
    train_user_id = []
    with open(train_rating, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(',')
            user, item = int(arr[0]), int(arr[1])
            if user not in train_user_id:
                train_user_id.append(user)
            line = f.readline()

    train_num_instance = len(train_user_id)
    train_user_id = np.array(train_user_id)

    return train_user_id, train_num_instance

def generate_user_dict_train_meta(train_rating):
    train_user_dict = {}
    with open(train_rating, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(',')
            user, item = int(arr[0]), int(arr[1])
            if user not in train_user_dict:
                train_user_dict[user] = []
                train_user_dict[user].append(item)
            else:
                train_user_dict[user].append(item)

            line = f.readline()

    train_support_user_dict = {}

    for train_user in train_user_dict.keys():
        train_support_user_dict[train_user] = []
        if len(train_user_dict[train_user]) > setting.support_num:
            select_instance = np.random.choice(train_user_dict[train_user], setting.support_num, replace=False)
        else:
            select_instance = np.random.choice(train_user_dict[train_user], setting.support_num, replace=True)

        for item in select_instance:
            train_support_user_dict[train_user].append(item)

    return train_support_user_dict


def generate_meta_train_user_set(train_support_user_dict):
    '''
    generate meta training/valid set
    '''
    target_user_ebd = load_target_user_embedding(setting.pretrain_user_ebd_path)

    train_user_id, train_support_set_item, train_target_user = [], [], []
    for user_key in train_support_user_dict.keys():
        pos_item = train_support_user_dict[user_key]
        user_ebd = target_user_ebd[user_key]

        train_user_id.append(user_key)
        train_support_set_item.append(pos_item)
        train_target_user.append(user_ebd)

    train_num_instance = len(train_user_id)

    train_user_id = np.array(train_user_id)
    train_support_set_item = np.array(train_support_set_item)
    train_target_user = np.array(train_target_user)

    return train_user_id, train_support_set_item, train_target_user, train_num_instance


def generate_meta_valid_user_set(valid_support_user_dict):
    target_user_ebd = load_target_user_embedding(setting.pretrain_user_ebd_path)

    valid_user_id, valid_support_set_item, valid_target_user = [], [], []
    for user_key in valid_support_user_dict.keys():
        pos_item = valid_support_user_dict[user_key]
        user_ebd = target_user_ebd[user_key]

        valid_user_id.append(user_key)
        valid_support_set_item.append(pos_item)
        valid_target_user.append(user_ebd)

    valid_num_instance = len(valid_user_id)

    return valid_user_id, valid_support_set_item, valid_target_user, valid_num_instance


def batch_gen_user_task(batches, i, batch_size):
    '''
    apply to train or valid dataset
    '''
    max_instances = batches[3]
    start_index = i * batch_size
    end_index = min(start_index + batch_size, max_instances)

    return [(batches[r])[start_index:end_index] for r in range(3)]

def batch_gen_bert_task(batches, i, batch_size):
    '''
    apply to train or valid dataset
    '''
    max_instances = len(batches[0])
    start_index = i * batch_size
    end_index = min(start_index + batch_size, max_instances)

    return [(batches[r])[start_index:end_index] for r in range(3)]


#################################################################
##########       item-task                         ##############


def generate_item_dict_valid_meta(valid_rating):
    valid_item_dict = {}
    with open(valid_rating, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(',')
            user, item = int(arr[0]), int(arr[1])
            if item not in valid_item_dict:
                valid_item_dict[item] = []
                valid_item_dict[item].append(user)
            else:
                valid_item_dict[item].append(user)

            line = f.readline()

    valid_support_item_dict = {}
    np.random.seed(0)
    for valid_item in valid_item_dict.keys():
        valid_support_item_dict[valid_item] = []
        if len(valid_item_dict[valid_item]) >= setting.support_num:
            select_instance = np.random.choice(valid_item_dict[valid_item], setting.support_num, replace=False)
        else:
            select_instance = np.random.choice(valid_item_dict[valid_item], setting.support_num, replace=True)
        for user in select_instance:
            valid_support_item_dict[valid_item].append(user)

    return valid_support_item_dict


def generate_item_dict_train_meta(train_rating):
    train_item_dict = {}
    with open(train_rating, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(',')
            user, item = int(arr[0]), int(arr[1])
            if item not in train_item_dict:
                train_item_dict[item] = []
                train_item_dict[item].append(user)
            else:
                train_item_dict[item].append(user)

            line = f.readline()

    train_support_item_dict = {}

    for train_item in train_item_dict.keys():
        train_support_item_dict[train_item] = []
        if len(train_item_dict[train_item]) < setting.support_num:
            select_instance = np.random.choice(train_item_dict[train_item], setting.support_num, replace=True)
        else:
            select_instance = np.random.choice(train_item_dict[train_item], setting.support_num, replace=False)
        for user in select_instance:
            train_support_item_dict[train_item].append(user)

    return train_support_item_dict


def generate_item_dict_valid(valid_rating):
    valid_item_id = []
    with open(valid_rating, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(',')
            user, item = int(arr[0]), int(arr[1])
            if item not in valid_item_id:
                valid_item_id.append(item)
            line = f.readline()

    valid_num_instance = len(valid_item_id)
    valid_item_id = np.array(valid_item_id)

    return valid_item_id, valid_num_instance

def generate_item_dict_train(train_rating):

    train_item_id = []
    with open(train_rating, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(',')
            user, item = int(arr[0]), int(arr[1])
            if item not in train_item_id:
                train_item_id.append(item)
            line = f.readline()

    train_num_instance = len(train_item_id)
    train_item_id = np.array(train_item_id)

    return train_item_id, train_num_instance


def generate_meta_train_item_set(train_support_item_dict):
    '''
    generate meta training/valid set
    '''
    target_item_ebd = load_target_user_embedding(setting.pretrain_item_ebd_path)

    train_item_id, train_support_set_user, train_target_item = [], [], []
    for item_key in train_support_item_dict.keys():
        pos_user = train_support_item_dict[item_key]
        item_ebd = target_item_ebd[item_key]

        train_item_id.append(item_key)
        train_support_set_user.append(pos_user)
        train_target_item.append(item_ebd)

    train_num_instance = len(train_item_id)

    train_item_id = np.array(train_item_id)
    train_support_set_user = np.array(train_support_set_user)
    train_target_item = np.array(train_target_item)

    return train_item_id, train_support_set_user, train_target_item, train_num_instance


def generate_meta_valid_item_set(valid_support_item_dict):
    target_item_ebd = load_target_user_embedding(setting.pretrain_item_ebd_path)

    valid_item_id, valid_support_set_user, valid_target_item = [], [], []
    for item_key in valid_support_item_dict.keys():
        pos_user = valid_support_item_dict[item_key]
        item_ebd = target_item_ebd[item_key]

        valid_item_id.append(item_key)
        valid_support_set_user.append(pos_user)
        valid_target_item.append(item_ebd)

    valid_num_instance = len(valid_item_id)

    return valid_item_id, valid_support_set_user, valid_target_item, valid_num_instance


def batch_gen_item_task(batches, i, batch_size):
    '''
    apply to train or valid dataset
    '''
    max_instances = batches[3]
    start_index = i * batch_size
    end_index = min(start_index + batch_size, max_instances)

    return [(batches[r])[start_index:end_index] for r in range(3)]

def batch_gen_contrastive_user_task(batches, i, batch_size):
    '''
    apply to train or valid dataset
    '''
    max_instances = batches[5]
    start_index = i * batch_size
    end_index = min(start_index + batch_size, max_instances)

    return [(batches[r])[start_index:end_index] for r in range(5)]


def batch_gen_contrastive_item_task(batches, i, batch_size):
    '''
    apply to train or valid dataset
    '''
    max_instances = batches[5]
    start_index = i * batch_size
    end_index = min(start_index + batch_size, max_instances)

    return [(batches[r])[start_index:end_index] for r in range(5)]


def batch_gen_contrastive_gnn_user_task(batches, i, batch_size):
    '''
    apply to train or valid dataset
    '''
    start_index = i * batch_size
    end_index = start_index + batch_size

    return [(batches[r])[start_index:end_index] for r in range(1)]



def batch_gen_contrastive_gnn_item_task(batches, i, batch_size):
    '''
    apply to train or valid dataset
    '''
    start_index = i * batch_size
    end_index = start_index + batch_size

    return [(batches[r])[start_index:end_index] for r in range(1)]



###########################################################################
############################## contrastive task ###########################

# def generate_user_dict_train_contrastive(train_rating):
#     train_user_dict = {}
#     with open(train_rating, 'r') as f:
#         line = f.readline()
#         while line != "" and line != None:
#             arr = line.strip().split(',')
#             user, item = int(arr[0]), int(arr[1])
#             if user not in train_user_dict:
#                 train_user_dict[user] = []
#                 train_user_dict[user].append(item)
#             else:
#                 train_user_dict[user].append(item)

#             line = f.readline()

#     train_support_user_dict, train_pos_query_user_dict, train_neg_query_user_dict = {}, {}, {}

#     for train_user in train_user_dict.keys():
#         train_support_user_dict[train_user] = []
#         if len(train_user_dict[train_user]) > setting.support_num:
#             select_instance = np.random.choice(train_user_dict[train_user], setting.support_num, replace=False)
#         else:
#             select_instance = np.random.choice(train_user_dict[train_user], setting.support_num, replace=True)

#         for item in select_instance:
#             train_support_user_dict[train_user].append(item)

#         train_pos_query_user_dict[train_user] = []
#         remain_pos_instance = list(set(train_user_dict[train_user]) - set(list(select_instance)))
#         if len(remain_pos_instance) == 0:
#             remain_pos_instance = list(select_instance)
#         if len(remain_pos_instance) > setting.support_num:
#             remain_pos_instance_ = np.random.choice(remain_pos_instance, setting.support_num, replace=False)
#         else:
#             remain_pos_instance_ = np.random.choice(remain_pos_instance, setting.support_num, replace=True)

#             # the length of remain pos instance is the same as train support user dict instance
#         for item in remain_pos_instance_:
#             train_pos_query_user_dict[train_user].append(item)

#         # the length of negative instance is the same as train support user dict instance
#         train_neg_query_user_dict[train_user] = []
#         for _ in range(setting.support_num):
#             j = np.random.randint(setting.num_items)
#             while j in train_user_dict[train_user]:
#                 j = np.random.randint(setting.num_items)
#             train_neg_query_user_dict[train_user].append(j)

#     return train_support_user_dict, train_pos_query_user_dict, train_neg_query_user_dict


# def generate_user_dict_valid_contrastive(valid_rating):
#     valid_user_dict = {}
#     with open(valid_rating, 'r') as f:
#         line = f.readline()
#         while line != "" and line != None:
#             arr = line.strip().split(',')
#             user, item = int(arr[0]), int(arr[1])
#             if user not in valid_user_dict:
#                 valid_user_dict[user] = []
#                 valid_user_dict[user].append(item)
#             else:
#                 valid_user_dict[user].append(item)

#             line = f.readline()

#     np.random.seed(0)
#     valid_support_user_dict, valid_pos_query_user_dict, valid_neg_query_user_dict = {}, {}, {}
#     for valid_user in valid_user_dict.keys():
#         valid_support_user_dict[valid_user] = []
#         if len(valid_user_dict[valid_user]) > setting.support_num:
#             select_instance = np.random.choice(valid_user_dict[valid_user], setting.support_num, replace=False)
#         else:
#             select_instance = np.random.choice(valid_user_dict[valid_user], setting.support_num, replace=True)
#         for item in select_instance:
#             valid_support_user_dict[valid_user].append(item)

#         valid_pos_query_user_dict[valid_user] = []
#         remain_pos_instance = list(set(valid_user_dict[valid_user]) - set(list(select_instance)))
#         if len(remain_pos_instance) == 0:
#             remain_pos_instance = list(select_instance)
#         if len(remain_pos_instance) > setting.support_num:
#             remain_pos_instance_ = np.random.choice(remain_pos_instance, setting.support_num, replace=False)
#         else:
#             remain_pos_instance_ = np.random.choice(remain_pos_instance, setting.support_num, replace=True)

#             # the length of remain pos instance is the same as train support user dict instance
#         for item in remain_pos_instance_:
#             valid_pos_query_user_dict[valid_user].append(item)

#         # the length of negative instance is the same as train support user dict instance
#         valid_neg_query_user_dict[valid_user] = []
#         for _ in range(setting.support_num):
#             j = np.random.randint(setting.num_items)
#             while j in valid_user_dict[valid_user]:
#                 j = np.random.randint(setting.num_items)
#             valid_neg_query_user_dict[valid_user].append(j)

#     return valid_support_user_dict, valid_pos_query_user_dict, valid_neg_query_user_dict


# def generate_contrastive_train_user_set(train_support_user_dict, train_pos_query_user_dict, train_neg_query_user_dict,
#                                         train_rating):
#     '''
#     generate meta training/valid set
#     '''
#     train_user_dict = {}
#     with open(train_rating, 'r') as f:
#         line = f.readline()
#         while line != "" and line != None:
#             arr = line.strip().split(',')
#             user, item = int(arr[0]), int(arr[1])
#             if user not in train_user_dict:
#                 train_user_dict[user] = []
#                 train_user_dict[user].append(item)
#             else:
#                 train_user_dict[user].append(item)

#             line = f.readline()

#     target_user_ebd = load_target_user_embedding(setting.oracle_user_ebd_path)

#     train_user_id, train_support_set_item, train_query_pos_item, train_query_neg_item, train_target_user = [], [], [], [], []
#     for user_key in train_support_user_dict.keys():
#         support_item = train_support_user_dict[user_key]
#         user_ebd = target_user_ebd[user_key]

#         for index in range(setting.support_num):
#             pos_item = train_pos_query_user_dict[user_key][index]
#             neg_item = train_neg_query_user_dict[user_key][index]

#             train_user_id.append(user_key)
#             train_support_set_item.append(support_item)
#             train_query_pos_item.append(pos_item)
#             train_query_neg_item.append(neg_item)
#             train_target_user.append(user_ebd)

#     train_num_instance = len(train_user_id)

#     train_user_id = np.array(train_user_id)
#     train_support_set_item = np.array(train_support_set_item)
#     train_query_pos_item = np.array(train_query_pos_item)
#     train_query_neg_item = np.array(train_query_neg_item)
#     train_target_user = np.array(train_target_user)

#     return train_user_id, train_support_set_item, train_query_pos_item, train_query_neg_item, train_target_user, train_num_instance


# def generate_contrastive_valid_user_set(valid_support_user_dict, valid_pos_query_user_dict, valid_neg_query_user_dict,
#                                         valid_rating):
#     valid_user_dict = {}
#     with open(valid_rating, 'r') as f:
#         line = f.readline()
#         while line != "" and line != None:
#             arr = line.strip().split(',')
#             user, item = int(arr[0]), int(arr[1])
#             if user not in valid_user_dict:
#                 valid_user_dict[user] = []
#                 valid_user_dict[user].append(item)
#             else:
#                 valid_user_dict[user].append(item)

#             line = f.readline()

#     target_user_ebd = load_target_user_embedding(setting.oracle_user_ebd_path)

#     valid_user_id, valid_support_set_item, valid_query_pos_item, valid_query_neg_item, valid_target_user = [], [], [], [], []
#     for user_key in valid_support_user_dict.keys():
#         support_item = valid_support_user_dict[user_key]
#         user_ebd = target_user_ebd[user_key]

#         for index in range(setting.support_num):
#             pos_item = valid_pos_query_user_dict[user_key][index]
#             neg_item = valid_neg_query_user_dict[user_key][index]

#             valid_user_id.append(user_key)
#             valid_support_set_item.append(support_item)
#             valid_query_pos_item.append(pos_item)
#             valid_query_neg_item.append(neg_item)
#             valid_target_user.append(user_ebd)

#     valid_num_instance = len(valid_user_id)

#     valid_user_id = np.array(valid_user_id)
#     valid_support_set_item = np.array(valid_support_set_item)
#     valid_query_pos_item = np.array(valid_query_pos_item)
#     valid_query_neg_item = np.array(valid_query_neg_item)
#     valid_target_user = np.array(valid_target_user)

#     return valid_user_id, valid_support_set_item, valid_query_pos_item, valid_query_neg_item, valid_target_user, valid_num_instance


# def generate_contrastive_train_item_set(train_support_item_dict, train_pos_query_item_dict, train_neg_query_item_dict,
#                                         train_rating):
#     '''
#     generate meta training/valid set
#     '''
#     train_item_dict = {}
#     with open(train_rating, 'r') as f:
#         line = f.readline()
#         while line != "" and line != None:
#             arr = line.strip().split(',')
#             user, item = int(arr[0]), int(arr[1])
#             if item not in train_item_dict:
#                 train_item_dict[item] = []
#                 train_item_dict[item].append(user)
#             else:
#                 train_item_dict[item].append(user)

#             line = f.readline()

#     target_item_ebd = load_target_item_embedding(setting.oracle_item_ebd_path)

#     train_item_id, train_support_set_user, train_query_pos_user, train_query_neg_user, train_target_item = [], [], [], [], []

#     for item_key in train_support_item_dict.keys():
#         support_user = train_support_item_dict[item_key]
#         item_ebd = target_item_ebd[item_key]

#         for index in range(setting.support_num):
#             pos_user = train_pos_query_item_dict[item_key][index]
#             neg_user = train_neg_query_item_dict[item_key][index]

#             train_item_id.append(item_key)
#             train_support_set_user.append(support_user)
#             train_query_pos_user.append(pos_user)
#             train_query_neg_user.append(neg_user)
#             train_target_item.append(item_ebd)

#     train_num_instance = len(train_item_id)

#     train_item_id = np.array(train_item_id)
#     train_support_set_user = np.array(train_support_set_user)
#     train_query_pos_user = np.array(train_query_pos_user)
#     train_query_neg_user = np.array(train_query_neg_user)
#     train_target_item = np.array(train_target_item)

#     return train_item_id, train_support_set_user, train_query_pos_user, train_query_neg_user, train_target_item, train_num_instance


# def generate_contrastive_valid_item_set(valid_support_item_dict, valid_pos_query_item_dict, valid_neg_query_item_dict,
#                                         valid_rating):
#     valid_item_dict = {}
#     with open(valid_rating, 'r') as f:
#         line = f.readline()
#         while line != "" and line != None:
#             arr = line.strip().split(',')
#             user, item = int(arr[0]), int(arr[1])
#             if item not in valid_item_dict:
#                 valid_item_dict[item] = []
#                 valid_item_dict[item].append(user)
#             else:
#                 valid_item_dict[item].append(user)

#             line = f.readline()

#     target_item_ebd = load_target_item_embedding(setting.oracle_item_ebd_path)

#     valid_item_id, valid_support_set_user, valid_query_pos_user, valid_query_neg_user, valid_target_item = [], [], [], [], []
#     for item_key in valid_support_item_dict.keys():
#         support_user = valid_support_item_dict[item_key]
#         item_ebd = target_item_ebd[item_key]

#         for index in range(setting.support_num):
#             pos_user = valid_pos_query_item_dict[item_key][index]
#             neg_user = valid_neg_query_item_dict[item_key][index]

#             valid_item_id.append(item_key)
#             valid_support_set_user.append(support_user)
#             valid_query_pos_user.append(pos_user)
#             valid_query_neg_user.append(neg_user)
#             valid_target_item.append(item_ebd)

#     valid_num_instance = len(valid_item_id)

#     valid_item_id = np.array(valid_item_id)
#     valid_support_set_user = np.array(valid_support_set_user)
#     valid_query_pos_user = np.array(valid_query_pos_user)
#     valid_query_neg_user = np.array(valid_query_neg_user)
#     valid_target_item = np.array(valid_target_item)

#     return valid_item_id, valid_support_set_user, valid_query_pos_user, valid_query_neg_user, valid_target_item, valid_num_instance


# def generate_item_dict_train_contrastive(train_rating):
#     train_item_dict = {}
#     with open(train_rating, 'r') as f:
#         line = f.readline()
#         while line != "" and line != None:
#             arr = line.strip().split(',')
#             user, item = int(arr[0]), int(arr[1])
#             if item not in train_item_dict:
#                 train_item_dict[item] = []
#                 train_item_dict[item].append(user)
#             else:
#                 train_item_dict[item].append(user)

#             line = f.readline()

#     train_support_item_dict, train_pos_query_item_dict, train_neg_query_item_dict = {}, {}, {}

#     for train_item in train_item_dict.keys():
#         train_support_item_dict[train_item] = []
#         if len(train_item_dict[train_item]) > setting.support_num:
#             select_instance = np.random.choice(train_item_dict[train_item], setting.support_num, replace=False)
#         else:
#             select_instance = np.random.choice(train_item_dict[train_item], setting.support_num, replace=True)

#         if len(select_instance) == 0:
#             contine

#         for user in select_instance:
#             train_support_item_dict[train_item].append(user)

#         train_pos_query_item_dict[train_item] = []
#         remain_pos_instance = list(set(train_item_dict[train_item]) - set(list(select_instance)))
        
#         if len(remain_pos_instance) == 0:
#             train_pos_query_item_dict[train_item] = list(select_instance)

#             # the length of negative instance is the same as train support user dict instance
#             train_neg_query_item_dict[train_item] = []
#             for _ in range(setting.support_num):
#                 j = np.random.randint(setting.num_users)
#                 while j in train_item_dict[train_item]:
#                     j = np.random.randint(setting.num_users)
#                 train_neg_query_item_dict[train_item].append(j)

#             continue


#         if len(remain_pos_instance) > setting.support_num:
#             remain_pos_instance_ = np.random.choice(remain_pos_instance, setting.support_num, replace=False)
#         else:
#             remain_pos_instance_ = np.random.choice(remain_pos_instance, setting.support_num, replace=True)

#             # the length of remain pos instance is the same as train support user dict instance
#         for user in remain_pos_instance_:
#             train_pos_query_item_dict[train_item].append(user)

#         # the length of negative instance is the same as train support user dict instance
#         train_neg_query_item_dict[train_item] = []
#         for _ in range(setting.support_num):
#             j = np.random.randint(setting.num_users)
#             while j in train_item_dict[train_item]:
#                 j = np.random.randint(setting.num_users)
#             train_neg_query_item_dict[train_item].append(j)

#     return train_support_item_dict, train_pos_query_item_dict, train_neg_query_item_dict


# def generate_item_dict_valid_contrastive(valid_rating):
#     valid_item_dict = {}
#     with open(valid_rating, 'r') as f:
#         line = f.readline()
#         while line != "" and line != None:
#             arr = line.strip().split(',')
#             user, item = int(arr[0]), int(arr[1])
#             if item not in valid_item_dict:
#                 valid_item_dict[item] = []
#                 valid_item_dict[item].append(user)
#             else:
#                 valid_item_dict[item].append(user)

#             line = f.readline()

#     np.random.seed(0)
#     valid_support_item_dict, valid_pos_query_item_dict, valid_neg_query_item_dict = {}, {}, {}
#     for valid_item in valid_item_dict.keys():
#         valid_support_item_dict[valid_item] = []
#         if len(valid_item_dict[valid_item]) > setting.support_num:
#             select_instance = np.random.choice(valid_item_dict[valid_item], setting.support_num, replace=False)
#         else:
#             select_instance = np.random.choice(valid_item_dict[valid_item], setting.support_num, replace=True)
#         for user in select_instance:
#             valid_support_item_dict[valid_item].append(user)

#         valid_pos_query_item_dict[valid_item] = []
#         remain_pos_instance = list(set(valid_item_dict[valid_item]) - set(list(select_instance)))
#         if len(remain_pos_instance) > setting.support_num:
#             remain_pos_instance_ = np.random.choice(remain_pos_instance, setting.support_num, replace=False)
#         else:
#             remain_pos_instance_ = np.random.choice(remain_pos_instance, setting.support_num, replace=True)

#             # the length of remain pos instance is the same as train support user dict instance
#         for user in remain_pos_instance_:
#             valid_pos_query_item_dict[valid_item].append(user)

#         # the length of negative instance is the same as train support user dict instance
#         valid_neg_query_item_dict[valid_item] = []
#         for _ in range(setting.support_num):
#             j = np.random.randint(setting.num_users)
#             while j in valid_item_dict[valid_item]:
#                 j = np.random.randint(setting.num_users)
#             valid_neg_query_item_dict[valid_item].append(j)

#     return valid_support_item_dict, valid_pos_query_item_dict, valid_neg_query_item_dict


# def generate_contrastive_all_user_set(train_support_user_dict, train_pos_query_user_dict, train_neg_query_user_dict,
#                                         train_rating):
#     '''
#     generate meta training/valid set
#     '''
#     train_user_dict = {}
#     with open(train_rating, 'r') as f:
#         line = f.readline()
#         while line != "" and line != None:
#             arr = line.strip().split(',')
#             user, item = int(arr[0]), int(arr[1])
#             if user not in train_user_dict:
#                 train_user_dict[user] = []
#                 train_user_dict[user].append(item)
#             else:
#                 train_user_dict[user].append(item)

#             line = f.readline()

#     target_user_ebd = load_target_user_embedding(setting.oracle_user_ebd_path)

#     train_user_id, train_support_set_item, train_query_pos_item, train_query_neg_item, train_target_user = [], [], [], [], []
#     for user_key in train_support_user_dict.keys():
#         support_item = train_support_user_dict[user_key]
#         user_ebd = target_user_ebd[user_key]

#         index = 0

#         pos_item = train_pos_query_user_dict[user_key][index]
#         neg_item = train_neg_query_user_dict[user_key][index]

#         train_user_id.append(user_key)
#         train_support_set_item.append(support_item)
#         train_query_pos_item.append(pos_item)
#         train_query_neg_item.append(neg_item)
#         train_target_user.append(user_ebd)

#     train_num_instance = len(train_user_id)

#     train_user_id = np.array(train_user_id)
#     train_support_set_item = np.array(train_support_set_item)
#     train_query_pos_item = np.array(train_query_pos_item)
#     train_query_neg_item = np.array(train_query_neg_item)
#     train_target_user = np.array(train_target_user)

#     return train_user_id, train_support_set_item, train_query_pos_item, train_query_neg_item, train_target_user, train_num_instance

# def generate_contrastive_all_item_set(train_support_item_dict, train_pos_query_item_dict, train_neg_query_item_dict,
#                                         train_rating):
#     '''
#     generate meta training/valid set
#     '''
#     train_item_dict = {}
#     with open(train_rating, 'r') as f:
#         line = f.readline()
#         while line != "" and line != None:
#             arr = line.strip().split(',')
#             user, item = int(arr[0]), int(arr[1])
#             if item not in train_item_dict:
#                 train_item_dict[item] = []
#                 train_item_dict[item].append(user)
#             else:
#                 train_item_dict[item].append(user)

#             line = f.readline()

#     target_item_ebd = load_target_item_embedding(setting.oracle_item_ebd_path)

#     train_item_id, train_support_set_user, train_query_pos_user, train_query_neg_user, train_target_item = [], [], [], [], []

#     for item_key in train_support_item_dict.keys():
#         support_user = train_support_item_dict[item_key]
#         item_ebd = target_item_ebd[item_key]

#         index = 0
#         pos_user = train_pos_query_item_dict[item_key][index]
#         neg_user = train_neg_query_item_dict[item_key][index]

#         train_item_id.append(item_key)
#         train_support_set_user.append(support_user)
#         train_query_pos_user.append(pos_user)
#         train_query_neg_user.append(neg_user)
#         train_target_item.append(item_ebd)

#     train_num_instance = len(train_item_id)

#     train_item_id = np.array(train_item_id)
#     train_support_set_user = np.array(train_support_set_user)
#     train_query_pos_user = np.array(train_query_pos_user)
#     train_query_neg_user = np.array(train_query_neg_user)
#     train_target_item = np.array(train_target_item)

#     return train_item_id, train_support_set_user, train_query_pos_user, train_query_neg_user, train_target_item, train_num_instance
def cos(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

##################################  Path generate ##############################################
import random
import networkx as nx


class BasicWalker:
    def __init__(self, G, workers):
        self.G = G.G
        self.node_size = G.node_size
        self.look_up_dict = G.look_up_dict

    def deepwalk_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def deepwalk_walk_test(self, walk_length, start_node, mask_index):
        '''
        Simulate a random walk starting from start node.
        mask index is target user/item position, we use [mask] symbol
        '''
        G = self.G
        walk = [start_node]

        while len(walk) < walk_length:
            while len(walk) - 1 < mask_index:
                cur = walk[-1]
                cur_nbrs = list(G.neighbors(cur))
                if len(cur_nbrs) > 0:
                    walk.append(random.choice(cur_nbrs))
                else:
                    print(cur)  # 没有孤立点
                    break
            # if mask_index ==1 or mask_index==2:
            #     print(walk)
            if len(walk) - 1 == mask_index:
                walk = list(reversed(walk))
            # if mask_index ==1 or mask_index ==2:
            #     print(walk)

            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                print(cur)  # 没有孤立点
                break
        if mask_index == walk_length - 1:
            walk = walk[:-1]

        walk.append(mask_index)
        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            # pool = multiprocessing.Pool(processes = 4)
            # print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                # walks.append(pool.apply_async(deepwalk_walk_wrapper, (self, walk_length, node, )))
                walks.append(self.deepwalk_walk(
                    walk_length=walk_length, start_node=node))
            # pool.close()
            # pool.join()
        # print(len(walks))
        return walks

    def simulate_walks_test(self, num_walks, walk_length, mask_index):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration, mask index is %d:' % mask_index)
        for walk_iter in range(num_walks):
            # pool = multiprocessing.Pool(processes = 4)
            # print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                # print(node)
                # walks.append(pool.apply_async(deepwalk_walk_wrapper, (self, walk_length, node, )))
                walks.append(self.deepwalk_walk_test(
                    walk_length=walk_length, start_node=node, mask_index=mask_index))
            # pool.close()
            # pool.join()
        # print(len(walks))
        return walks


class Graph(object):
    def __init__(self):
        self.G = None
        self.look_up_dict = {}
        self.node_size = 0
        self.look_back_list = []

    def encode_node(self):
        for node in self.G.nodes():
            self.look_up_dict[node] = self.node_size
            self.look_back_list.append(node)
            self.node_size += 1

    def read_adjlist(self, filename):
        """ Read graph from adjacency file in which the edge must be unweighted
        the format of each line: v1 n1 n2 n3 ... nk
        :param filename: the filename of input file
        """
        self.G = nx.read_adjlist(filename, create_using=nx.DiGraph())
        for i, j in self.G.edges():
            self.G[i][j]['weight'] = 1.0

        self.encode_node()

    def read_edgelist(self, filename, weighted=False, directed=False):
        self.G = nx.DiGraph()

        if directed:
            def read_unweighted(l):
                src, dst = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = float(w)
        else:
            def read_unweighted(l):
                src, dst = l.split()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = 1.0
                self.G[dst][src]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = float(w)
                self.G[dst][src]['weight'] = float(w)
        fin = open(filename, 'r')
        func = read_unweighted
        if weighted:
            func = read_weighted
        while 1:
            l = fin.readline()
            if l == '':
                break
            func(l)
        fin.close()
        self.encode_node()


class DeepWalk(object):
    def __init__(self, graph, path_length, num_paths, dim, **kwargs):

        kwargs["workers"] = kwargs.get("workers", 1)
        kwargs["hs"] = 1
        self.graph = graph
        self.walker = BasicWalker(graph, workers=kwargs["workers"])
        # self.sentences = self.walker.simulate_walks(num_walks=num_paths, walk_length=path_length)
        args = Model_Config()
        with open(args.user_item_path, 'w') as writer:
            for mask_index in range(path_length):
                exist_user_item = list(range(args.vocab_size))

                self.sentences_all = self.walker.simulate_walks_test(num_walks=num_paths, walk_length=path_length,
                                                                     mask_index=mask_index)

                # writer.write('first' + ',' + 'second' + ',' + 'third' + ',' + 'fourth' + ',' + 'fifth' + ',' + 'sixth' + ',' + 'seventh' + ',' + 'eighth' + ',' + 'maskid' + '\n')
                for list1 in self.sentences_all:
                    exist_user_item.remove(int(list1[mask_index]))
                    for index in range(len(list1)):
                        if index!= len(list1)-1:
                            writer.write(str(list1[index]) + ',')
                        else:
                            writer.write(str(list1[index]) + '\n')


                for item in exist_user_item:
                    for index in range(path_length):

                        writer.write(str(item) + ',')
                    writer.write(str(mask_index) + '\n')

def gen_batch_user_item_path(args, rating_path):
    input_ids, masked_positions, masked_ids, gen_ebd_mask_position = [], [], [], []
    with open(rating_path, 'r') as f:
        line = f.readline()
        while line!="" and line!=None:
            current_list = []
            arr = line.strip().split(',')
            masked_ = int(arr[-1])
            masked_ids.append(int(arr[masked_]))
            for index in range(len(arr[:-1])):
                if index == int(arr[-1]):
                    current_list.append(args.num_users + args.num_items + 2) # [MASK]
                else:
                    current_list.append(int(arr[index]))

            input_ids.append(current_list)
            masked_positions.append(int(arr[-1]))
            line = f.readline()

    return input_ids, masked_positions, masked_ids


##################################  Path generate  Contrastive #########################################################
    # generate basic anchor meta path                                                             ######################
    # g = Graph()  g.read_adjlist(filename=args.input)                                           #######################
    # model = DeepWalk(graph=g, path_length=args.walk_length,num_paths=args.number_walks, dim=args.representation_size,#
    #         workers=args.workers, window=args.window_size)

def replace_delete_meta_path():
    user_item_dict = {}
    with open(setting.data_path + setting.dataset + '/train.txt', 'r') as f:
        line = f.readline()
        while line!="" and line!=None:
            arr = line.strip().split(' ')
            user = int(arr[0])
            items = arr[1:]
            user_item_dict[user] = []
            for item in items:
                user_item_dict[user].append(int(item)+setting.num_users)
                if item not in user_item_dict:
                    user_item_dict[int(item)+setting.num_users] = []
                    user_item_dict[int(item)+setting.num_users].append(user)
                else:
                    user_item_dict[int(item)+setting.num_users].append(user)
            line = f.readline()

    with open(setting.user_item_path, 'r') as f:
        with open(setting.user_item_replace_path, 'w') as writer_replace:
            with open(setting.user_item_delete_path, 'w') as writer_delete:
                line = f.readline()
                while line!="" and line!=None:
                    current_replace_list, current_delete_list = [], []
                    arr = line.strip().split(',')
                    for index in range(len(arr)):
                        if index!= len(arr) -1 and index!= int(arr[-1]):
                            prob = np.random.random()
                            if prob < setting.replace_ratio:
                                if int(arr[index]) not in user_item_dict:
                                    current_replace_list.append(arr[index])
                                else:
                                    current_replace_list.append(np.random.choice(user_item_dict[int(arr[index])], 1)[0])

                            else:
                                current_replace_list.append(arr[index])
                                current_delete_list.append(arr[index])

                        else:
                            current_replace_list.append(arr[index])
                            current_delete_list.append(arr[index])

                    for index in range(len(current_replace_list)):
                        if index != len(current_replace_list) - 1:
                            writer_replace.write(str(current_replace_list[index]) + ',')
                        else:
                            writer_replace.write(str(current_replace_list[index]) + '\n')

                    # padding
                    while len(current_delete_list) < setting.walk_length +1:
                        index_flag = current_delete_list[-1]
                        padding_flag = setting.num_users+setting.num_items+1
                        current_delete_list[-1] = padding_flag
                        current_delete_list.append(index_flag)

                    for index in range(len(current_delete_list)):
                        if index != len(current_delete_list) - 1:
                            writer_delete.write(str(current_delete_list[index]) + ',')
                        else:
                            writer_delete.write(str(current_delete_list[index]) + '\n')

                    line = f.readline()



if __name__ == '__main__':

    args = Model_Config()
    # g = Graph()
    # print('reading user item training bipartite graph...')
    # g.read_adjlist(filename=args.input)
    #
    # model = DeepWalk(graph=g, path_length=args.walk_length,
    #                           num_paths=args.number_walks, dim=args.representation_size,
    #                           workers=args.workers, window=args.window_size)
    #
    #
    # path_batches =  gen_batch_user_item_path(args)
    #
    # num_batch = len(path_batches[0]) // args.batch_size + 1
    # batch_index = range(num_batch)
    #
    # for index in batch_index:
    #     input_ids, masked_positions, masked_ids =  batch_gen_bert_task(path_batches, index, setting.batch_size)
    #     print(np.array(input_ids).shape)
    #     print(np.array(masked_positions).shape)
    #     print(np.array(masked_ids).shape)
    #     print(masked_ids[0])

    # path_batches = gen_batch_user_item_path(setting, setting.user_item_train_path)
    # path_valid_batches = gen_batch_user_item_path(setting, setting.user_item_valid_path)

    # valid_user_set, valid_item_set = set(), set()
    # with open('./Data/movielens_dataset/user_task/user_task_valid_oracle_rating.csv', 'r') as f:
    #     line = f.readline()
    #     while line!="" and line!=None:
    #         arr = line.strip().split(',')
    #         user, item = int(arr[0]), int(arr[1])
    #         valid_user_set.add(user)
    #         line = f.readline()
    #
    # with open('./Data/movielens_dataset/item_task/item_task_valid_oracle_rating.csv', 'r') as f:
    #     line = f.readline()
    #     while line!="" and line!=None:
    #         arr = line.strip().split(',')
    #         user, item = int(arr[0]), int(arr[1]) + args.num_users
    #         valid_item_set.add(item)
    #         line = f.readline()
    # print(len(valid_user_set))
    # print(len(valid_item_set))
    # print(len(valid_user_set) + len(valid_item_set)) # 2806  选2500
    # valid_set = set.union(valid_item_set, valid_user_set)
    # print(valid_set)
    # print(len(valid_set))
    # valid_list = list(valid_set)
    #
    # user_item_path_valid_list = np.random.choice(valid_list, 2500, replace=False)
    # print(len(user_item_path_valid_list))
    # with open(args.user_item_path, 'r') as f:
    #     with open(args.user_item_train_path, 'w') as writer_train:
    #         with open(args.user_item_valid_path, 'w') as writer_valid:
    #             line = f.readline()
    #             while line!="" and line!=None:
    #                 arr = line.strip().split(',')
    #                 if int(arr[int(arr[-1])]) in user_item_path_valid_list:
    #                     for index in range(len(arr)):
    #                         if index!= len(arr)-1:
    #                             writer_valid.write(str(arr[index])+',')
    #                         else:
    #                             writer_valid.write(str(arr[index]) + '\n')
    #                 else:
    #                     for index in range(len(arr)):
    #                         if index!= len(arr) -1:
    #                             writer_train.write(str(arr[index])+',')
    #                         else:
    #                             writer_train.write(str(arr[index])+'\n')
    #                 line = f.readline()

    replace_delete_meta_path()