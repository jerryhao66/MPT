import tensorflow as tf
import logging
import os
import numpy as np
from utility.Config import Model_Config
import gendata as data
from time import time
import tqdm

setting = Model_Config()



def training_user_task(model, sess):
    print('training meta user task...')
    best_loss = 0
    saver = tf.train.Saver()
    ts_u = data.generate_user_dict_train_meta(setting.oracle_training_file_user_task)

    vs_u = data.generate_user_dict_valid_meta(setting.oracle_valid_file_user_task)
    train_batches = data.generate_meta_train_user_set(ts_u)
    valid_batches = data.generate_meta_valid_user_set(vs_u)

    num_batch = int(train_batches[3]) // setting.batch_size
    batch_index = range(num_batch)
    valid_num_batch = int(valid_batches[3]) // setting.batch_size
    valid_batch_index = range(valid_num_batch)

    for epoch_count in range(setting.user_epoch):
        train_begin = time()
        training_batch_user_task(batch_index, model, sess, train_batches, True)
        train_time = time() - train_begin
        if epoch_count % setting.verbose == 0:
            loss_begin = time()
            train_loss = training_loss_user_task(batch_index, model, sess, train_batches, True)
            loss_time = time() - loss_begin
            eval_begin = time()
            cosine = evaluate_user_task(valid_batch_index, model, sess, valid_batches, True)
            eval_time = time() - eval_begin
            print(
                'epoch %d, train time is %.4f, loss time is %.4f, eval_time is %.4f, train_loss is %.4f, test cosine value is %.4f' % (
                    epoch_count, train_time, loss_time, eval_time, train_loss, cosine))

            if cosine < best_loss:
                best_loss = cosine
                saver.save(sess, setting.checkpoint_path_user_task, global_step=epoch_count)

        ts_u = data.generate_user_dict_train_meta(setting.oracle_training_file_user_task)

        vs_u = data.generate_user_dict_valid_meta(setting.oracle_valid_file_user_task)
        train_batches = data.generate_meta_train_user_set(ts_u)
        valid_batches = data.generate_meta_valid_user_set(vs_u)


def evaluate_user_task(valid_batch_index, model, sess, valid_data, is_training):
    '''
    the train_batch_size not necessarily equal to the test_batch_size
    '''
    evaluate_loss = 0.0
    for index in tqdm.tqdm(valid_batch_index):
        user_id, support_item, target_user = data.batch_gen_user_task(valid_data, index, setting.batch_size)

        feed_dict = {model.support_item: support_item, model.target_user: target_user,
                     model.training_phrase_user_task: is_training}
        evaluate_loss += sess.run(model.loss_user_task, feed_dict)


        batch_predict_ebd, batch_target_ebd = sess.run([model.final_support_encode_user_task, model.target_user], feed_dict)
        

    return evaluate_loss / len(valid_batch_index)


def training_batch_user_task(batch_index, model, sess, train_data, is_training):
    for index in batch_index:
        _, support_item, target_user = data.batch_gen_user_task(train_data, index, setting.batch_size)
        feed_dict = {model.support_item: support_item,
                     model.target_user: target_user,
                     model.training_phrase_user_task: is_training}
        sess.run([model.loss_user_task, model.optimizer_user_task], feed_dict)


def training_loss_user_task(batch_index, model, sess, train_data, is_training):
    train_loss = 0.0
    num_batch = int(train_data[3] / setting.batch_size)
    for index in batch_index:
        _, support_item, target_user = data.batch_gen_user_task(train_data, index, setting.batch_size)
        feed_dict = {model.support_item: support_item,
                     model.target_user: target_user,
                     model.training_phrase_user_task: is_training}
        train_loss += sess.run(model.loss_user_task, feed_dict)

    return train_loss / num_batch

def training_batch_user_reconstruct_task(model, sess, config):
    best_loss, best_perason = 0.0, 0.0
    saver = tf.train.Saver(max_to_keep=3)
    train_u, test_u, num_batch, test_num_batch = load_user_batch(config)
    print(num_batch, test_num_batch)
    batch_index = range(num_batch)
    test_batch_index = range(test_num_batch)

    for epoch in range(config.user_reconstruct_epoch):
        train_loss = 0.0
        for index in tqdm.tqdm(batch_index):
            batch_user = train_u[index]
            # print(np.array(batch_user).shape)

            feed_dict = {model.users: batch_user}
            sess.run([model.optimizer_u_reconstruct, model.batch_loss_u_reconstruct], feed_dict)

            # calculate loss
            train_loss += sess.run(model.batch_loss_u_reconstruct, feed_dict)
        train_loss = train_loss / num_batch

        '''
        evaluate
        '''
        test_loss = 0.0
        for index in test_batch_index:
            test_batch_user = test_u[index]
            # print(np.array(test_batch_user).shape)
            feed_dict = {model.users: test_batch_user}
            test_loss += sess.run(model.batch_loss_u_reconstruct, feed_dict)
            batch_predict_ebd, batch_target_ebd = sess.run([model.batch_predict_u_ebd, model.batch_target_u_ebd],
                                                           feed_dict)


        test_loss = test_loss / test_num_batch

        if test_loss < best_loss:
            best_loss = test_loss
            saver.save(sess, config.checkpoint_path_downstream, global_step=epoch)

        print('[Epoch %d], train_loss is %.4f, test loss is %.4f' % (
            epoch, train_loss, test_loss))


def training_batch_item_reconstruct_task(model, sess, config):
    best_loss = 0.0
    saver = tf.train.Saver(max_to_keep=3)
    train_i, test_i, num_batch, test_num_batch = load_item_batch(config)
    batch_index = range(num_batch)
    test_batch_index = range(test_num_batch)

    for epoch in range(config.item_reconstruct_epoch):
        train_loss = 0.0
        for index in tqdm.tqdm(batch_index):
            batch_item = train_i[index]
            # print(np.array(batch_item).shape)

            feed_dict = {model.pos_items: batch_item}
            sess.run([model.optimizer_i_reconstruct, model.batch_loss_i_reconstruct], feed_dict)

            # calculate loss
            train_loss += sess.run(model.batch_loss_i_reconstruct, feed_dict)
        train_loss = train_loss / num_batch

        '''
        evaluate
        '''
        test_loss = 0.0
        for index in test_batch_index:
            test_batch_item = test_i[index]
            # print(np.array(test_batch_item).shape)
            feed_dict = {model.pos_items: test_batch_item}
            test_loss += sess.run(model.batch_loss_i_reconstruct, feed_dict)

            batch_predict_ebd, batch_target_ebd = sess.run([model.batch_predict_i_ebd, model.batch_target_i_ebd],
                                                           feed_dict)
 

        test_loss = test_loss / test_num_batch

        if test_loss < best_loss:
            best_loss = test_loss
            saver.save(sess, config.checkpoint_path_downstream, global_step=epoch)

        print('[Epoch %d], train_loss is %.4f, test loss is %.4f' % (
            epoch, train_loss, test_loss))

def load_user_batch(Config):
    train_u, test_u = [], []
    with open(Config.data_path + Config.dataset + '/train.txt', 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(' ')
            user, items = arr[0], arr[1:]
            if len(items) >= Config.few_shot_number ** 3:
                train_u.append(user)
            else:
                test_u.append(user)
            line = f.readline()
    train_user_list, test_user_list = [], []
    num_batch = len(train_u) // Config.batch_size_reconstruct
    for batch_index in range(num_batch + 1):
        begin_index = batch_index * Config.batch_size_reconstruct
        end_index = min(batch_index * Config.batch_size_reconstruct + Config.batch_size_reconstruct, len(train_u))
        batch_users = train_u[begin_index:end_index]
        train_user_list.append(batch_users)

    test_num_batch = len(test_u) // Config.batch_size_reconstruct
    for batch_index in range(test_num_batch + 1):
        begin_index = batch_index * Config.batch_size_reconstruct
        end_index = min(batch_index * Config.batch_size_reconstruct + Config.batch_size_reconstruct, len(test_u))
        batch_users = test_u[begin_index:end_index]
        test_user_list.append(batch_users)
    return train_user_list, test_user_list, num_batch + 1, test_num_batch + 1


def load_item_batch(Config):
    item_dict = {}
    with open(Config.data_path + Config.dataset + '/train.txt', 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(' ')
            user, items = arr[0], arr[1:]
            for item in items:
                if item not in item_dict:
                    item_dict[item] = []
                    item_dict[item].append(user)
                else:
                    item_dict[item].append(user)
            line = f.readline()

    train_i, test_i = [], []
    for item in item_dict.keys():
        if len(item_dict[item]) >= Config.few_shot_number ** 3:
            train_i.append(item)
        else:
            test_i.append(item)
    train_item_list, test_item_list = [], []
    num_batch = len(train_i) // Config.batch_size_reconstruct
    for batch_index in range(num_batch + 1):
        begin_index = batch_index * Config.batch_size_reconstruct
        end_index = min(batch_index * Config.batch_size_reconstruct + Config.batch_size_reconstruct, len(train_i))
        batch_items = train_i[begin_index:end_index]
        train_item_list.append(batch_items)

    test_num_batch = len(test_i) // Config.batch_size_reconstruct
    for batch_index in range(test_num_batch + 1):
        begin_index = batch_index * Config.batch_size_reconstruct
        end_index = min(batch_index * Config.batch_size_reconstruct + Config.batch_size_reconstruct, len(test_i))
        batch_items = test_i[begin_index:end_index]
        test_item_list.append(batch_items)

    return train_item_list, test_item_list, num_batch + 1, test_num_batch + 1



'''
item_task
'''


def training_batch_item_task(batch_index, model, sess, train_data, is_training):
    for index in batch_index:
        _, support_user, target_item = data.batch_gen_item_task(train_data, index, setting.batch_size)
        feed_dict = {model.support_user: support_user,
                     model.target_item: target_item,
                     model.training_phrase_item_task: is_training}
        sess.run([model.loss_item_task, model.optimizer_item_task], feed_dict)


def training_loss_item_task(batch_index, model, sess, train_data, is_training):
    train_loss = 0.0
    num_batch = int(train_data[3] / setting.batch_size)
    for index in batch_index:
        _, support_user, target_item = data.batch_gen_item_task(train_data, index, setting.batch_size)
        feed_dict = {model.support_user: support_user,
                     model.target_item: target_item,
                     model.training_phrase_item_task: is_training}
        train_loss += sess.run(model.loss_item_task, feed_dict)

    return train_loss / num_batch


def training_item_task(model, sess):
    print('training item task...')
    best_loss = 0
    saver = tf.train.Saver(max_to_keep=3)
    ts_i = data.generate_item_dict_train_meta(setting.oracle_training_file_item_task)
    vs_i = data.generate_item_dict_valid_meta(setting.oracle_valid_file_item_task)

    train_batches = data.generate_meta_train_item_set(ts_i)
    valid_batches = data.generate_meta_valid_item_set(vs_i)

    num_batch = int(train_batches[3]) // setting.batch_size
    batch_index = range(num_batch)
    valid_num_batch = int(valid_batches[3]) // setting.batch_size
    valid_batch_index = range(valid_num_batch)

    for epoch_count in range(setting.item_epoch):
        train_begin = time()
        training_batch_item_task(batch_index, model, sess, train_batches, True)
        train_time = time() - train_begin
        if epoch_count % setting.verbose == 0:
            loss_begin = time()
            train_loss = training_loss_item_task(batch_index, model, sess, train_batches, True)
            loss_time = time() - loss_begin
            eval_begin = time()
            cosine = evaluate_item_task(valid_batch_index, model, sess, valid_batches, True)
            eval_time = time() - eval_begin
            print(
                'epoch %d, train time is %.4f, loss time is %.4f, eval_time is %.4f, train_loss is %.4f, test cosine value is %.4f' % (
                    epoch_count, train_time, loss_time, eval_time, train_loss, cosine))
            if cosine < best_loss:
                best_loss = cosine
                saver.save(sess, setting.checkpoint_path_item_task, global_step=epoch_count)

        ts_i = data.generate_item_dict_train_meta(setting.oracle_training_file_item_task)
        vs_i = data.generate_item_dict_valid_meta(setting.oracle_valid_file_item_task)

        train_batches = data.generate_meta_train_item_set(ts_i)
        valid_batches = data.generate_meta_valid_item_set(vs_i)


def evaluate_item_task(valid_batch_index, model, sess, valid_data, is_training):
    '''
    the train_batch_size not necessarily equal to the test_batch_size
    '''
    evaluate_loss = 0.0
    for index in tqdm.tqdm(valid_batch_index):
        item_id, support_user, target_item = data.batch_gen_item_task(valid_data, index, setting.batch_size)

        feed_dict = {model.support_user: support_user, model.target_item: target_item,
                     model.training_phrase_item_task: is_training}
        evaluate_loss += sess.run(model.loss_item_task, feed_dict)

        batch_predict_ebd, batch_target_ebd = sess.run([model.final_support_encode_item_task, model.target_item], feed_dict)

    return evaluate_loss / len(valid_batch_index)


def generate_ebd_batch_user_task(batch_index, model, sess, train_data, is_training):
    meta_user_ebd = np.random.random((setting.num_users, setting.embedding_size))
    for index in batch_index:
        train_user_id, support_item, target_user = data.batch_gen_user_task(train_data, index, setting.batch_size)
        feed_dict = {model.support_item: support_item,
                     model.training_phrase_user_task: is_training}

        gen_user_ebd = np.array(sess.run(model.final_support_encode_user_task, feed_dict))
        meta_user_ebd[train_user_id, :] = gen_user_ebd

    return meta_user_ebd


def generate_ebd_batch_item_task(batch_index, model, sess, train_data, is_training):
    meta_item_ebd = np.random.random((setting.num_items, setting.embedding_size))
    for index in batch_index:
        train_item_id, support_user, target_item = data.batch_gen_item_task(train_data, index, setting.batch_size)
        feed_dict = {model.support_user: support_user,
                     model.training_phrase_item_task: is_training}

        gen_item_ebd = np.array(sess.run(model.final_support_encode_item_task, feed_dict))
        meta_item_ebd[train_item_id,:] = gen_item_ebd

    return meta_item_ebd



def generate_user_task_data(model, sess):
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.get_checkpoint_state(
        os.path.dirname(setting.checkpoint_path_user_task + 'checkpoint')).model_checkpoint_path)

    
    all_u = data.generate_user_dict_train_meta(setting.training_plus_test_support_file)
    print(len(all_u.keys()))
    all_batches = data.generate_meta_train_user_set(all_u)
    num_batch = int(all_batches[3]) // setting.batch_size + 1
    batch_index = range(num_batch)
    gen_user_ebd = generate_ebd_batch_user_task(batch_index, model, sess, all_batches, False)
    print(gen_user_ebd.shape)
    np.save(setting.meta_user_ebd_path, gen_user_ebd)


def generate_item_task_data(model, sess):
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.get_checkpoint_state(
        os.path.dirname(setting.checkpoint_path_item_task + 'checkpoint')).model_checkpoint_path)

    
    all_i = data.generate_item_dict_train_meta(setting.training_plus_test_support_file)
    print(len(all_i.keys()))
    all_batches = data.generate_meta_train_item_set(all_i)
    num_batch = int(all_batches[3]) // setting.batch_size + 1

    batch_index = range(num_batch)
    gen_item_ebd = generate_ebd_batch_item_task(batch_index, model, sess, all_batches, False)
    print(gen_item_ebd.shape)
    np.save(setting.meta_item_ebd_path, gen_item_ebd)




def generate_transformer_data(model, sess):
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.get_checkpoint_state(
        os.path.dirname(setting.checkpoint_path_bert_task + 'checkpoint')).model_checkpoint_path)

    path_batches = data.gen_batch_user_item_path(setting, setting.user_item_path)

    num_batch = len(path_batches[0]) // setting.batch_size_tranformer  # 整除
    batch_index = range(num_batch)

    gen_user_ebd, gen_item_ebd = generate_ebd_batch_bert_task(batch_index, model, sess, path_batches)
    print(gen_user_ebd.shape)
    print(gen_item_ebd.shape)
    np.save(setting.bert_user_ebd_path, gen_user_ebd)
    np.save(setting.bert_item_ebd_path, gen_item_ebd)


def generate_ebd_batch_bert_task(batch_index, model, sess, train_data):
    bert_ebd = np.random.random((setting.vocab_size, setting.embedding_size))
    for index in batch_index:
        input_ids, masked_positions, masked_ids = data.batch_gen_bert_task(train_data, index, setting.batch_size_tranformer)
        feed_dict = {model.input_ids: np.array(input_ids), model.masked_ids: np.array(masked_ids),
                model.gen_ebd_mask_position: masked_positions[0] }
        gen_ebd = np.array(sess.run([model.mask_generate_ebd], feed_dict))
        bert_ebd[masked_ids, :] = gen_ebd

    ebd = np.reshape(bert_ebd, (-1, setting.vocab_size, setting.embedding_size))
    mean_ebd = np.mean(ebd, axis=0)
    print(mean_ebd.shape)
    bert_user_ebd, bert_item_ebd = mean_ebd[0:setting.num_users,:], mean_ebd[setting.num_users:setting.num_users+setting.num_items,:]
    print(bert_user_ebd.shape)
    print(bert_item_ebd.shape)

    return bert_user_ebd, bert_item_ebd

def generate_contrastive_transformer_data(model, sess):
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.get_checkpoint_state(
        os.path.dirname(setting.checkpoint_path_bert_contrastive_task + 'checkpoint')).model_checkpoint_path)

    batches1 = data.gen_batch_user_item_path(setting, setting.user_item_delete_path)
    batches2 = data.gen_batch_user_item_path(setting, setting.user_item_replace_path)

    num_batch1 = len(batches1[0]) // setting.batch_size_tranformer  # 整除
    batch_index1 = range(num_batch1)
    num_batch2 = len(batches2[0]) // setting.batch_size_tranformer  # 整除
    batch_index2 = range(num_batch2)

    gen_user_ebd1, gen_item_ebd1 = generate_ebd_batch_bert_task(batch_index1, model, sess, batches1)
    gen_user_ebd2, gen_item_ebd2 = generate_ebd_batch_bert_task(batch_index2, model, sess, batches2)

    gen_user_ebd = (gen_user_ebd1 + gen_user_ebd2) / 2
    gen_item_ebd = (gen_item_ebd1 + gen_item_ebd2) / 2
    print(gen_user_ebd.shape)
    print(gen_item_ebd.shape)
    np.save(setting.bert_user_ebd_path, gen_user_ebd)
    np.save(setting.bert_item_ebd_path, gen_item_ebd)


############################################################################################################
################## embedding contrastive with GNN ###########################################################
######################## user task #######################################################################
def training_batch_user_contrastive_gnn_task(model, sess):
    best_loss = 0.0
    saver = tf.train.Saver()
    train_batches = data.generate_user_dict_train(setting.oracle_training_file_user_task)
    valid_batches = data.generate_user_dict_valid(setting.oracle_valid_file_user_task)

    num_batch = int(train_batches[1]) // setting.batch_size
    batch_index = range(num_batch)
    valid_num_batch = int(valid_batches[1]) // setting.batch_size
    valid_batch_index = range(valid_num_batch)

    for epoch_count in range(setting.user_epoch):
        train_begin = time()
        training_batch_contrastive_gnn_user_task(batch_index, model, sess, train_batches)
        train_time = time() - train_begin
        if epoch_count % setting.verbose == 0:
            loss_begin = time()
            train_loss = training_loss_contrastive_gnn_user_task(batch_index, model, sess, train_batches)
            loss_time = time() - loss_begin
            eval_begin = time()
            evaluate_loss = evaluate_user_contrastive_gnn_task(valid_batch_index, model, sess, valid_batches)
            eval_time = time() - eval_begin
            print(
                'epoch %d, train time is %.4f, loss time is %.4f, eval_time is %.4f, train_loss is %.4f, test loss is %.4f' % (
                    epoch_count, train_time, loss_time, eval_time, train_loss, evaluate_loss))

            if evaluate_loss < best_loss:
                best_loss = evaluate_loss
                saver.save(sess, setting.checkpoint_path_user_task, global_step=epoch_count)

        train_batches = data.generate_user_dict_train(setting.oracle_training_file_user_task)
        valid_batches = data.generate_user_dict_valid(setting.oracle_valid_file_user_task)




def training_batch_contrastive_gnn_user_task(batch_index, model, sess, train_data):
    for index in batch_index:
        user_id = data.batch_gen_contrastive_gnn_user_task(train_data, index, setting.batch_size)
        feed_dict = {model.cl_pos_user: user_id}
        sess.run([model.optimizer_user_contrastive_task, model.user_contrastive_loss], feed_dict)


def training_loss_contrastive_gnn_user_task(batch_index, model, sess, train_data):
    train_loss = 0.0
    for index in batch_index:
        user_id = data.batch_gen_contrastive_gnn_user_task(train_data, index, setting.batch_size)
        feed_dict = {model.cl_pos_user: user_id}
        train_loss += sess.run(model.user_contrastive_loss, feed_dict)

    return train_loss / len(batch_index)


def evaluate_user_contrastive_gnn_task(batch_index, model, sess, valid_data):
    valid_loss = 0.0
    for index in batch_index:
        user_id = data.batch_gen_contrastive_gnn_user_task(valid_data, index, setting.batch_size)
        feed_dict = {model.cl_pos_user: user_id}
        valid_loss += sess.run(model.user_contrastive_loss, feed_dict)

    return valid_loss / len(batch_index)


###################### item task ###########################
def training_batch_item_contrastive_gnn_task(model, sess):
    best_loss = 0.0
    saver = tf.train.Saver()

    train_batches = data.generate_item_dict_train(setting.oracle_training_file_item_task)
    valid_batches = data.generate_item_dict_valid(setting.oracle_valid_file_item_task)


    num_batch = int(train_batches[1]) // setting.batch_size
    batch_index = range(num_batch)
    valid_num_batch = int(valid_batches[1]) // setting.batch_size
    valid_batch_index = range(valid_num_batch)

    for epoch_count in range(setting.item_epoch):
        train_begin = time()
        training_batch_contrastive_gnn_item_task(batch_index, model, sess, train_batches)
        train_time = time() - train_begin
        if epoch_count % setting.verbose == 0:
            loss_begin = time()
            train_loss = training_loss_contrastive_gnn_item_task(batch_index, model, sess, train_batches)
            loss_time = time() - loss_begin
            eval_begin = time()
            evaluate_loss = evaluate_item_contrastive_gnn_task(valid_batch_index, model, sess, valid_batches)
            eval_time = time() - eval_begin
            print(
                'epoch %d, train time is %.4f, loss time is %.4f, eval_time is %.4f, train_loss is %.4f, test loss is %.4f' % (
                    epoch_count, train_time, loss_time, eval_time, train_loss, evaluate_loss))

            if evaluate_loss < best_loss:
                best_loss = evaluate_loss
                saver.save(sess, setting.checkpoint_path_user_task, global_step=epoch_count)

        train_batches = data.generate_item_dict_train(setting.oracle_training_file_item_task)
        valid_batches = data.generate_item_dict_valid(setting.oracle_valid_file_item_task)



def training_batch_contrastive_gnn_item_task(batch_index, model, sess, train_data):
    for index in batch_index:
        item_id = data.batch_gen_contrastive_gnn_item_task(train_data, index, setting.batch_size)
        feed_dict = {model.cl_pos_item: item_id}
        sess.run([model.optimizer_item_contrastive_task, model.item_contrastive_loss], feed_dict)


def training_loss_contrastive_gnn_item_task(batch_index, model, sess, train_data):
    train_loss = 0.0
    for index in batch_index:
        item_id = data.batch_gen_contrastive_gnn_item_task(train_data, index, setting.batch_size)
        feed_dict = {model.cl_pos_item: item_id}
        train_loss += sess.run(model.item_contrastive_loss, feed_dict)

    return train_loss / len(batch_index)


def evaluate_item_contrastive_gnn_task(batch_index, model, sess, valid_data):
    valid_loss = 0.0
    for index in batch_index:
        item_id = data.batch_gen_contrastive_gnn_item_task(valid_data, index, setting.batch_size)
        feed_dict = {model.cl_pos_item: item_id}
        valid_loss += sess.run(model.item_contrastive_loss, feed_dict)

    return valid_loss / len(batch_index)


#########################################################################################################
##################### end embedding contrastive gnn #####################################################



################### reconstruction with Transformer #######################################
def training_reconstruction_transformer(GeneralConv, sess):
    best_loss = 0.0
    saver = tf.train.Saver()

    path_batches = data.gen_batch_user_item_path(setting, setting.user_item_train_path)
    path_valid_batches = data.gen_batch_user_item_path(setting, setting.user_item_valid_path)

    num_batch = len(path_batches[0]) // setting.batch_size_tranformer  # 整除
    batch_index = range(num_batch)

    num_valid_batch = len(path_valid_batches[0]) // setting.batch_size_tranformer
    valid_batch_index = range(num_valid_batch)

    for epoch_count in range(setting.user_epoch):
        train_begin = time()
        training_batch_reconstruction_tr_task(batch_index, GeneralConv, sess, path_batches)
        train_time = time() - train_begin
        loss_begin = time()
        train_loss = training_loss_reconstruction_tr_task(batch_index, GeneralConv, sess, path_batches)
        loss_time = time() - loss_begin
        eval_begin = time()
        evaluate_loss = evaluate_reconstruction_tr_task(valid_batch_index, GeneralConv, sess, path_valid_batches)
        eval_time = time() - eval_begin
        print(
            'epoch %d, train time is %.4f, loss time is %.4f, eval_time is %.4f, train_loss is %.4f, test loss is %.4f' % (
                epoch_count, train_time, loss_time, eval_time, train_loss, evaluate_loss))
        if evaluate_loss < best_loss:
            best_loss = evaluate_loss
            saver.save(sess, setting.checkpoint_path_bert_task, global_step=epoch_count)


def training_batch_reconstruction_tr_task(batch_index, model, sess, batches):
    for index in batch_index:
        input_ids, masked_positions, masked_ids = data.batch_gen_bert_task(batches, index, setting.batch_size_tranformer)
        feed_dict = {model.input_ids: np.array(input_ids), model.masked_ids: np.array(masked_ids),
                model.gen_ebd_mask_position: masked_positions[0] }
        sess.run([model.optimizer_reconstruct], feed_dict)

def training_loss_reconstruction_tr_task(batch_index, model, sess, batches):
    train_loss = 0.0
    for index in batch_index:
        input_ids, masked_positions, masked_ids = data.batch_gen_bert_task(batches, index, setting.batch_size_tranformer)
        feed_dict = {model.input_ids: np.array(input_ids), model.masked_ids: np.array(masked_ids),
                model.gen_ebd_mask_position: masked_positions[0] }
        train_loss += sess.run(model.bert_batch_loss_reconstruct, feed_dict)

    return train_loss / len(batch_index)

def evaluate_reconstruction_tr_task(batch_index, model, sess, batches):
    valid_loss = 0.0
    for index in batch_index:
        input_ids, masked_positions, masked_ids = data.batch_gen_bert_task(batches, index,
                                                                           setting.batch_size_tranformer)
        feed_dict = {model.input_ids: np.array(input_ids), model.masked_ids: np.array(masked_ids),
                     model.gen_ebd_mask_position: masked_positions[0]}
        valid_loss += sess.run(model.bert_batch_loss_reconstruct, feed_dict)

    return valid_loss / len(batch_index)

################### Contrastive with Transformer #######################################
def training_contrastive_transformer(GeneralConv, sess):
    best_loss = 10.0
    saver = tf.train.Saver()

    path_batches_del = data.gen_batch_user_item_path(setting, setting.user_item_delete_path)
    path_batches_rep = data.gen_batch_user_item_path(setting, setting.user_item_replace_path)

    path_valid_batches = data.gen_batch_user_item_path(setting, setting.user_item_valid_path)

    num_batch = len(path_batches_del[0]) // setting.batch_size_tranformer  # 整除
    batch_index = range(num_batch)

    num_valid_batch = len(path_valid_batches[0]) // setting.batch_size_tranformer
    valid_batch_index = range(num_valid_batch)

    for epoch_count in range(setting.user_epoch):
        train_begin = time()
        training_batch_contrastive_tr_task(batch_index, batch_index, path_batches_del, path_batches_rep,  GeneralConv, sess)
        train_time = time() - train_begin
        loss_begin = time()
        train_loss = training_loss_contrastive_tr_task(batch_index, batch_index, path_batches_del, path_batches_rep,  GeneralConv, sess)
        loss_time = time() - loss_begin
        eval_begin = time()
        evaluate_loss = evaluate_contrastive_tr_task(valid_batch_index, valid_batch_index, path_batches_del, path_batches_rep,  GeneralConv, sess)
        eval_time = time() - eval_begin
        print(
            'epoch %d, train time is %.4f, loss time is %.4f, eval_time is %.4f, train_loss is %.4f, test loss is %.4f' % (
                epoch_count, train_time, loss_time, eval_time, train_loss, evaluate_loss))
        if evaluate_loss < best_loss:
            best_loss = evaluate_loss
            saver.save(sess, setting.checkpoint_path_bert_contrastive_task, global_step=epoch_count)


def training_batch_contrastive_tr_task(batch_index1,batch_index2, batches1, batches2, model, sess):
    assert batch_index1 == batch_index2
    for index in batch_index1:
        input_ids1, masked_positions1, masked_ids1 = data.batch_gen_bert_task(batches1, index, setting.batch_size_tranformer)
        input_ids2, masked_positions2, masked_ids2 = data.batch_gen_bert_task(batches2, index, setting.batch_size_tranformer)
        feed_dict = {model.cl_input_ids1: np.array(input_ids1), model.cl_masked_ids1: np.array(masked_ids1),
                model.cl_gen_ebd_mask_position1: masked_positions1[0],
                     model.cl_input_ids2: np.array(input_ids2), model.cl_masked_ids2: np.array(masked_ids2),
                     model.cl_gen_ebd_mask_position2: masked_positions2[0]}
        sess.run([model.optimizer_contrastive_bert_task], feed_dict)

def training_loss_contrastive_tr_task(batch_index1,batch_index2, batches1, batches2, model, sess):
    train_loss = 0.0
    assert batch_index1 == batch_index2
    for index in batch_index1:
        input_ids1, masked_positions1, masked_ids1 = data.batch_gen_bert_task(batches1, index,
                                                                              setting.batch_size_tranformer)
        input_ids2, masked_positions2, masked_ids2 = data.batch_gen_bert_task(batches2, index,
                                                                              setting.batch_size_tranformer)
        feed_dict = {model.cl_input_ids1: np.array(input_ids1), model.cl_masked_ids1: np.array(masked_ids1),
                model.cl_gen_ebd_mask_position1: masked_positions1[0],
                     model.cl_input_ids2: np.array(input_ids2), model.cl_masked_ids2: np.array(masked_ids2),
                     model.cl_gen_ebd_mask_position2: masked_positions2[0]}
        train_loss += sess.run(model.contrastive_transformer_loss, feed_dict)

    return train_loss / len(batch_index1)

def evaluate_contrastive_tr_task(batch_index1, batch_index2, batches1, batches2, model, sess):
    valid_loss = 0.0
    assert batch_index1 == batch_index2
    for index in batch_index1:
        input_ids1, masked_positions1, masked_ids1 = data.batch_gen_bert_task(batches1, index,
                                                                              setting.batch_size_tranformer)
        input_ids2, masked_positions2, masked_ids2 = data.batch_gen_bert_task(batches2, index,
                                                                              setting.batch_size_tranformer)
        feed_dict = {model.cl_input_ids1: np.array(input_ids1), model.cl_masked_ids1: np.array(masked_ids1),
                model.cl_gen_ebd_mask_position1: masked_positions1[0],
                     model.cl_input_ids2: np.array(input_ids2), model.cl_masked_ids2: np.array(masked_ids2),
                     model.cl_gen_ebd_mask_position2: masked_positions2[0]}
        valid_loss += sess.run(model.contrastive_transformer_loss, feed_dict)

    return valid_loss / len(batch_index1)
