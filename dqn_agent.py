
import numpy as np
import Environment
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import random
import math
import os
import cv2
import copy

TRAIN_FREQUENCY = 4

MAX_MEMORY_SIZE = 10000
# hyperparameters
height = 50
width = 100
n_obs = 100 * 50  # dimensionality of observations
h = 200  # number of hidden layer neurons
n_actions = 4  # number of available actions
learning_rate = 0.00025
gamma = .99  # discount factor for reward

batch_size = 32

learn_start = 1000
learning_rate_minimum = 0.00025,
learning_rate_decay_step = 5 * 10000,
learning_rate_decay = 0.96,
target_q_update_step = 1000
decay = 0.99  # decay rate for RMSProp gradients
save_path = 'models_Attemp32/Attempt32'
INITIAL_EPSILON = 1

# gamespace
display = True
training = False

game=Environment.GameV1(display)
game.populateGameArray()



def clipped_error(x):
  # Huber loss
  try:
    return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
  except:
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

def conv2d(x,
           output_dim,
           kernel_size,
           stride,
           initializer=tf.contrib.layers.xavier_initializer(),
           activation_fn=tf.nn.relu,
           data_format='NHWC',
           padding='VALID',
           name='conv2d'):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]
        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
        conv = tf.nn.conv2d(x, w, stride, padding, data_format=data_format)

        b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, b, data_format)

        if activation_fn != None:
            out = activation_fn(out)
        return out, w, b

def linear(input_, output_size, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear'):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(name):
    w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
        tf.random_normal_initializer(stddev=stddev))
    b = tf.get_variable('bias', [output_size],
        initializer=tf.constant_initializer(bias_start))

    out = tf.nn.bias_add(tf.matmul(input_, w), b)

    if activation_fn != None:
        return activation_fn(out), w, b
    else:
        return out, w, b

def setup_training_step_count():
    with tf.variable_scope('step'):
        step_op = tf.Variable(0, trainable=False, name='step')
        step_input = tf.placeholder('int32', None, name='step_input')
        step_assign_op = step_op.assign(step_input)
    return step_op, step_input, step_assign_op

class dqn_Model():
    def __init__(self, network_scope_name, sess):
        self.network_scope_name = network_scope_name
        self.sess = sess
        pass

    def forward_graph(self):
        self.w = {}

        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        with tf.variable_scope(self.network_scope_name):
            self.s_t = tf.placeholder(dtype=tf.float32,
                                      shape=[None, height, width, 1], name='s_t')
            self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t,
                                                             32, [8, 8], [4, 4], initializer, activation_fn,
                                                             "NHWC", name='l1')
            self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
                                                             64, [4, 4], [2, 2], initializer, activation_fn,
                                                             "NHWC", name='l2')
            self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
                                                             64, [3, 3], [1, 1], initializer, activation_fn,
                                                             "NHWC", name='l3')

            self.l3_flat = flatten(self.l3)
            self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = \
                linear(self.l3_flat, 512, activation_fn=activation_fn, name='value_hid')

            self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = \
                linear(self.l3_flat, 512, activation_fn=activation_fn, name='adv_hid')

            self.value, self.w['val_w_out'], self.w['val_w_b'] = \
                linear(self.value_hid, 1, name='value_out')

            self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
                linear(self.adv_hid, n_actions, name='adv_out')

            # Average Dueling
            self.q = self.value + (self.advantage -
                                   tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))

            self.a_action = tf.argmax(self.q, axis=1)

    def select_q_graph(self):
        self.q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
        self.q_with_idx = tf.gather_nd(self.q, self.q_idx)

    def train_graph(self):
        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.target_q_t = tf.placeholder(shape=[None], dtype=tf.float32, name='target_q_t')
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
        self.actions_onehot = tf.one_hot(self.actions, n_actions, dtype=tf.float32, name='action_one_hot')

        self.q_acted = tf.reduce_sum(tf.multiply(self.q, self.actions_onehot), axis=1, name='q_acted')

        self.delta = tf.square(self.target_q_t - self.q_acted)
        #self.global_step = tf.Variable(0, trainable=False)
        # apply huber loss to clipping the error, and derivative
        self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')

        self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
        self.learning_rate_op = tf.maximum(learning_rate_minimum,
                                           tf.train.exponential_decay(
                                               learning_rate,
                                               self.learning_rate_step,
                                               learning_rate_decay_step,
                                               learning_rate_decay,
                                               staircase=True))
        self.optim = tf.train.RMSPropOptimizer(
            learning_rate=learning_rate, momentum=0.95, epsilon=0.01)
        tf_grads = self.optim.compute_gradients(self.loss,  var_list=tf.trainable_variables())
        self.train_op = self.optim.apply_gradients(tf_grads)

        grad_summaries = []
        for g, v in tf_grads:
            if g is not None:
                #grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                #grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        loss_summary = tf.summary.scalar("main_q_loss", self.loss)
        #learning_rate_ = tf.summary.scalar("learning_rate", self.learning_rate_op)

        self.train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])



    def summary_graph(self):
        q_summary = []
        avg_q = tf.reduce_mean(self.q, 0)
        for idx in range(n_actions):
            q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))

        self.q_summary = tf.summary.merge(q_summary, 'q_summary')

        self.win_rate_holder = tf.placeholder("float32", None, name="running_win_rate")
        self.win_rate_op = tf.summary.scalar("win_rate", self.win_rate_holder)

        scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
                               'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game',
                               'training.learning_rate']

        self.summary_placeholders = {}
        self.summary_ops = {}

        for tag in scalar_summary_tags:
            self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
            self.summary_ops[tag] = tf.summary.scalar("%s" % (tag),
                                                      self.summary_placeholders[tag])

        histogram_summary_tags = ['episode.rewards', 'episode.actions']

        for tag in histogram_summary_tags:
            self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
            self.summary_ops[tag] = tf.summary.histogram(tag, self.summary_placeholders[tag])

        train_summary_dir = os.path.join(save_path, "summaries", "train")
        self.writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)


    def update_network(self):
        if self.network_scope_name == "target":
            self.w_input = {}
            self.w_assign_op = {}
            for name in self.w.keys():
                self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
                self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

def update_target_network(target_network, main_network):
    for name in target_network.w.keys():
        target_network.w_assign_op[name].eval({target_network.w_input[name] : main_network.w[name].eval()})


class experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 6])


def discount_rewards(rewardarray):

    gamenumber = 0
    episodeoneend = 0
    for i in range(len(rewardarray)):
        if rewardarray[i] > 0:
            if gamenumber ==0:
                rewardarray[i] = (177-i)/30 * rewardarray[i] *2
                episodeoneend = i
                gamenumber +=1
            else:
                rewardarray[i] = (177 - (i - episodeoneend)) / 30 * rewardarray[i] * 2
                gamenumber += 1
                episodeoneend = i

        elif rewardarray[i] < 0:
            rewardarray[i] = rewardarray[i] * math.pow(3, (170-i)/170)
            gamenumber += 1
            episodeoneend = i
        else:
            rewardarray[i] = rewardarray[i]

    rewardarray.reverse()
    for i in range(len(rewardarray) -1):

        rewardarray[i+1] = rewardarray[i] * gamma
    rewardarray.reverse()
    return rewardarray


def discount_smallrewards(rewardarray):
    rewardarray.reverse()

    for i in range(len(rewardarray) -1):
        if rewardarray[i] != 0:
            rewardarray[i] = rewardarray[i] * 15
    rewardarray.reverse()
    return rewardarray

def displayImage(image):
    __debug_diff__ = False
    if __debug_diff__:
        cv2.namedWindow("debug image")
        cv2.imshow('debug image', image)
        cv2.waitKey(2000)
        cv2.destroyWindow("debug image")


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 5000 (50x100) 1D float vector """
    I = I[::4, ::3]  # downsample by factor of 2
    return I.astype('float32')

def diff(X0, X1):
    #X0[X0==0.5] = 0 # erase the player car in the first image
    X0[X0==0.1] = 0 # erase the background
    X1[X1==0.1] = 0 # erase the background

    diff_image = X1 - 0.75*X0
    if display:
        displayImage(diff_image)
    return diff_image

def count_win_percentage(win_loss):
    count = 0
    for win in win_loss:
        if win == 1:
            count +=1
    return 100 * count / len(win_loss)


def restore_model(sess):
    episode_number = 0

    # try load saved model
    saver = tf.train.Saver(tf.global_variables())
    load_was_success = True  # yes, I'm being optimistic
    try:
        save_dir = '/'.join(save_path.split('/')[:-1])
        ckpt = tf.train.get_checkpoint_state(save_dir)
        load_path = ckpt.model_checkpoint_path
        saver.restore(sess, load_path)

    except:
        print(
            "no saved model to load. starting new session")
        tf.global_variables_initializer().run()
        load_was_success = False
    else:
        print(
            "loaded model: {}".format(load_path))
        #saver = tf.train.Saver(tf.global_variables())
        episode_number = int(load_path.split('-')[-1])

    return saver, episode_number, save_dir



def train(sess):

    # create forward graph
    main_q_net = dqn_Model("main", sess)
    target_q_net = dqn_Model("target", sess)

    # create graphs, although there are two graphs for main and target networks, but they all belong to the same session
    # hence, the model save and restore will save them all and restore them all
    # however, the scope name will differentiate the main network from the target network
    main_q_net.forward_graph()
    main_q_net.train_graph()

    target_q_net.forward_graph()
    target_q_net.select_q_graph()
    target_q_net.update_network()

    step_op, step_input, step_assign_op = setup_training_step_count()

    main_q_net.summary_graph() # create the summary graph last, so ops will be in tensorboard

    # restore the model with previously saved weights

    saver, start_episode_number, save_dir = restore_model(sess)

    step = step_op.eval(session=sess)

    #train_summary_dir = os.path.join(save_dir, "summaries", "train")
    #train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    #epsilon = 1
    epsilon = math.pow(0.5, start_episode_number/3000)


    wait_time = 1
    waited_time = 0

    running_reward = None
    reward_sum = 0
    observation = np.zeros(shape=(200, 300))
    win_loss = []
    # training loop
    replay_buffer = experience_buffer()

    # preprocess the observation, set input to network to be difference image
    s_t = prepro(observation)
    episode_number = start_episode_number
    episodeBuffer = experience_buffer()

    last_saved_win_rate = 0

    sess.graph.finalize()
    while True: # looping over every step. episode consists of many steps till end of a game?
        if waited_time < wait_time:
            action = 0
            waited_time += 1
            observation, reward, smallreward, done = game.runGame(action, False)
        else:
            # perform epsilon greedy for explorationa nd exploitation
            if random.random() < epsilon:
                action = random.randint(0, n_actions-1)
            else:
                feed = {main_q_net.s_t:np.reshape(s_t, [-1, height, width, 1]) }
                action = sess.run(main_q_net.a_action, feed)[0];

            # roll out a step
            observation, reward, smallreward, done = game.runGame(action, False)
            s_t_plus_1 = prepro(observation)

            # save everything about this step into the episode buffer
            episodeBuffer.add(np.reshape(np.array([s_t, action, reward, smallreward, s_t_plus_1, done]), [1, 6]))

            #reset state variable for next step
            s_t = s_t_plus_1


            # parameter update
            if episode_number-start_episode_number > learn_start: # wait till enough in the play buffer
                # sample a batch from the replay buffer
                # ready to train the networks
                if step % TRAIN_FREQUENCY == 0:
                    train_batch = replay_buffer.sample(batch_size)

                    # unpack the samples
                    batch_s_t = np.expand_dims(np.stack(train_batch[:, 0]), -1)
                    batch_action = train_batch[:, 1]
                    batch_reward = np.expand_dims(np.stack(train_batch[:, 2]), -1)
                    batch_smallreward = train_batch[:, 3]
                    batch_s_t_plus_1 = np.expand_dims(np.stack(train_batch[:, 4]), -1)
                    batch_done = train_batch[:, 5]

                    # get a_t_plus_1 from the main net
                    pred_action = main_q_net.a_action.eval({main_q_net.s_t: batch_s_t_plus_1})
                    # get q_t_plus_1 from the target net (not max, a main diff from the vallina dqn)
                    q_t_plus_1_with_pred_action = target_q_net.q_with_idx.eval({target_q_net.s_t: batch_s_t_plus_1,
                                                                                target_q_net.q_idx: \
                                                                                    [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]})
                    # compute q estimates
                    target_q_t = (1.0 - batch_done)*gamma * q_t_plus_1_with_pred_action + batch_smallreward

                    # prepare input for backprop on main network
                    feed = {main_q_net.s_t: batch_s_t,
                            main_q_net.actions: batch_action,
                            main_q_net.target_q_t: target_q_t,
                            main_q_net.learning_rate_step: step
                            }

                    _, q_t, loss, train_summaries, q_summaries = sess.run([main_q_net.train_op, main_q_net.q, main_q_net.loss,\
                                                              main_q_net.train_summary_op, main_q_net.q_summary],\
                                                             feed_dict=feed)

                    main_q_net.writer.add_summary(train_summaries, step)
                    main_q_net.writer.add_summary(q_summaries, step)

                # update target network, and save the model to hd
                if step % target_q_update_step == target_q_update_step -1:
                    # update the target network
                    update_target_network(target_q_net,main_q_net)

                    # persist models only when win_rate is better
                    if win_rate > last_saved_win_rate:
                        step_assign_op.eval({step_input: step})
                        saver.save(sess, save_path, global_step=episode_number)

                        print("SAVED MODEL #step {}, episode{}".format(step, episode_number))
                        last_saved_win_rate = win_rate

            step += 1
            if done: #end of episode
                #reset
                waited_time = 0

                win_loss.append(reward)
                while len(win_loss)>100:
                    win_loss.pop(0)
                # add the samples for the episode to the replay buffer
                # TODO: should we adjust sample rewards based on win or loss of the episode?
                sample_count = len(episodeBuffer.buffer)
                # keep only 2nd half of the samples, so that the replay buffer will have more negative samples
                replay_buffer.add(episodeBuffer.buffer[1::4]
                                  + [copy.deepcopy(episodeBuffer.buffer[-1]) for _ in range(20)])
                # update running reward
                epsilon = math.pow(0.5, episode_number / 3000)
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

                # print progress console
                win_rate = count_win_percentage(win_loss)
                print('\tep {}: running_reward: {}, won {:.2f} %'.format(episode_number, running_reward, win_rate))

                # write the win_rate for tensorboard
                step_assign_op.eval({step_input:step})
                win_rate_summary = sess.run(main_q_net.win_rate_op, feed_dict={main_q_net.win_rate_holder:win_rate})
                main_q_net.writer.add_summary(win_rate_summary, step)

                episode_number += 1  # the Next Episode

                reward_sum = 0
                # reset the the episode buffer for next episode
                episodeBuffer = experience_buffer()

def inference(sess):
    observation = np.zeros(shape=(200, 300))
    prev_x = None
    #create forward graph
    main_q_net = dqn_Model("main", sess)
    main_q_net.forward_graph()

    # restore the model with previously saved weights
    #step_op, step_input, step_assign_op = setup_training_step_count()
    saver, episode_number, save_dir = restore_model(sess)
    #step = step_op.eval(session=sess)
    wait_time = 1
    waited_time = 0


    win_loss = []
    reward_sum = 0
    running_reward = None
    while True:
        # preprocess the observation, set input to network to be difference image
        s_t = prepro(observation)

        if waited_time < wait_time:
            action = 0
            waited_time += 1
        else:
            feed = {main_q_net.s_t: np.reshape(s_t, [-1, height, width, 1])}
            action = sess.run(main_q_net.a_action, feed)[0];

        observation, reward, smallreward, done = game.runGame(action, False)

        reward_sum += reward

        if done:
            # reset
            waited_time = 0
            episode_number += 1

            win_loss.append(reward)
            while len(win_loss) > 100:
                win_loss.pop(0)

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print(
                '\tep {}: reward: {}, won {:.2f} %'.format(episode_number, reward_sum, count_win_percentage(win_loss)))

            reward_sum = 0

def main():

    # tf graph initialization
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()



    if training:
        train(sess)
    else:
        inference(sess)


if __name__=="__main__":
    main()