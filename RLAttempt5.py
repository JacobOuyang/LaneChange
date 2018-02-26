
import numpy as np
import Environment
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import random
import math
import os
import cv2
import copy
import itertools

TRAIN_FREQUENCY = 4

MAX_MEMORY_SIZE = 500
# hyperparameters
height = 50
width = 100
n_obs = 100 * 50  # dimensionality of observations
h = 200  # number of hidden layer neurons
n_actions = 4  # number of available actions
learning_rate = 0.000025
gamma = .99  # discount factor for reward

batch_size = 32

learn_start = 1000
learning_rate_minimum = 0.00025,
learning_rate_decay_step = 5 * 10000,
learning_rate_decay = 0.96,
target_q_update_step = 1000
decay = 0.99  # decay rate for RMSProp gradients
save_path = 'models_Attemp50/Attempt50'
INITIAL_EPSILON = 1

# gamespace
display = False
training = True

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

class RL_Model():
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
                                                             32, [3, 3], [4, 4], initializer, activation_fn,
                                                             "NHWC", name='l1')
            self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
                                                             64, [3, 3], [2, 2], initializer, activation_fn,
                                                             "NHWC", name='l2')
            self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
                                                             64, [3, 3], [1, 1], initializer, activation_fn,
                                                             "NHWC", name='l3')

            self.l3_flat = flatten(self.l3)

            self.fc1, self.w['l4_w'], self.w['l4_b'] = \
                linear(self.l3_flat, 512, activation_fn=activation_fn, name='fc1')

            self.output, self.w['output_w'], self.w['output_b'] = \
                linear(self.fc1, n_actions, name='output_out')


            tf_aprob = tf.nn.softmax(self.output)
            return self.s_t, tf_aprob, self.output
    def train_graph(self, tf_aprob, tf_logits):

        self.tf_y = tf.placeholder(dtype=tf.int32, shape=[None], name="tf_y")
        self.tf_epr = tf.placeholder(dtype=tf.float32, shape=[None], name="tf_epr")

        tf_one_hot = tf.one_hot(self.tf_y, n_actions)
        self.entropy_loss = tf.reduce_mean(tf_aprob * tf.log(tf_aprob), name="tf_ce_loss")
        self.responsbile_outputs = tf.reduce_sum(tf_aprob * tf_one_hot, [1])
        self.pg_loss = -tf.reduce_sum(tf.log(self.responsbile_outputs) * self.tf_epr)
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay)

        tvs = tf.trainable_variables()
        self.accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_vars]

        self.tf_grads = optimizer.compute_gradients(self.pg_loss, var_list=tvs)
        self.accum_ops = [self.accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(self.tf_grads)]

        self.tf_episode_divisor = tf.placeholder(dtype=tf.float32)
        self.episode_divisor = tf.Variable(0.0,  dtype=tf.float32, trainable=False, name='episode_divisor')
        self.tf_episode_divisor_assign = self.episode_divisor.assign(self.tf_episode_divisor)

        step_gradient = [(tf.scalar_mul(self.episode_divisor, self.accum_vars[i]), gv[1]) for i, gv in enumerate(self.tf_grads)]

        self.train_op = optimizer.apply_gradients(step_gradient)



        # write out losses to tensorboard
        grad_summaries = []
        for i, v in enumerate(self.tf_grads):
            if v[0] is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v[1].name), step_gradient[i])
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v[1].name), tf.nn.zero_fraction(step_gradient[i]))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)


        pg_loss_summary = tf.summary.scalar("pg_loss", self.pg_loss)

        self.train_summary_op = tf.summary.merge([pg_loss_summary, grad_summaries_merged])

        return self.tf_y, self.tf_epr, self.train_op, self.train_summary_op
class experience_buffer():
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 3])


def discount_rewards(rewardarray):

    '''gamenumber = 0
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
    '''
    rewardarray.reverse()
    if rewardarray[0] >0:
        rewardarray[0] += ((167-len(rewardarray))/30)
    else:
        rewardarray[0] += len(rewardarray)/170 #rewardarray[0] * math.pow(3, (170-len(rewardarray))/170)
    #rewardarray.reverse()
    for i in range(len(rewardarray) -1):

        rewardarray[i+1] += rewardarray[i] * gamma
    rewardarray.reverse()
    return rewardarray


def discount_smallrewards(rewardarray):

    for i in range(len(rewardarray) -1):
        if rewardarray[i] >=0:
            rewardarray[i] = rewardarray[i] * 5
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
    return I.astype(np.float)

def diff(X0, X1):
    #X0[X0==0.5] = 0 # erase the player car in the first image
    diff_image = X1 - X0 *0.75

    #displayImage(diff_image)
    return diff_image


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
        load_was_success = False
        tf.global_variables_initializer().run()
    else:
        print(
            "loaded model: {}".format(load_path))
        # saver = tf.train.Saver(tf.global_variables())
        episode_number = int(load_path.split('-')[-1])
    return saver, episode_number, save_dir


def train(sess):
    # create forward graph
    rl_model = RL_Model('main', sess)
    tf_x, tf_aprob, tf_logits = rl_model.forward_graph()

    # create train graph
    tf_y, tf_epr, train_op, train_summary_op = rl_model.train_graph(tf_aprob=tf_aprob, tf_logits=tf_logits)

    # restore the model with previously saved weights
    saver, episode_number, save_dir = restore_model(sess)

    train_summary_dir = os.path.join(save_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    epsilon = math.pow(0.5, episode_number / 2000)

    wait_time = 1
    waited_time = 0
    prev_x = None
    exs, ers, ers2, eys = [], [], [], []
    xs, rs, ys = [], [], []
    sess.run(rl_model.zero_ops)

    running_reward = None
    reward_sum = 0
    observation = np.zeros(shape=(200, 300))

    episode_number_in_batch = 0
    # training loop
    while True:

        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation)
        x = diff(prev_x, cur_x) if prev_x is not None else np.zeros(shape =(50,100))

        prev_x = cur_x

        if waited_time < wait_time:
            action = 0
            waited_time += 1
            observation, reward, smallreward, done = game.runGame(action, False)
        else:
            # stochastically sample a policy from the network
            feed = {rl_model.s_t:  np.reshape(x, [-1, height, width, 1])}
            aprob = sess.run(tf_aprob, feed);
            aprob = aprob[0, :]

            if random.random() < epsilon:
                action = random.randint(0, n_actions - 1)
            else:
                action = np.random.choice(n_actions, p=aprob)

            observation, reward, smallreward, done = game.runGame(action, False)

            label = action
            # step the environment and get new measurements
            reward_sum += reward

            x = np.expand_dims(x, -1)
            exs.append(x)
            eys.append(label)
            ers.append(reward)
            ers2.append(smallreward)

            if done:
                # reset
                waited_time = 0
                prev_x = None
                episode_number_in_batch += 1
                # update running reward
                epsilon = math.pow(0.5, episode_number / 2000)
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                running = len(ers)

                ers2 = discount_smallrewards(ers2)
                for i in range(len(ers)):
                    ers[i] += ers2[i]

                ers = discount_rewards(ers)
                mean = np.mean(ers)
                stddev = np.std(ers) + 1e-6
                ers -= mean
                ers /= stddev

                rs = list(itertools.chain(rs,ers))
                xs = list(itertools.chain(xs, exs))
                ys = list(itertools.chain(ys, eys))
                exs, ers, ers2, eys = [], [], [], []

                # remove things more than max memory
                excess = len(rs) - MAX_MEMORY_SIZE
                if excess > 0:
                    for i in range(excess):
                        rsabs = []
                        for i in range(len(rs)):
                            rsabs.append(math.fabs(rs[i]))
                        lowest = np.argmin(rsabs)
                        rs.pop(lowest)
                        xs.pop(lowest)
                        ys.pop(lowest)
                        rsabs.pop(lowest)


                if len(rs) > 200:
                    start = len(rs) % batch_size
                    end = start
                    count = int(len(rs)/32)

                    for i in range(count):
                        start = end
                        end = start + batch_size
                        x_t = np.stack(xs[start:end])
                        r_t = np.stack(rs[start:end])
                        y_t = np.stack(ys[start:end])

                        # parameter update
                        feed = {tf_x: x_t, tf_epr: r_t, tf_y: y_t}

                        sess.run([rl_model.accum_ops], feed)

                    feed = {tf_x: x_t, tf_epr: r_t, tf_y: y_t, rl_model.episode_divisor: 1.0/episode_number_in_batch}
                    _, train_summaries = sess.run([train_op, train_summary_op], feed)
                    train_summary_writer.add_summary(train_summaries, episode_number)
                        # bookkeeping
                    xs, rs, ys = [], [], []  # reset game history
                    sess.run(rl_model.zero_ops)
                    episode_number_in_batch = 0

                # print progress console
                if episode_number % 5 == 0:
                    print(
                        'ep {}: reward: {}, mean reward: {:3f}, running: {}'.format(episode_number, reward_sum,
                                                                                    running_reward, running))
                else:
                    print(
                        '\tep {}: reward: {}'.format(episode_number, reward_sum))

                episode_number += 1  # the Next Episode

                reward_sum = 0
                if episode_number % 1000 == 0:
                    saver.save(sess, save_path, global_step=episode_number)
                    print(
                        "SAVED MODEL #{}".format(episode_number))


def inference(sess):
    rl_model = RL_Model('main', sess)
    observation = np.zeros(shape=(200, 300))
    prev_x = None
    # create forward graph
    tf_x, tf_aprob, tf_logits = rl_model.forward_graph()

    # restore the model with previously saved weights
    saver, episode_number, save_dir = restore_model(sess)

    wait_time = 1
    waited_time = 0

    while True:
        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(n_obs)
        prev_x = cur_x

        if waited_time < wait_time:
            action = 0
            waited_time += 1
        else:
            feed = {tf_x: np.reshape(x, [-1, height, width, 1])}
            aprob = sess.run(tf_aprob, feed);
            action = np.argmax(aprob)

        observation, reward, smallreward, done = game.runGame(action, False)

        if done:
            # reset
            waited_time = 0
            prev_x = None


def main():
    # tf graph initialization
    sess = tf.InteractiveSession()


    if training:
        train(sess)
    else:
        inference(sess)


if __name__ == "__main__":
    main()
