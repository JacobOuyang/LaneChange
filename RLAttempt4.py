
import numpy as np
import Environment
import tensorflow as tf
import random
import math
import os
import cv2

MAX_MEMORY_SIZE = 350
# hyperparameters
n_obs = 100 * 50  # dimensionality of observations
h = 200  # number of hidden layer neurons
n_actions = 4  # number of available actions
learning_rate = 1e-4
gamma = .90  # discount factor for reward

decay = 0.99  # decay rate for RMSProp gradients
save_path = 'models_Attemp25/Attempt25'
INITIAL_EPSILON = 1

# gamespace
display = True
training = False

game=Environment.GameV1(display)
game.populateGameArray()



class RL_Model():
    def __init__(self):
        self.tf_model = {}
        with tf.variable_scope('layer_one', reuse=False):
            xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(n_obs), dtype=tf.float32)
            self.tf_model['W1'] = tf.get_variable("W1", [n_obs, h], initializer=xavier_l1)
            self.tf_model['b1'] = tf.Variable(tf.truncated_normal([h], stddev=1./np.sqrt(h), dtype=tf.float32), name='b1')
        with tf.variable_scope('layer_two', reuse=False):
            xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(h), dtype=tf.float32)
            self.tf_model['W2'] = tf.get_variable("W2", [h, n_actions], initializer=xavier_l2)
            self.tf_model['b2'] = tf.Variable(tf.truncated_normal([n_actions], stddev=.5, dtype=tf.float32), name="b2")

    def tf_policy_forward(self, tf_x, tf_model):  # x ~ [1,D]
        h = tf.matmul(tf_x, tf_model['W1']) + tf_model["b1"]
        h = tf.nn.relu(h)
        self.logits = tf.matmul(h, tf_model['W2']) + tf_model["b2"]
        self.p = tf.nn.softmax(self.logits)
        return self.p, self.logits

    def forward_graph(self):
        # tf placeholders
        self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, n_obs], name="tf_x")

        # tf optimizer op
        tf_aprob, tf_logits = self.tf_policy_forward(self.tf_x, tf_model=self.tf_model)

        return self.tf_x, tf_aprob, tf_logits

    def train_graph(self, tf_aprob, tf_logits):

        self.tf_y = tf.placeholder(dtype=tf.int32, shape=[None], name="tf_y")
        self.tf_epr = tf.placeholder(dtype=tf.float32, shape=[None], name="tf_epr")

        # tf reward processing (need tf_discounted_epr for policy gradient wizardry)
        # tf_discounted_epr = tf_discount_rewards(tf_epr)
        tf_mean, tf_variance = tf.nn.moments(self.tf_epr, [0], shift=None, name="reward_moments")
        tf_epr_normed = self.tf_epr - tf_mean
        tf_epr_normed /= tf.sqrt(tf_variance + 1e-6)
        tf_one_hot = tf.one_hot(self.tf_y, n_actions)

        self.entropy_loss = tf.reduce_mean(tf_aprob * tf.log(self.aprob), name="tf_ce_loss")
        self.responsbile_outputs = tf.reduce_sum(tf_aprob * tf_one_hot, [1])
        self.pg_loss = - tf.reduce_sum(tf_epr_normed * tf.log(self.responsbile_outputs), name="tf_pg_loss")
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay)
        tf_grads = optimizer.compute_gradients(self.pg_loss, var_list=tf.trainable_variables())

        self.train_op = optimizer.apply_gradients(tf_grads)

        # write out losses to tensorboard
        grad_summaries = []
        for g, v in tf_grads:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        l2_loss_summary = tf.summary.scalar("l2_loss", self.l2_loss)
        ce_loss_summary = tf.summary.scalar("ce_loss", self.ce_loss)
        pg_loss_summary = tf.summary.scalar("pg_loss", self.pg_loss)

        self.train_summary_op = tf.summary.merge([l2_loss_summary, ce_loss_summary, pg_loss_summary, grad_summaries_merged])

        return self.tf_y, self.tf_epr, self.train_op, self.train_summary_op

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
    return I.astype(np.float)

def diff(X0, X1):
    X0[X0==0.5] = 0 # erase the player car in the first image
    diff_image = X1 - X0

    displayImage(diff_image)
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
    else:
        print(
            "loaded model: {}".format(load_path))
        #saver = tf.train.Saver(tf.global_variables())
        episode_number = int(load_path.split('-')[-1])
    return saver, episode_number, save_dir



def train(sess, rl_model):

    # create forward graph
    tf_x, tf_aprob, tf_logits = rl_model.forward_graph()

    # create train graph
    tf_y, tf_epr, train_op, train_summary_op = rl_model.train_graph(tf_aprob=tf_aprob, tf_logits=tf_logits)

    # restore the model with previously saved weights
    saver, episode_number, save_dir = restore_model(sess)

    train_summary_dir = os.path.join(save_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)


    epsilon = math.pow(0.5, episode_number/1000)

    wait_time = 1
    waited_time = 0
    prev_x = None
    xs, rs, rs2, ys = [], [], [], []
    running_reward = None
    reward_sum = 0
    observation = np.zeros(shape=(200, 300))

    # training loop
    while True:

        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(n_obs)
        prev_x = cur_x


        if waited_time < wait_time:
            action = 0
            waited_time += 1
            observation, reward, smallreward, done = game.runGame(action, False)
        else:
            # stochastically sample a policy from the network
            feed = {tf_x: np.reshape(x, (1, -1))}
            aprob = sess.run(tf_aprob, feed);
            aprob = aprob[0, :]

            if random.random() < epsilon:
                action = random.randint(0, n_actions-1)
            else:
                action = np.random.choice(n_actions, p=aprob)


            observation, reward, smallreward, done = game.runGame(action, False)

            label = action
            # step the environment and get new measurements
            reward_sum += reward

            x= np.reshape(x, [-1])
            xs.append(x)
            ys.append(label)
            rs.append(reward)
            rs2.append(smallreward)


            if done:
                #reset
                waited_time = 0
                prev_x = None


                # update running reward
                epsilon = math.pow(0.5, episode_number / 1000)
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                running = len(rs)

                if episode_number % 2:
                    rs2 = discount_smallrewards(rs2)
                    rs = discount_rewards(rs)

                    # combine both rewards
                    for i in range(len(rs)):
                        rs[i] += rs2[i]

                    # remove
                    excess = len(rs) - MAX_MEMORY_SIZE
                    if excess < 0:
                        excess = 0
                    if excess > 0:
                        for i in range(excess):
                            rsabs = []
                            for i in range(len(rs)):
                                rsabs.append(math.fabs(rs[i]))
                            lowest = np.argmin(rsabs)
                            rs.pop(lowest)
                            xs.pop(lowest)
                            ys.pop(lowest)

                    x_t = np.vstack(xs)
                    r_t = np.stack(rs)
                    y_t = np.stack(ys)

                    # parameter update
                    feed = {tf_x: x_t, tf_epr: r_t, tf_y: y_t}

                    #epr_val, mean_val, variance_val, l2_loss_val, ce_loss_val, ce_val = \
                    #    sess.run([tf_epr_normed, tf_mean, tf_variance, l2_loss, ce_loss, cross_entropy],
                    #                                                    feed)
                    _, train_summaries = sess.run([train_op, train_summary_op], feed)


                    # bookkeeping
                    xs, rs, rs2, ys = [], [], [], []  # reset game history

                    train_summary_writer.add_summary(train_summaries, episode_number)

                # print progress console
                if episode_number % 5 == 0:
                    print(
                    'ep {}: reward: {}, mean reward: {:3f}, running: {}'.format(episode_number, reward_sum, running_reward, running))
                else:
                    print(
                    '\tep {}: reward: {}'.format(episode_number, reward_sum))


                episode_number += 1  # the Next Episode

                reward_sum = 0
                if episode_number % 1000 == 0:
                    saver.save(sess, save_path, global_step=episode_number)
                    print(
                    "SAVED MODEL #{}".format(episode_number))

def inference(sess, rl_model):
    observation = np.zeros(shape=(200, 300))
    prev_x = None
    #create forward graph
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
            feed = {tf_x: np.reshape(x, (1, -1))}
            aprob = sess.run(tf_aprob, feed);
            action = np.argmax(aprob)

        observation, reward, smallreward, done = game.runGame(action, False)


        if done:
            # reset
            waited_time = 0
            prev_x = None


def main():

    rl_model = RL_Model()

    # tf graph initialization
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()



    if training:
        train(sess, rl_model)
    else:
        inference(sess, rl_model)


if __name__=="__main__":
    main()