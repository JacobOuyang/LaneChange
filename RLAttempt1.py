
import numpy as np
import Environment
import tensorflow as tf
import random
import math
import os
MAX_MEMORY_SIZE = 350
# hyperparameters
n_obs = 200 * 300  # dimensionality of observations
h = 200  # number of hidden layer neurons
n_actions = 4  # number of available actions
learning_rate = 1e-3
gamma = .6  # discount factor for reward
gamma2 = 0.5
decay = 0.99  # decay rate for RMSProp gradients
save_path = 'models_Attemp11/Attempt11'
INITIAL_EPSILON = 1

# gamespace
display = True
game=Environment.GameV1(display)
game.populateGameArray()
prev_x = None
xs, ep_rs, ep_rs2, rs, ys = [], [], [], [], []
running_reward = None
reward_sum = 0
observation = np.zeros(shape=(200,300))
episode_number = 0
WillContinue = False

# initialize model
tf_model = {}
with tf.variable_scope('layer_one', reuse=False):
    xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(n_obs), dtype=tf.float32)
    tf_model['W1'] = tf.get_variable("W1", [n_obs, h], initializer=xavier_l1)
with tf.variable_scope('layer_two', reuse=False):
    xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(h), dtype=tf.float32)
    tf_model['W2'] = tf.get_variable("W2", [h, n_actions], initializer=xavier_l2)
def discount_rewards(rewardarray):
    rewardarray.reverse()
    if rewardarray[0] > 0:
        rewardarray[0] = len(rewardarray)/330 * rewardarray[0] *2
    else:
        rewardarray[0] = rewardarray[0]
    #     rewardarray[i] = rewardarray[i] * math.pow(6 - velocityarray[gamenumber], (300 - len(rewardarray) + i) / 300)
    # for i in range(len(rewardarray)):
    #     if rewardarray[i] != 0:
    #         if rewardarray[i] > 0:
    #             rewardarray[i] = (len(rewardarray) - i) / 300 * rewardarray[0]
    #             #gamenumber += 1
    #         else:
    #             rewardarray[i] = rewardarray[i] * math.pow(3,(300 - len(rewardarray) + i) / 300)
    #             #gamenumber += 1

    for i in range(len(rewardarray) -1):

        rewardarray[i+1] = rewardarray[i] * gamma
    rewardarray.reverse()
    return rewardarray
def discount_smallrewards(rewardarray):
    rewardarray.reverse()

    for i in range(len(rewardarray) -1):
        if rewardarray[i] != 0:
            rewardarray[i] = rewardarray[i] * 6
    rewardarray.reverse()
    return rewardarray


# tf operations
def tf_discount_rewards(tf_r):  # tf_r ~ [game_steps,1]
    discount_f = lambda a, v: a * gamma + v;
    tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r, [True, False]))
    tf_discounted_r = tf.reverse(tf_r_reverse, [True, False])
    return tf_discounted_r


def tf_policy_forward(x):  # x ~ [1,D]
    h = tf.matmul(x, tf_model['W1'])
    h = tf.nn.relu(h)
    logp = tf.matmul(h, tf_model['W2'])
    p = tf.nn.softmax(logp)
    return p


# downsampling
#def prepro(I):
 #   """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  #  I = I[35:195]  # crop
   # I = I[::2, ::2, 0]  # downsample by factor of 2
    #I[I == 144] = 0  # erase background (background type 1)
    #I[I == 109] = 0  # erase background (background type 2)
    #I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    #return I.astype(np.float).ravel()


# tf placeholders
tf_x = tf.placeholder(dtype=tf.float32, shape=[None, n_obs], name="tf_x")
tf_y = tf.placeholder(dtype=tf.float32, shape=[None, n_actions], name="tf_y")
tf_epr = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="tf_epr")

# tf reward processing (need tf_discounted_epr for policy gradient wizardry)
#tf_discounted_epr = tf_discount_rewards(tf_epr)
tf_mean, tf_variance = tf.nn.moments(tf_epr, [0], shift=None, name="reward_moments")
tf_epr -= tf_mean
tf_epr /= tf.sqrt(tf_variance + 1e-6)


# tf optimizer op
tf_aprob = tf_policy_forward(tf_x)
l2_loss = tf.nn.l2_loss(tf_y - tf_aprob, name="tf_l2_loss")
optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay)
tf_grads = optimizer.compute_gradients(l2_loss, var_list=tf.trainable_variables(), grad_loss=tf_epr)
train_op = optimizer.apply_gradients(tf_grads)


# write out losses to tensorboard
grad_summaries = []
for g, v in tf_grads:
    if g is not None:
        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        grad_summaries.append(grad_hist_summary)
        grad_summaries.append(sparsity_summary)
grad_summaries_merged = tf.summary.merge(grad_summaries)

l2_loss_summary = tf.summary.scalar("l2_loss", l2_loss)

train_summary_op = tf.summary.merge([l2_loss_summary, grad_summaries_merged])


# tf graph initialization
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()



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
    saver = tf.train.Saver(tf.global_variables())
    episode_number = int(load_path.split('-')[-1])


train_summary_dir = os.path.join(save_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

# training loop
epsilon = INITIAL_EPSILON
while True:


    WillContinue = False
    # preprocess the observation, set input to network to be difference image
    cur_x = observation
    x = cur_x - prev_x if prev_x is not None else np.zeros(n_obs)
    prev_x = cur_x

    # stochastically sample a policy from the network
    feed = {tf_x: np.reshape(x, (1, -1))}
    aprob = sess.run(tf_aprob, feed);
    aprob = aprob[0, :]
    while WillContinue == False:
        if random.random() > epsilon and epsilon > 0:
            findaction = random.random()
            for i in aprob:
                findaction -= aprob[i]
                if findaction <= 0:
                    action = i
                    break
            #action = random.randint(0, 3)
            epsilon -= episode_number / 100000
            observation, reward, smallreward, done = game.runGame(action, True)
            if observation == "REDO":
                WillContinue = False
            else:
                WillContinue = True
        else:
            action = np.random.choice(n_actions, p=aprob)
            observation, reward, smallreward, done = game.runGame(action, False)
            WillContinue = True


    label = np.zeros_like(aprob);
    label[action] = 1

    # step the environment and get new measurements

    reward_sum += reward

    # record game history
   # if len(rs) < MAX_MEMORY_SIZE:
    #    xs.append(x)
     #   ys.append(label)
      #  rs.append(reward)
    #else:
     #   xs.pop(0)
      #  ys.pop(0)
       # rs.pop(0)
    #if np.shape(x) == (2):
    x= np.reshape(x, [-1])
    xs.append(x)
    ys.append(label)
    ep_rs.append(reward)
    ep_rs2.append(smallreward)

    if done:
        # update running reward

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        running = len(ep_rs)
        ep_rs2 = discount_smallrewards(ep_rs2)
        ep_rs = discount_rewards(ep_rs)

        for i in range(len(ep_rs)):
            ep_rs[i] += ep_rs2[i]

        # combine episode rewards
        rs = rs + ep_rs
        #reset episode rewards
        ep_rs = []
        ep_rs2 = []

        if episode_number % 2:
        #if True:
            #rs = np.reshape(np.concatenate(rs), newshape=[-1, 1])
            excess = len(rs) - MAX_MEMORY_SIZE
            if excess < 0:
                excess = 0
            if excess > 0:
                for i in range(excess):
                    rsabs = []
                    for i in range(len(rs)):
                        rsabs.append(math.fabs(rs[i]))
                    lowest = np.argmin(rsabs)
                    rsabs = []
                    rs.pop(lowest)
                    xs.pop(lowest)
                    ys.pop(lowest)



            x_t = np.vstack(xs)
            r_t = np.vstack(rs)
            y_t = np.vstack(ys)

            # parameter update
            feed = {tf_x: x_t, tf_epr: r_t, tf_y: y_t}
            _, train_summaries = sess.run([train_op, train_summary_op], feed)
            # bookkeeping
            xs, rs, ys = [], [], []  # reset game history
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
