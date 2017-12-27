
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
gamma = .99  # discount factor for reward
gamma2 = 0.5
decay = 0.99  # decay rate for RMSProp gradients
save_path = 'Attempt3_models/Attempt3'
INITIAL_EPSILON = .25

# gamespace
display = True
game=Environment.GameV1(display)
game.populateGameArray()
prev_x = None
xs, rs, rs2, ys, vel, velavg = [], [], [], [], [], []
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
def discount_rewards(rewardarray, velocityarray):
    rewardarray.reverse()
    velocityarray.reverse()
    gamenumber = 0
    for i in range(len(rewardarray)):
        if rewardarray[i] != 0:
            if rewardarray[i] > 0:
                rewardarray[i] = (len(rewardarray)-i)/300 * rewardarray[0]
                gamenumber +=1
            else:
                rewardarray[i] = rewardarray[i] * math.pow(6-velocityarray[gamenumber], (300-len(rewardarray)+i)/300)
                gamenumber+=1


    for i in range(len(rewardarray)-1):
        if rewardarray[i+1] == 0:
            rewardarray[i+1] = rewardarray[i] * gamma
    rewardarray.reverse()
    return rewardarray
"""
def discount_smallrewards(rewardarray):
    rewardarray.reverse()

    for i in range(len(rewardarray) -1):
        if rewardarray[i] != 0:
            rewardarray[i] = rewardarray[i]
    rewardarray.reverse()
    return rewardarray
"""

# tf operations
def tf_discount_rewards(tf_r):  # tf_r ~ [game_steps,1]
    discount_f = lambda a, v: a * gamma + v;
    tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r, [True, False]))
    tf_discounted_r = tf.reverse(tf_r_reverse, [True, False])
    return tf_discounted_r


def tf_policy_forward(x):  # x ~ [1,D]
    h = tf.matmul(x, tf_model['W1'])
    h = tf.nn.relu(h)
    logits = tf.matmul(h, tf_model['W2'])
    p = tf.nn.softmax(logits)
    return p, logits


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
tf_y = tf.placeholder(dtype=tf.int32, shape=[None], name="tf_y")
tf_epr = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="tf_epr")

# tf reward processing (need tf_discounted_epr for policy gradient wizardry)
#tf_discounted_epr = tf_discount_rewards(tf_epr)
tf_mean, tf_variance = tf.nn.moments(tf_epr, [0], shift=None, name="reward_moments")
tf_epr -= tf_mean
tf_epr /= tf.sqrt(tf_variance + 1e-6)




# tf optimizer op
tf_aprob, tf_logits = tf_policy_forward(tf_x)
tf_one_hot = tf.one_hot(tf_y, n_actions)
l2_loss = tf.nn.l2_loss(tf_one_hot - tf_aprob, name="tf_l2_loss")
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_y, logits = tf_logits)
ce_loss = tf.reduce_mean(cross_entropy, name="tf_ce_loss")
pg_loss = tf.reduce_mean(tf_epr * cross_entropy, name="tf_pg_loss")
optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay)
tf_grads = optimizer.compute_gradients(pg_loss, var_list=tf.trainable_variables()) #, grad_loss=tf_epr)
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
ce_loss_summary = tf.summary.scalar("ce_loss", ce_loss)
pg_loss_summary = tf.summary.scalar("pg_loss", pg_loss)

train_summary_op = tf.summary.merge([l2_loss_summary, ce_loss_summary, pg_loss_summary, grad_summaries_merged])


# tf graph initialization
sess = tf.InteractiveSession()


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
    episode_number = int(load_path.split('-')[-1])


train_summary_dir = os.path.join(save_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)


# training loop
epsilon = .25 * math.pow(episode_number+1, -1/4)
while True:

    # preprocess the observation, set input to network to be difference image
    cur_x = observation
    x = cur_x - prev_x if prev_x is not None else np.zeros(n_obs)
    prev_x = cur_x

    # stochastically sample a policy from the network
    feed = {tf_x: np.reshape(x, (1, -1))}
    aprob = sess.run(tf_aprob, feed);
    aprob = aprob[0, :]

    if random.random() < epsilon and epsilon > 0:
        #action = random.randint(0, 3)
        action = np.random.choice(n_actions, p=aprob)
        observation, reward, smallreward, velocity, done = game.runGame(action)
    else:
        #action = np.random.choice(n_actions, p=aprob)
        action = np.argmax(aprob)
        observation, reward, smallreward, velocity, done = game.runGame(action)


    label = action

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
    rs.append(reward)
    rs2.append(smallreward)
    vel.append(velocity)
    if done:
        # update running reward
        epsilon = .25 * math.pow(episode_number+1, -1/4)
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        running = len(rs)
        velavg.append(np.mean(vel))
        vel = []
        if episode_number % 2:
        #if True:

           # rs2 = discount_smallrewards(rs2)
            rs = discount_rewards(rs, velavg)

            for i in range(len(rs)):
                rs[i] += rs2[i]

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
            y_t = np.stack(ys)

            # parameter update
            feed = {tf_x: x_t, tf_epr: r_t, tf_y: y_t}
            loss_val, _, train_summaries = sess.run([ce_loss, train_op, train_summary_op], feed)

            train_summary_writer.add_summary(train_summaries, episode_number)


            # bookkeeping
            xs, rs, rs2, ys, velavg = [], [], [], [], [] # reset game history

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

