e = 0.05  # The chance of chosing a random action
# How many episodes of game environment to train network with.
num_episodes = 10000
load_model = True  # Whether to load a saved model.
path = "./fix_map_models"  # The path to save/load our model to/from.
# path = "./Central"  # The path to save our log files for use in Control Center.
# The size of the final convolutional layer before splitting it into Advantage and Value streams.
h_size = 512
max_epLength = 100  # The max allowed length of our episode.tf.reset_default_graph()
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)

init = tf.initialize_all_variables()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

#create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0


with tf.Session() as sess:
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        print ckpt
        saver.restore(sess, ckpt.model_checkpoint_path)
    for i in range(num_episodes):
        episodeBuffer = experience_buffer()
        s = env.reset()
        s = processState(s)
        rAll = 0
        j = 0
        #The Q-Network
        # If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
        while j < max_epLength:
            j += 1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e:
                a = np.random.randint(0, 4)
            else:
                a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]
            s1, r = env.step(a)
            s1 = processState(s1)
            total_steps += 1
            # Save the experience to our episode buffer.
            episodeBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 4]))
            rAll += r
            s = s1

        #Get all experiences from this episode and discount their rewards.
        episodeRewards = np.array(episodeBuffer.buffer)[:, 2]
        discountRewards = discount_rewards(episodeRewards)
        bufferArray = np.array(episodeBuffer.buffer)
        bufferArray[:, 2] = discountRewards
        episodeBuffer.buffer = bufferArray
        #Add the discounted experiences to our experience buffer.
        jList.append(j)
        rList.append(rAll)

        if len(rList) % 100 == 0:
            with open('./Center/log.csv', 'a') as myfile:
                images = zip(bufferArray[:, 0])
                images.append(bufferArray[-1, 3])
                images = np.vstack(images)
                images = np.resize(images, [len(images), 84, 84, 3])
                make_gif(images, './Center/frames/image' + str(i) +
                         '.gif', duration=len(images), true_image=True)
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow([i, np.mean(jList[-50:]), np.mean(rList[-50:]),
                             './frames/image' + str(i) + '.gif', './frames/log' + str(i) + '.csv'])
                myfile.close()
            with open('./Center/frames/log' + str(i) + '.csv', 'w') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(["ACTION", "REWARD", "A0", "A1", 'A2', 'A3', 'V'])
                a, v = sess.run([mainQN.Advantage, mainQN.Value], feed_dict={
                                mainQN.scalarInput: np.vstack(bufferArray[:, 0])})
                wr.writerows(zip(
                    bufferArray[:, 1], bufferArray[:, 2], a[:, 0], a[:, 1], a[:, 2], a[:, 3], v[:, 0]))
            print total_steps, np.mean(rList[-50:]), e
print "Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%"
