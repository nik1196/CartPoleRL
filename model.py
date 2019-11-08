import tensorflow as tf, Game, numpy as np, os, random
from collections import deque

game = Game.Game()
game.env._max_episode_steps = 10000
model = []
retrain = True
if not os.path.exists('new_models/model') or retrain:
    print("new model")
    if not os.path.exists('new_models/'):
        os.mkdir('new_models')
    inputs = tf.keras.Input(shape=(4,))

    layer1 = tf.keras.layers.Dense(24, activation=tf.keras.activations.relu)(inputs)
    layer2 = tf.keras.layers.Dense(24, activation=tf.keras.activations.relu)(layer1)

    output = tf.keras.layers.Dense(2, activation=tf.keras.activations.linear)(layer2)

    model = tf.keras.Model(inputs=inputs, outputs = output)

    optimizer = tf.keras.optimizers.Adam(1e-3)


    model.compile(loss=tf.keras.losses.mean_squared_error.__name__, optimizer = optimizer)

    num_episodes = 5000
    num_turns_per_episode = 500

    epsilon = 1.0
    epsilon_min = 0.25
    epsilon_decay = 0.995

    game_turns = deque(maxlen=2000)

    gamma = 0.95
##
##    if retrain:
##        model = tf.keras.models.load_model('new_models/model')
    for episode in range(num_episodes):
        cur_state = game.env.reset()
        cur_state = np.reshape(cur_state, [1, 4])
        turn = 0
        while True:
            turn += 1
            rewards = model.predict(cur_state)
            best_action = 0
            if np.random.rand() >= epsilon:
                best_action = game.env.action_space.sample()
            else:
                best_action = np.argmax(rewards[0])
            next_state, reward, done, info = game.env.step(best_action)
            next_state = np.reshape(next_state, [1, 4])
            game_turns.append((cur_state, best_action, next_state, reward, done))
            cur_state = next_state
            if done:
                print("Episode:", episode, " Score:", turn, " Epsilon:", epsilon)
                break
        if len(game_turns) >= 32:
            sample = random.sample(game_turns, 32)
            for cur_state, best_action, next_state, reward, done in sample:
                potential_reward = reward
                if not done:
                    potential_reward = reward + gamma*np.amax(model.predict(next_state)[0])
                target_reward = model.predict(cur_state)
                target_reward[0][best_action] = potential_reward
                model.fit(cur_state,target_reward, epochs=1,verbose=0)
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            else:
                epsilon = 1.0
    tf.keras.models.save_model(model, 'new_models/model')
else:
    model = tf.keras.models.load_model('new_models/model')   
cur_state = game.env.reset()
cur_state = np.reshape(cur_state, [1, 4])
game_over = False
time = 0
while not game_over:
    game.env.render()
    time += 1
    action = np.argmax(model.predict(cur_state))
    observation, reward, done, info = game.env.step(action)
    cur_state = np.reshape(observation, [1, 4])
    game_over = done
print(time)
game.env.close()
