import tensorflow as tf, Game, numpy as np

game = Game.Game()

inputs = tf.keras.Input(shape=(4,))

layer1 = tf.keras.layers.Dense(16, activation=tf.keras.activations.relu)(inputs)
layer2 = tf.keras.layers.Dense(16, activation=tf.keras.activations.relu)(layer1)

output = tf.keras.layers.Dense(2, activation=tf.keras.activations.linear)(layer2)

model = tf.keras.Model(inputs=inputs, outputs = output)

optimizer = tf.keras.optimizers.Adam(1e-3)


model.compile(loss=tf.keras.losses.mean_squared_error.__name__, optimizer = optimizer)

num_episodes = 5000
num_turns_per_episode = 500



cur_state = game.env.reset()
cur_state = np.reshape(cur_state, [1, 4])
game_turns = []

gamma = 0.95


for episode in range(num_episodes):
    for turn in range(num_turns_per_episode):
        rewards = model.predict(cur_state)
        best_action = np.argmax(rewards)
        next_state, reward, done, info = game.env.step(best_action)
        next_state = np.reshape(next_state, [1, 4])
        game_turns.append([cur_state, best_action, next_state, reward, done])
        if done:
            break
    for cur_state, best_action, next_state, reward, done in game_turns:
        if not done:
            cur_reward = reward
            potential_reward = cur_reward + gamma*np.amax(model.predict(next_state)[0])
            target_reward = model.predict(cur_state)
            target_reward[0][best_action] = potential_reward
            model.fit(cur_state,target_reward, epochs=1)
    game_turns = []
    game.env.reset()
            
cur_state = game.env.reset()
cur_state = np.reshape(cur_state, [1, 4])
game_over = false
while not game_over:
    game.env.render()
    action = np.argmax(model.predict(cur_state))
    observation, reward, done, info = game.env.step(action)
    cur_state = np.reshape(observation, [1, 4])
game.env.close()
