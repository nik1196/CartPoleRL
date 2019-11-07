import gym

class Game:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.env.reset()

    def play(self):
        for _ in range(1000):
            self.env.render()
            observation, reward, done, info = self.env.step(self.env.action_space.sample())
            if done:
                break
        self.env.close()
