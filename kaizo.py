# gymの環境を自分用に改造する方法
# gymに用意されている基本的にwrapperクラスを使うことでコードを変えずに振る舞いを上書きできる
# 10/31
import gym
import mujoco_py
import gym_foo
from gym import wrappers

env = gym.make('foo-v0')
# env = wrappers.Monitor(env, './mj-video/')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())



class MyRemodeledEnv(gym.Wrapper): # class作成の時の引数は継承を表す
    def __init__(self,env): # classの重要な関数（コンストラクタと呼ぶ）。インスタンスを作成する時の重要な処理を含むもの
        super().__init_(env) # 親クラスの(__init__の)呼び出し

    def reset(self, **kwargs): # **kwargs: 複数のキーワード引数を辞書として受け取る
        obs = self.env.reset(**kwargs)
        """
        ここにstepの振る舞いを記述
        """
        return obs # 振る舞い変更によって書き換えられた観測を返す

    def step(self,action):
        obs, reward, done, info = self.env.step(action)
        """
        ここにstepの振る舞いを記述
        """
        return obs, reward, done, info

# 報酬を変える場合
class RewardRescallingEnv(gym.RewardWrapper):
    def __init__(self,env,c=0.1): # RewardRescallingEnvを呼び出すときに必要な引数envとc!
        super().__init__(env)
        self.c = c
        lb_r,ub_r = self.env.reward_range # 報酬の外界と上界を取得
        self.reward_range = (lb_r*c,ub_r*c) # self.reward_rangeの上書き

    def reward(self,reward):
        return reward *self.c

# このように、報酬を少し変えるだけならgym.RewardWrapperを継承するだけで十分。
# 同様に、gymには状態や行動を変える専用のラッパークラスObservationWrapperや
# ActionWrapperも用意されています

