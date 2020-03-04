# RL手法ppo2をpybullet環境でstable-baselinesライブラリを用いて実装
import numpy as np
import gym
import pybullet as p
import pybullet_data
import pybullet_envs
import pybulletgym
import os
import time
from datetime import datetime

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv # マルチプロセスで効率よく学習することができます。マルチプロセスが必要な場合は「SubprocVecEnv」を使います。
from stable_baselines.results_plotter import load_results,ts2xy
from stable_baselines import PPO2

import gym_foo

# tensorflow gpu 設定
import tensorflow as tf
tf.Session(config=tf.ConfigProto(device_count = {'GPU':1}))

def make_env(env_name,rank,seed=0):
    """
    Utility function for multiprocessed env.

    :param env_name: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """

    def _init():
        env = gym.make(env_name)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init

# mainがないと　エラーが出た
if __name__ == '__main__':
    

    #学習設定
    train = True    # 学習するかどうか
    validation = True   # 学習結果を使って評価（検証）するかどうか
    # ------------------------------
    # env_name = 'HumanoidPyBulletEnv-v0'
    # env_name ='HumanoidMuJoCoEnv-v0'        #mujoco環境でやってみる
    #env_name ='AntMuJoCoEnv-v0'  #自作のantエージェント環境
    # env_name ='AntBulletEnv-v0'
    # env_name ='Walker2DMuJoCoEnv-v0'
    env_name ='MyAnt-v0'  #自作のantエージェント環境 #これができない
    #--------------------------
    num_cpu = 4     # 学習に使用するCPU数
    #learn_timesteps = 10**3     # 学習タイムステップ  歩行できなかった？多分してない
    # learn_timesteps = 4*10**8
    learn_timesteps = 10**6


    ori_env = gym.make(env_name)
    #ori_env.render() #-----

    env = SubprocVecEnv([make_env(env_name,i) for i in range(num_cpu)])
    # SubprocVecEnv　は複数環境でそれぞれ学習する場合に用いる
    #env.render() #
    env.reset() 
    #time.sleep(5) #

    #savedir = './stable_baselines/{}_{}timesteps_normalant/'.format(env_name,learn_timesteps)
    savedir = './stable_baselines/{}_{}timesteps_myant2/'.format(env_name,learn_timesteps)
    logdir = '/{}tensorboard_log/'.format(savedir)
    os.makedirs(savedir,exist_ok=True)
    # os.mkdir()ではなくos.makedirs()を使うと、中間ディレクトリを作成してくれるので、深い階層のディレクトリを再帰的に作成できる。
    # exist_ok=Trueとすると既に末端ディレクトリが存在している場合もエラーが発生しない。末端ディレクトリが存在していなければ新規作成するし、
    # 存在していれば何もしない。前もって末端ディレクトリの存在確認をする必要がないので便利。

    starttime = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    # 学習の実行
    if train:
        model = PPO2(MlpPolicy,env,verbose=1,tensorboard_log=logdir)
        model.learn(total_timesteps=learn_timesteps)
        model.save('{}ppo2_model'.format(savedir))

    endtime = datetime.now().strftime("%Y/%m/%d %H:%M:%S")


    # 学習結果の確認
    if validation:
        model = PPO2.load('{}ppo2_model'.format(savedir))
       # from gym import wrappers # wrappersを使うとエラーが出るので使わずに画面キャプチャで代用することにした


        video_path = '{}video'.format(savedir)
        #wrap_env = wrappers.Monitor(ori_env,video_path,force=True)
        wrap_env = gym.make(env_name)

        done = False

        #pybulleではreset()の前にrender()を呼ぶ 
        wrap_env.render() #
        obs = wrap_env.reset()

        #--
        # physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version
        # p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        # p.setGravity(0,0,-10)
        # planeId = p.loadURDF("samurai.urdf") #お城のステージ
        #--


        for step in range(200000):
            if step % 10 == 0: print("step :",step)
            if done:
                time.sleep(0.3) # 1秒間処理を止める
                #wrap_env.render() #意味ない
                obs = wrap_env.reset()

               # break

            action, _state = model.predict(obs)
            obs, rewards, done, info = wrap_env.step(action)

        wrap_env.close()

    env.close()

    print(starttime)
    print(endtime)


   
