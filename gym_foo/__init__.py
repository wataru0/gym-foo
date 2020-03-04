from gym.envs.registration import register

register(
    id = 'foo-v0',
    entry_point = 'gym_foo.envs:FooEnv',
    #envsの後コンマじゃなくてコロン！！
    # ディレクトリの名前:自分で作った環境environmentのクラス名を書く
)
register(
    id = 'MyAnt-v0',
    entry_point = 'gym_foo.envs:AntMuJoCoEnv',
)
