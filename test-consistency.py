from atari import Atari
import os
import pickle
import numpy as np

if __name__ == '__main__':
    name = 'MontezumaRevenge'

    with open(
        os.path.join('.', 'go-explore-action-seqs-all-trajectories.pkl'), 'rb'
    ) as f:
        action_seqs = pickle.load(f)
        break_flag = False
        for item in action_seqs:
            if break_flag:
                break
            list_of_actions, cumulative_reward, timestamp = item

            timestamp = 'go-explore-{}'.format(timestamp)
            go_explore_episode_s_a = []
            go_explore_env = Atari(
                name=name,
                sticky=False,
            )
            score = 0
            s_0 = go_explore_env.reset(seed=0)
            for a in list_of_actions:
                s_1, reward, done, _ = go_explore_env.step(a)
                score += reward
                go_explore_episode_s_a.append(
                    (
                        s_0, a, np.clip(np.asarray(reward), -1,
                                        1), s_1, float(done)
                    )
                )
                s_0 = s_1
                if done:
                    assert score == cumulative_reward, 'recovering the exact trajectory error! score: {} cumulative_reward: {}'.format(
                        score, cumulative_reward
                    )
                    if score < 10_000:
                        print(
                            'dequeue. go-explore t: {: <20}, score: {: <10}, #transitions: {}'
                            .format(
                                timestamp, score, len(go_explore_episode_s_a)
                            )
                        )
                        #go_explore_expert_replay.append(
                        #    {
                        #        'timestamp': timestamp,
                        #        'return': score,
                        #        'exp': go_explore_episode_s_a,
                        #    }
                        #)
                    else:
                        break_flag = True
