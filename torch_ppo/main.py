import torch
import retro
from torch_ppo.ppo import Memory, PPO
from torch_ppo.wrappers import UserControllerWrapper, TrainingWrapper

# starter code from https://github.com/nikhilbarhate99/PPO-PyTorch


############## Hyperparameters ##############
action_dim = 18
render = False

solved_reward = 230  # stop training if avg_reward > solved_reward
log_interval = 20  # print avg reward in the interval
max_episodes = 500  # max training episodes
max_timesteps = 1000  # max timesteps in one episode
n_latent_var = 64  # number of variables in hidden layer
update_timestep = 2000  # update policy every n timesteps


lr = [0.01, 0.03, 0.09, 0.1, 0.3]
gamma = [0.90, 0.95, 0.99]  # discount factor
eps_clip = [0.1, 0.2, 0.3]  # clip parameter for PPO

betas = (0.9, 0.999)
K_epochs = 4  # update policy for K epochs
random_seed = None
#############################################

def main():
    for mylr in lr:
        for mygamma in gamma:
            for myclip in eps_clip:
                tun_with_hyperparameters(mylr, mygamma, myclip)


def tun_with_hyperparameters(learning_rate, gamma_value, clip_value):

    print("learning rate: ", learning_rate, "gamma value: ", gamma_value, "clip value: ", clip_value)
    # 1. Create gym environment
    retro.data.path("./")

    # add the custom path
    integration = retro.data.Integrations.CUSTOM_ONLY
    integration.add_custom_path("roms")

    # env = retro.make(game='TeenageMutantNinjaTurtles-Nes', state="LeoVSDonCPU", inttype=integration, players=1)
    env = retro.make(game='TeenageMutantNinjaTurtles-Nes', state="DonVSLeoCPU", inttype=integration, players=2)

    # this wraps the environment and shapes the input and reward so that we like it
    env = TrainingWrapper(False, env, integration)

    # this also wraps the environment it translates the output of the nn to the player 1 controls
    env = UserControllerWrapper(env, False)

    state_dim = env.observation_space.shape[0]

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, learning_rate, betas, gamma_value, K_epochs, clip_value)
    print(learning_rate, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1

            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval * solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format("nes"))
            break

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


if __name__ == '__main__':
    main()
