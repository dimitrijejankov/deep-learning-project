import retro
import cv2
import neat
import pickle
import numpy as np

env = None


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:

        # reset
        ob = env.reset()

        # the shape of the screen
        inx, iny, inc = env.observation_space.shape

        # image reduction for faster processing
        inx = int(inx / 8)
        iny = int(iny / 8)

        # 20 Networks
        net = neat.nn.RecurrentNetwork.create(genome, config)

        # stats about our training
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        done = False

        # hp starts at 176
        player_1_hp = 176
        player_2_hp = 176

        action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0])
        while not done:
            env.render()  # Optional

            if frame % 10 == 0:

                # ob is the current frame
                ob = cv2.resize(ob, (inx, iny))

                # make it grayscale
                ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
                ob = np.reshape(ob, (inx, iny))

                oned_image = np.ndarray.flatten(ob)
                neural_net_output = net.activate(oned_image)  # Give an output for current frame from neural network

                # did we kick last time if so disable kicking, since in order to kick we need to release the button
                did_kick = action[0]
                did_punch = action[8]

                #                  P           J, D, R, L, K
                action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                action[0] = neural_net_output[0] < 0.5 and not did_punch
                action[4] = neural_net_output[1]
                action[5] = neural_net_output[2]
                action[6] = neural_net_output[3]
                action[7] = neural_net_output[4]
                action[8] = (neural_net_output[5] < 0.5) and not did_kick

                # try given output from network in the game
                ob, rew, done, info = env.step(action)

                # save player 1 reward
                player_1_reward = player_1_hp - info['health1']
                player_1_hp = info['health1']

                # save player 2 reward
                player_2_reward = player_2_hp - info['health2']
                player_2_hp = info['health2']

                fitness_current += player_1_reward * 4 - player_2_reward
                if fitness_current > current_max_fitness:
                    current_max_fitness = fitness_current
                    counter = 0
                else:
                    counter += 1
                    # count the frames until it successful

                genome.fitness = fitness_current

            else:

                ob, rew, done, info = env.step(action)  # Try given output from network in the game

            # train for max 500
            if done or counter == 50:
                done = True
                print(genome_id, fitness_current)

            # increment the frame
            frame += 1


def main():

    # set the path
    retro.data.path("./")

    # add the custom path
    integration = retro.data.Integrations.CUSTOM_ONLY
    integration.add_custom_path("roms")

    global env
    env = retro.make(game='TeenageMutantNinjaTurtles-Nes', state="LeoVSDonCPU", inttype=integration, players=1)
    #env = retro.make(game='TeenageMutantNinjaTurtles-Nes', state="DonVSLeoCPU", inttype=integration, players=1)
    #env = retro.make(game='TeenageMutantNinjaTurtles-Nes', inttype=integration, players=1)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward-leo')

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # Save the process after each 10 frames
    p.add_reporter(neat.Checkpointer(10))

    winner = p.run(eval_genomes)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)


if __name__ == "__main__":
    main()
