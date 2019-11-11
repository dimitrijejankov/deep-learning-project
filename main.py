import retro


# health starts at 176
# put the game rom into retro/data/roms/TeenageMutantNinjaTurtles-Nes/rom.nes

def main():
    retro.data.path("./")

    # add the custom path
    integration = retro.data.Integrations.CUSTOM_ONLY
    integration.add_custom_path("roms")

    # load the game for two players
    env = retro.make(game='TeenageMutantNinjaTurtles-Nes', inttype=integration, players=2)
    obs = env.reset()

    # while the game is running do stuff
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        print("Player 1 %s health, Player 1 won %s rounds, Player 2 %s health, Player 2 won %s rounds" % (
            info['health1'], info['rounds1'], info['health2'], info['rounds2']))
        if done:
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()
