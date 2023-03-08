# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import pickle
import time
import pygame
import matplotlib as plt

size = 10
enemy_pen = -200
move_pen = -1
food_reward = 100
episodes = 25000
epsilon = 0.9
epsilon_decay = 0.9998
learning = 0.1
discount = 0.9
start_q_table = None


def py(qtable):
    # pygame stuff
    pygame.init()
    screen = pygame.display.set_mode((300, 300))
    clock = pygame.time.Clock()
    player = pygame.Rect(0, 0, 30, 30)
    food = pygame.Rect(270, 270, 30, 30)
    enemy1 = pygame.Rect(0, 90, 30, 30)
    enemy2 = pygame.Rect(30, 60, 30, 30)
    enemy3 = pygame.Rect(90, 90, 30, 30)
    enemy4 = pygame.Rect(120, 90, 30, 30)
    enemy5 = pygame.Rect(210, 150, 30, 30)

    print(len(pygame.event.get()))
    once = True
    # run = False
    end = False

    # 0 is left
    # 1 is up
    # 2 is right
    # 3 is down
    action_list = []
    pre_action_list = []
    reward = 0
    while len(action_list) < 200:
        obs = (player.x/30, player.y/30)
        if np.random.random() > epsilon:
            action_list.append(np.argmax(qtable[obs]))
        else:
            action_list.append(np.random.randint(0, 4))

        # if action_list[-1] == 2:
        #     print(action_list[-1])
        # print(action_list, 'cur')
        # print(pre_action_list, 'pre')

        screen.fill((0, 0, 0))

        if action_list[-1] == 0 and pre_action_list != action_list and 0 < player.x <= 270:
            player.x -= 30
        elif action_list[-1] == 1 and pre_action_list != action_list and 0 < player.y <= 270:
            player.y -= 30
        elif action_list[-1] == 2 and pre_action_list != action_list and 0 <= player.x < 270:
            player.x += 30
        elif action_list[-1] == 3 and pre_action_list != action_list and 0 <= player.y < 270:
            player.y += 30

        if player.colliderect(food):
            reward_c = food_reward
            print('collision green')
            end = True
        elif player.colliderect(enemy1):
            reward_c = enemy_pen
            print('collision red')
            end = True
        elif player.colliderect(enemy2):
            reward_c = enemy_pen
            print('collision red')
            end = True
        elif player.colliderect(enemy3):
            reward_c = enemy_pen
            print('collision red')
            end = True
        elif player.colliderect(enemy4):
            reward_c = enemy_pen
            print('collision red')
            end = True
        elif player.colliderect(enemy5):
            reward_c = enemy_pen
            print('collision red')
            end = True
        else:
            reward_c = move_pen

        pygame.draw.rect(screen, 'blue', player)
        pygame.draw.rect(screen, 'green', food)
        pygame.draw.rect(screen, 'red', enemy1)
        pygame.draw.rect(screen, 'red', enemy2)
        pygame.draw.rect(screen, 'red', enemy3)
        pygame.draw.rect(screen, 'red', enemy4)
        pygame.draw.rect(screen, 'red', enemy5)

        new_obs = (player.x/30, player.y/30)
        max_future_q = np.max(qtable[new_obs])
        cur_q = qtable[obs][action_list[-1]]

        if reward_c == food_reward:
            new_q = food_reward
        elif reward_c == enemy_pen:
            new_q = enemy_pen
        else:
            new_q = (1 - learning) * cur_q + learning * (reward_c + discount * max_future_q)

        qtable[obs][action_list[-1]] = new_q

        reward += reward_c

        pre_action_list = action_list.copy()
        clock.tick(15)
        if end:
            pygame.quit()
            break
        # print(end, 'end')
        # for event in pygame.event.get():
        #     print('in here')
        #     print(event.type == pygame.QUIT or end, 'comparison')
        #     if event.type == pygame.QUIT or end:
        #         print('broke')
        #         pygame.quit()
        #         run = True
        #         break
        try:
            pygame.display.update()
        except pygame.error:
            pass
    return reward


if start_q_table is None:
    q_table = {}
    for x in range(size):
        for y in range(size):
            q_table[(x, y)] = [np.random.uniform(-5, 0) for i in range(4)]
            # print(q_table)
else:
    with open(start_q_table, 'rb') as f:
        q_table = pickle.load(f)

rewards = []
for i in range(episodes):
    print('currently on episode: ', i)
    rewards.append(py(q_table))
    epsilon *= epsilon_decay

moving_avg = np.convolve(rewards, np.ones((1,)) / 1, mode="valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {1}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
