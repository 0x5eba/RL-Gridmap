import pygame
import time
import numpy as np
import scipy.misc
import copy

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


class gameOb():
    def __init__(self, coordinates, reward, name, color):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.reward = reward
        self.name = name
        self.color = color

class Map:
    def __init__(self, rows):
        self.orginal_row = rows
        self.rows = rows
        self.size_margin = 5
        self.size_square = 30
        self.window_size, self.screen = self.set_window()
        self.actions = 8
        self.objects = []

    def set_window(self):
        win_size = self.size_square * self.rows + (self.size_margin * (self.rows + 1))
        screen = pygame.display.set_mode((win_size, win_size))
        return win_size, screen

    def reset(self):
        self.rows = copy.deepcopy(self.orginal_row)
        self.window_size, self.screen = self.set_window()
        self.objects = []
        hero = gameOb(self.newPosition(), None, 'hero', BLUE)
        self.objects.append(hero)
        for _ in range(2):
            goal = gameOb(self.newPosition(), 3, 'goal', GREEN)
            self.objects.append(goal)
        for _ in range(5):
            holes = gameOb(self.newPosition(), -1, 'fire', RED)
            self.objects.append(holes)
        return self.renderEnv()
        
    def newPosition(self):
        points = []
        for row in range(1, self.rows+1):
            for column in range(1, self.rows+1):
                points.append((self.size_margin * row + self.size_square * (row - 1),
                               self.size_margin * column + self.size_square * (column - 1)))
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x, objectA.y) not in currentPositions:
                currentPositions.append((objectA.x, objectA.y))
        for pos in currentPositions:
            points.remove(pos)
        location = np.random.choice(range(len(points)), replace=False)
        return points[location]

    def renderEnv(self):
        self.screen.fill(BLACK)
        for row in range(self.rows):
            for column in range(self.rows):
                pygame.draw.rect(self.screen, WHITE,
                        [(self.size_margin + self.size_square) * column + self.size_margin,
                        (self.size_margin + self.size_square) * row + self.size_margin,
                        self.size_square, self.size_square])

        for block in self.objects:
            if block.name == 'hero':
                pygame.draw.rect(self.screen, block.color, pygame.Rect(
                    block.x, block.y, self.size_square, self.size_square))
            else:
                pygame.draw.rect(self.screen, block.color, pygame.Rect(
                    block.x, block.y, self.size_square, self.size_square))
        
        pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.flip()

        imgdata = pygame.surfarray.array3d(self.screen)
        # imgdata.swapaxes(0,1)
        b = scipy.misc.imresize(
            imgdata[:, :, 0], [84, 84, 1], interp='nearest')
        c = scipy.misc.imresize(
            imgdata[:, :, 1], [84, 84, 1], interp='nearest')
        d = scipy.misc.imresize(
            imgdata[:, :, 2], [84, 84, 1], interp='nearest')
        img = np.stack([b, c, d], axis=2)
        return img

    def updateStart(self, x, y):
        hero = None
        for block in self.objects:
            if block.name == 'hero':
                hero = block
                break
        hero.x += x
        hero.y += y

        score = 0
        if hero.x >= self.window_size or hero.y >= self.window_size or hero.x <= 0 or hero.y <= 0:
            score = -1

        if score != 0:
            hero.x -= x
            hero.y -= y

        for i in range(len(self.objects)):
            if self.objects[i].name == 'hero':
                self.objects[i] = hero
                break

        return score

    def checkGoal(self):
        hero = None
        for block in self.objects:
            if block.name == 'hero':
                hero = block
                break
        for other in self.objects:
            if other.name != 'hero' and hero.x == other.x and hero.y == other.y:
                # self.objects.remove(other)
                # if other.name == 'goal':
                #     self.objects.append(gameOb(self.newPosition(), 3, 'goal', GREEN))
                #     # self.objects.append(gameOb(self.newPosition(), 1, 'goal', GREEN))
                #     # n = int(len([i for i in self.objects if i.reward == -1])*0.2)
                #     # n = 1 if n < 1 else n
                #     # self.rows += int(1/n)
                #     # self.window_size, self.screen = self.set_window()
                #     # self.objects.extend([gameOb(self.newPosition(), -1, 'fire', RED) for i in range(n)])
                # else:
                #     self.objects.append(gameOb(self.newPosition(), -1, 'fire', RED))
                return other.reward
        return -0.1  # penality of a move


    def move(self, action):
        pygame.event.pump()

        if (action == 0): # up
            score = self.updateStart(0, -self.size_margin - self.size_square)
        elif (action == 1): # right
            score = self.updateStart(self.size_margin + self.size_square, 0)
        elif (action == 2): # down
            score = self.updateStart(0, self.size_margin + self.size_square)
        elif (action == 3): # left
            score = self.updateStart(-self.size_margin - self.size_square, 0)
        elif (action == 4): # up-right
            score = self.updateStart(self.size_margin + self.size_square, -self.size_margin - self.size_square)
        elif (action == 5): # right-down
            score = self.updateStart(self.size_margin + self.size_square, self.size_margin + self.size_square)
        elif (action == 6): # down-left
            score = self.updateStart(-self.size_margin - self.size_square, self.size_margin + self.size_square)
        elif (action == 7): # left-up
            score = self.updateStart(-self.size_margin - self.size_square,-self.size_margin - self.size_square)

        pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.flip()

        return score

    def step(self, action):
        penalty = self.move(action)
        reward = self.checkGoal()
        state = self.renderEnv()
        return state, (reward + penalty)
