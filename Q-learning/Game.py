import pygame
import time

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


class Map:
    def __init__(self, window_size, square, nsr, margin):
        self.window_size = window_size
        self.square = square
        self.nsr = nsr
        self.margin = margin
        self.screen = pygame.display.set_mode((window_size, window_size))
        self.start = None
        self.x_start = 0
        self.y_start = 0
        self.blocks = []
        self.end = []

    def updateStart(self, x, y):
        new_x = self.x_start + x
        new_y = self.y_start + y

        score = 0

        if new_x >= self.window_size or new_y >= self.window_size or new_x <= 0 or new_y <= 0:
            score =  -1

        for block in self.blocks:
            if new_x == block[0] and new_y == block[1]:
                score = -1
                break

        if score == -1:
            self.start = pygame.Rect(self.x_start, self.y_start, self.square, self.square)
            pygame.draw.rect(self.screen, BLUE, self.start)
            return score

        self.start = pygame.Rect(new_x, new_y, self.square, self.square)
        pygame.draw.rect(self.screen, BLUE, self.start)
        
        if new_x == self.end[0] and new_y == self.end[1]:
            score = 1

        self.x_start = new_x
        self.y_start = new_y
        return score

    def updateMap(self):
        self.screen.fill(BLACK)
        for row in range(self.nsr):
            for column in range(self.nsr):
                pygame.draw.rect(self.screen, WHITE,
                                 [(self.margin + self.square) * column + self.margin, (self.margin + self.square) * row + self.margin,
                                  self.square, self.square])

        for block in self.blocks:
            pygame.draw.rect(self.screen, RED, pygame.Rect(block[0], block[1], self.square, self.square))

        end = pygame.Rect(self.end[0], self.end[1], self.square, self.square)
        pygame.draw.rect(self.screen, GREEN, end)

    def getPresentFrame(self):
        pygame.event.pump()


        self.blocks.append([self.margin * 2 + self.square, self.margin * 3 + self.square * 2])
        self.blocks.append([self.margin * 3 + self.square * 2, self.margin * 3 + self.square * 2])
        self.blocks.append([self.margin * 4 + self.square * 3, self.margin * 5 + self.square * 4])
        self.blocks.append([self.margin * 5 + self.square * 4, self.margin * 5 + self.square * 4])
        self.blocks.append([self.margin * 5 + self.square * 4, self.margin * 4 + self.square * 3])
        self.blocks.append([self.margin * 6 + self.square * 5, self.margin * 4 + self.square * 3])


        self.end = [self.window_size - self.square - self.margin, self.window_size - self.square - self.margin]

        self.updateMap()

        self.updateStart(self.margin, self.margin)


        pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.flip()

    def getNextAction(self, action):
        pygame.event.pump()

        self.updateMap()

        if (action[0] == 1): # up
            score = self.updateStart(0, -self.margin - self.square)
        elif (action[1] == 1): # right
            score = self.updateStart(self.margin + self.square, 0)
        elif (action[2] == 1): # down
            score = self.updateStart(0, self.margin + self.square)
        elif (action[3] == 1): # left
            score = self.updateStart(-self.margin - self.square, 0)
        elif (action[4] == 1): # up-right
            score = self.updateStart(self.margin + self.square, -self.margin - self.square)
        elif (action[5] == 1): # right-down
            score = self.updateStart(self.margin + self.square, self.margin + self.square)
        elif (action[6] == 1): # down-left
            score = self.updateStart(-self.margin - self.square, self.margin + self.square)
        elif (action[7] == 1): # left-up
            score = self.updateStart(-self.margin - self.square,-self.margin - self.square)

        score -= 0.04 # penality of a move
            
        pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.flip()

        return score, (self.x_start, self.y_start)
