import random

class RandomStrategyService:
    def run(self, data):
        return random.randint(0, 1)


service = RandomStrategyService()
