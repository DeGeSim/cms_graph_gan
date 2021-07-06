class CountIterations:
    def __init__(self, iterable):
        self.iterator = iter(iterable)
        self.completed = False
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            res = next(self.iterator)
            self.count += 1
            return res
        except StopIteration:
            self.completed = True
            raise StopIteration
