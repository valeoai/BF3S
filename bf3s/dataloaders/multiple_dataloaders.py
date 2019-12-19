class TwoParallelDataloaders:
    def __init__(self, dataloader, dataloader2):
        self.equal_length = len(dataloader) == len(dataloader2)
        self.dataloader = dataloader
        self.dataloader2 = dataloader2
        self.dataloader_length = len(self.dataloader)
        self.dataset = dataloader.dataset

        if not self.equal_length:
            self.iterator2 = None

    def __call__(self, epoch=0):
        def iterator_equal_length():
            iterator1 = self.dataloader(epoch)
            iterator2 = self.dataloader2(epoch)
            for batch1, batch2 in zip(iterator1, iterator2):
                if not isinstance(batch1, (list, tuple)):
                    batch1 = (batch1,)
                if not isinstance(batch2, (list, tuple)):
                    batch2 = (batch2,)

                yield batch1, batch2

        def iterator():
            iterator1 = self.dataloader(epoch)
            if self.iterator2 is None:
                self.iterator2 = iter(self.dataloader2())

            for batch1 in iterator1:
                try:
                    batch2 = self.iterator2.next()
                except StopIteration:
                    self.iterator2 = iter(self.dataloader2())
                    batch2 = self.iterator2.next()

                if not isinstance(batch1, (list, tuple)):
                    batch1 = (batch1,)
                if not isinstance(batch2, (list, tuple)):
                    batch2 = (batch2,)

                yield batch1, batch2

        if self.equal_length:
            return iterator_equal_length()
        else:
            return iterator()

    def __len__(self):
        return self.dataloader_length
