class _Indexable:
    __slots__ = ()

    def __getitem__(self, index):
        return index


index = _Indexable()
