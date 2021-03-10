"""Class static method test"""


class TestClass():
    """Test class"""

    def __init__(self, a=1, b=2):
        self.a = a
        self.b = b

    def add(self):
        return self.a + self.b

    @staticmethod
    def static_add(a, b):
        return 2 * a + 2 * b

    def add2(self):
        return self.static_add(self.a, self.b)


if __name__ == '__main__':
    C = TestClass(a=1, b=2)
    print(C.add())
    print(C.add2())
