
class Domain:
    def __init__(self, values):
        self.values = tuple(values)

    def __hash__(self):
        return hash(self.values)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.values == other.values
        )


class RV:
    def __init__(self, domain, value=None, name=None):
        # e.g. binary vr, domain: (0, 1)
        self.domain = domain
        # example value: A = 0
        self.value = value
        self.name = name
        self.nb = list()


class F:
    def __init__(self, table, nb):
        self.table = table
        # factor connects random variables
        self.nb = nb


class Graph:
    def __init__(self, rvs, fs):
        self.rvs = rvs
        self.fs = fs
        self.init_nb()

    def init_nb(self):
        for rv in self.rvs:
            rv.nb = list()
        for f in self.fs:
            for rv in f.nb:
                rv.nb.append(f)

    def rv_name_to_rv(self, name):
        for rv in self.rvs:
            if rv.name == name:
                return rv

        print("No such variable")
