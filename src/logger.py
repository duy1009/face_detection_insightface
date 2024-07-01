
class LogCSV():
    def __init__(self, path:str, header:list):
        self.data = []
        self.path = path
        self.header = header
    
    def save(self):
        txt = ", ".join(self.header) + "\n"
        f = open(self.path, "w")
        for val in self.data:
            v_str = [str(i) if type(i) is not str else i for i in val]
            txt+= ", ".join(v_str) + "\n"
        f.write(txt)
        f.close()

    def update(self, values:list):
        self.data.append(values)
