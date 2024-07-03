import os.path as osp
class LogCSV():
    def __init__(self, path:str, header:list, mode:str='w'):
        self.data = []
        self.path = path
        self.header = header
        self.mode = mode
        if mode == "a" and not osp.exists(path):
            self.update_a(header)

    
    def save(self):
        txt = ", ".join(self.header) + "\n"
        f = open(self.path, self.mode)
        for val in self.data:
            v_str = [str(i) if type(i) is not str else i for i in val]
            txt+= ", ".join(v_str) + "\n"
        f.write(txt)
        f.close()

    def update(self, values:list):
        self.data.append(values)

    def update_a(self, value):
        f = open(self.path, self.mode)
        v_str = [str(i) if type(i) is not str else i for i in value]
        txt= ", ".join(v_str) + "\n"
        f.write(txt)
        f.close()