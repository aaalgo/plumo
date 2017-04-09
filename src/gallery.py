import os
from jinja2 import Environment, FileSystemLoader

TMPL_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                './templates')
env = Environment(loader=FileSystemLoader(searchpath=TMPL_DIR))
tmpl = env.get_template('gallery.html')

class Gallery:
    def __init__ (self, path, cols = 1, header = None, ext = '.png'):
        self.next_id = 0
        self.path = path
        self.cols = cols
        self.header = header
        self.ext = ext
        self.images = []
        try:
            os.makedirs(path)
        except:
            pass
        pass

    def text (self, tt, br = False):
        self.images.append({
            'text': tt})
        if br:
            for i in range(1, self.cols):
                self.images.append({
                    'text': ''})
        pass

    def next (self, text=None, link=None):
        path = '%03d%s' % (self.next_id, self.ext)
        self.images.append({
            'image': path,
            'text': text,
            'link': link})
        self.next_id += 1
        return os.path.join(self.path, path)

    def flush (self, temp=None, extra={}):
        if temp is None:
            temp = tmpl
        else:
            temp = env.get_template(temp)
        with open(os.path.join(self.path, 'index.html'), 'w') as f:
            images = [self.images[i:i+self.cols] for i in range(0, len(self.images), self.cols)]
            f.write(temp.render(images=images, header=self.header, extra=extra))
            pass
        pass

