import os

if __name__ == '__main__':
    path = 'models/detic'
    if not os.path.exists(path):
        os.makedirs(path)