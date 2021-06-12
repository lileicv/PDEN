
def log(txt, path):
    print(txt)
    with open(path, 'a') as f:
        f.write(txt + '\n')
        f.flush()
        f.close()


