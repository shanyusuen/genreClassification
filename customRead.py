def read(name):
    f = open(name, 'r')
    data = False
    x, y = list(), list()
    for line in f.readlines():
        if line.strip() == "@data" or line.strip() == "@DATA":
            data = True
            continue
        if data:
            x.append([float(i) for i in line.strip().split(",")[:-1]])
            y.append(line.strip().split(",")[-1].strip())
    return (x, y)


if __name__ == "__main__":
    test_x, test_y = read("out/data-partition.train.arff")
    print(test_x)
    print(test_y)
