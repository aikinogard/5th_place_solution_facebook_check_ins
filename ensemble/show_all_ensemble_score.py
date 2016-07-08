import glob

flist = glob.glob("*.txt")
for txt in flist:
    with open(txt, "r") as f:
        print txt, f.read()
