import json
import glob

files = glob.glob('TalkPages_*.txt')

for fname in files:
    d = []
    with open(fname) as f:
        print(f'Opened {fname}')
        for line in f:
            cols = line.strip('\n').split(' ::: ')
            d.append([int(cols[0]), cols[1], cols[2].split()])

    print(f'Closed {fname}')
    JSONfname = fname.replace('.txt', '.json')
    json.dump(d, open(JSONfname, 'w'), ensure_ascii=False)
    print(f'Successfully written {JSONfname}')
    print('Read it in with json.load(open(\'FNAME\'))')

