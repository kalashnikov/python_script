import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class HDB:
    def __init__(self):
        self.lvheap_used = [0]
        self.lvheap_allocated = [0]
        self.lvheap_max = [0]

    def add_heap(self, heap_str):
        l1, l2, l3 = heap_str.split('/')
        # print("DEBUG: {}, {}, {} | {}".format(l1, l2, l3, heap_str))
        self.lvheap_used.append(int(l1))
        self.lvheap_allocated.append(int(l2))
        self.lvheap_max.append(int(l3))
        return heap_str

    def add_dummy(self):
        self.lvheap_used.append(self.lvheap_used[-1])
        self.lvheap_allocated.append(self.lvheap_allocated[-1])
        self.lvheap_max.append(self.lvheap_max[-1])


def parse_log(input_file):

    lv_str = re.compile('LVHEAP = (\S+)')

    # fSwissCheese (HIER TYP=1 CFG=1 HGC=322629 FGC=322629 HEC=1290516 FEC=1290516 IGC=585 VHC=F VPC=F)
    op1 = re.compile('(\S+) \((\S+) TYP=(\d+) CFG=(\d+) HGC=(\d+) FGC=(\d+) HEC=(\d+) FEC=(\d+) IGC=(\d+) VHC=(\w) VPC=(\w)\)')

    last_op = ''
    ops = ['Init']

    # HDB 0
    hdbs = []
    h = HDB()
    hdbs.append(h)
    hdb_strings = ["HDB 0"]

    start = False
    with open(input_file, "r") as f:
        for line in f:
            # //  Initializing MT on pseudo HDB 1
            if "Initializing MT on pseudo HDB" in line:
                hdb_strings.append("HDB " + str(len(hdbs)))
                h = HDB()
                hdbs.append(h)
                continue

            # Skip lines until operation transcript start 
            if not start and 'EXECUTIVE MODULE' not in line:
                continue
        
            start = True

            l = line.strip()
            result = op1.match(l)
            if result:
                # print("Add op: {}".format(result.groups()[0]))
                last_op=result.groups()[0]
                continue

            # Not LVHEAP summary line
            if "LVHEAP = " not in line or "Operation COMPLETED" not in line:
                continue

            # Get LVHEAP string
            array = l.split(' ')
            rev_idx = -1 if "RSS" not in line else -5
            hdb_str = array[rev_idx]

            hdb_idx = 0
            if " HDB " in line:
                for idx, s in enumerate(hdb_strings):
                    if s in line:
                        hdb_idx = idx
                        break
            # if "HDB x" not in line, it's for HDB 0 

            # print("DEBUG: {} to {} | {}".format(l, hdb_strings[hdb_idx], line.strip()))

            for idx, hdb in enumerate(hdbs):
                if idx == hdb_idx:
                    hdb.add_heap(hdb_str)
                else:
                    hdb.add_dummy()

            ops.append(last_op)

    lvheap_used = {}
    lvheap_allocated = {}
    lvheap_max  = {}

    lvheap_used['op'] = ops
    lvheap_allocated['op'] = ops
    lvheap_max['op'] = ops

    for idx, s in enumerate(hdb_strings):
        lvheap_used[s] = hdbs[idx].lvheap_used
        lvheap_allocated[s] = hdbs[idx].lvheap_allocated
        lvheap_max[s] = hdbs[idx].lvheap_max

    return pd.DataFrame(lvheap_used), pd.DataFrame(lvheap_allocated), pd.DataFrame(lvheap_max)


def calculate_sum_dif(df):
    df['Sum'] = df['HDB 0'] + df['HDB 1'] + df['HDB 2'] + df['HDB 3'] + df['HDB 4']
    df['Dif'] = df['Sum'].diff(1) / df['Sum'].shift(1)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    # df = df.set_index('op')
    return df

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Please provide input file.")
        exit()

    input_file = sys.argv[1]

    # Parse log and get DataFrame format
    d1, d2, d3 = parse_log(input_file)
    
    d1 = calculate_sum_dif(d1)
    d2 = calculate_sum_dif(d2)
    d3 = calculate_sum_dif(d3)

    # LVHEAP used
    d1.loc[d1.Dif > 0.1, ['HDB 0', 'HDB 1', 'HDB 2', 'HDB 3', 'HDB 4', 'Sum']].plot()
    plt.title('LVHEAP Used')
    plt.xticks(rotation='vertical')
    plt.xlabel('Operations')
    plt.ylabel('LVHEAP')
    plt.savefig('lvheap_used.png', bbox_inches='tight')
    d1.to_csv('lvheap_used.csv')

    # LVHEAP allocated 
    d2.loc[d2.Dif > 0.1, ['HDB 0', 'HDB 1', 'HDB 2', 'HDB 3', 'HDB 4', 'Sum']].plot()
    plt.title('LVHEAP Allocated')
    plt.xticks(rotation='vertical')
    plt.xlabel('Operations')
    plt.ylabel('LVHEAP')
    plt.savefig('lvheap_allocated.png', bbox_inches='tight')
    d2.to_csv('lvheap_allocated.csv')

    # LVHEAP maximum 
    d3.loc[d3.Dif > 0.1, ['HDB 0', 'HDB 1', 'HDB 2', 'HDB 3', 'HDB 4', 'Sum']].plot()
    plt.title('LVHEAP Maximum')
    plt.xticks(rotation='vertical')
    plt.xlabel('Operations')
    plt.ylabel('LVHEAP')
    plt.savefig('lvheap_max.png', bbox_inches='tight')
    d3.to_csv('lvheap_max.csv')

