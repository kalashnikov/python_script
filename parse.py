import re
import sys
import pandas as pd

class Ops:
    # match op1 to create object for each calibre operations
    def __init__(self):
        self.name = ""
        self.sub_type = "FullOp"
        self.op_type = ""
        self.optype = ""
        self.typ = ""
        self.cfg = 0
        self.hgc = 0
        self.fgc = 0
        self.hec = 0
        self.fec = 0
        self.igc = 0
        self.vhc = 0
        self.vpc = 0
        self.cpu_time = 0.0
        self.real_time = 0.0
        self.lvheap = ""
        self.shared = ""
        self.elapsed_time = 0
        self.scale_factor = 0.0
        self.lvheap_used, self.lvheap_allocated, self.lvheap_max = 0, 0, 0
        self.shared_used, self.shared_allocated = 0, 0

    def init_op1(self, name, optype, cfg, typ, hgc, fgc, hec, fec, igc, vhc, vpc):
        self.name = name
        self.optype = optype
        self.typ = typ
        self.cfg = cfg
        self.hgc = hgc
        self.fgc = fgc
        self.hec = hec
        self.fec = fec
        self.igc = igc
        self.vhc = vhc
        self.vpc = vpc

        # Reset
        self.cpu_time = 0.0
        self.real_time = 0.0
        self.lvheap = ""
        self.shared = ""
        self.elapsed_time = 0

        # calculated number
        self.scale_factor = 0.0

    # For sub_op1
    def init_sub_op1(self, cpu_time, real_time, lvheap, shared, name):
        self.sub_type = name
        self.cpu_time = float(cpu_time)
        self.real_time = float(real_time)
        self.lvheap = lvheap
        self.shared = shared
        self.scale_factor = self.cpu_time / self.real_time if self.real_time and self.real_time is not None else 0
        self.lvheap_used, self.lvheap_allocated, self.lvheap_max = self.lvheap.split('/')
        self.shared_used, self.shared_allocated = self.shared.split('/')

    # For sub_op2
    def init_sub_op2(self, cpu_time, real_time, name):
        self.sub_typ = name
        self.cpu_time = float(cpu_time)
        self.real_time = float(real_time)
        self.scale_factor = self.cpu_time / self.real_time if self.real_time and self.real_time is not None else 0

    def add_main_op(self, name, optype, cfg, typ, hgc, fgc, hec, fec, igc, vhc, vpc):
        self.name = name + " - " + self.sub_type
        self.optype = optype
        self.typ = typ
        self.cfg = cfg
        self.hgc = hgc
        self.fgc = fgc
        self.hec = hec
        self.fec = fec
        self.igc = igc
        self.vhc = vhc
        self.vpc = vpc

    def add_op2(self, cpu_time, real_time, lvheap, elapsed_time):
        self.cpu_time = float(cpu_time)
        self.real_time = float(real_time)
        self.lvheap = lvheap
        self.elapsed_time = float(elapsed_time)
        self.scale_factor = self.cpu_time / self.real_time if self.real_time and self.real_time is not None else 0
        self.lvheap_used, self.lvheap_allocated, self.lvheap_max = self.lvheap.split('/')

    def add_op3(self, cpu_time, real_time, lvheap, shared, elapsed_time):
        self.cpu_time = float(cpu_time)
        self.real_time = float(real_time)
        self.lvheap = lvheap
        self.shared = shared
        self.elapsed_time = float(elapsed_time)
        self.scale_factor = self.cpu_time / self.real_time if self.real_time and self.real_time is not None else 0
        self.lvheap_used, self.lvheap_allocated, self.lvheap_max = self.lvheap.split('/')
        self.shared_used, self.shared_allocated = self.shared.split('/')

    def to_dict(self):
        return {
            'name': self.name,
            'optype': self.optype,
            'op_group': self.op_group,
            'cfg': self.cfg,
            'typ': self.typ,
            'hgc': self.hgc,
            'fgc': self.fgc,
            'hec': self.hec,
            'fec': self.fec,
            'igc': self.igc,
            'vhc': self.vhc,
            'vpc': self.vpc,
            'cpu_time': self.cpu_time,
            'real_time': self.real_time,
            'scale_factor': self.scale_factor,
            # 'lvheap': self.lvheap, 
            'lvheap_used': int(self.lvheap_used),
            'lvheap_allocated': int(self.lvheap_allocated),
            'lvheap_max': int(self.lvheap_max),
            # 'shared': self.shared,
            'shared_used': int(self.shared_used),
            'shared_allocated': int(self.shared_allocated),
            'elapsed_time': self.elapsed_time,
            'sub_type': self.sub_type
        }

    def __str__(self):
        return '### {} ### | Type({}, {}, {}, {}), Geometry#({}, {}), Edge#({}, {}), igc:{}, VHC:{}, VPC:{}\n\
                | CPU TIME: {}, REAL TIME: {}, Scale: {:.2f}, LVHEAP: {}, SHARED: {}, ELAPSED TIME: {}'.format(
            self.name, self.optype, self.cfg, self.typ, self.sub_type,
            self.hgc, self.fgc, self.hec, self.fec, self.igc, self.vhc, self.vpc,
            self.cpu_time, self.real_time, self.scale_factor, self.lvheap, self.shared, self.elapsed_time)


def parse_log(input_file, all_ops):
    sub_ops  = []
    last_ops = []

    # fSwissCheese (HIER TYP=1 CFG=1 HGC=322629 FGC=322629 HEC=1290516 FEC=1290516 IGC=585 VHC=F VPC=F)
    op1 = re.compile('(\S+) \((\S+) TYP=(\d+) CFG=(\d+) HGC=(\d+) FGC=(\d+) HEC=(\d+) FEC=(\d+) IGC=(\d+) VHC=(\w) VPC=(\w)\)')

    # CPU TIME = 2  REAL TIME = 2  LVHEAP = 3/5/5  OPS COMPLETE = 8 OF 16  ELAPSED TIME = 7
    op2 = re.compile('CPU TIME = (\d+)  REAL TIME = (\d+)  LVHEAP = (\S+)  OPS COMPLETE = \d+ OF \d+  ELAPSED TIME = (\d+)')

    # CPU TIME = 370  REAL TIME = 292  LVHEAP = 5/21/21  SHARED = 1/32  OPS COMPLETE = 15 OF 16  ELAPSED TIME = 299
    op3 = re.compile('CPU TIME = (\d+)  REAL TIME = (\d+)  LVHEAP = (\S+)  SHARED = (\S+)  OPS COMPLETE = \d+ OF \d+  ELAPSED TIME = (\d+)')

    # CPU TIME = 0  REAL TIME = 0, LVHEAP = 7059/7061/7061  SHARED = 0/0 - INIT_OPT 
    sub_op1 = re.compile('WARNING:\s+CPU TIME = (\d+)  REAL TIME = (\d+), LVHEAP = (\S+)  SHARED = (\S+) - (\S+)')

    # CPU TIME = 14  REAL TIME = 8 - PUSH_OUT
    sub_op2 = re.compile('CPU TIME = (\d+)  REAL TIME = (\d+).?\s? - (\S+)')

    with open(input_file, "r") as f:
        for line in f:
            if "CPU TIME" not in line and "FEC" not in line:
                continue;

            l = line.strip()

            if "WARNING" in line and "REAL TIME = 0" not in line:
                sub_op1_result = sub_op1.match(l)
                if sub_op1_result:
                    op = Ops()
                    op.init_sub_op1(*sub_op1_result.groups())
                    sub_ops.append(op)
                    continue

                sub_op2_result = sub_op2.match(l)
                if sub_op2_result:
                    op = Ops()
                    op.init_sub_op2(*sub_op2_result.groups())
                    sub_ops.append(op)
                    continue

            result1 = op1.match(l)

            # Operation information 
            if result1:
                op = Ops()
                op.init_op1(*result1.groups())
                op.op_group = last_ops[0].name if len(last_ops)!=0 else op.name
                last_ops.append(op)

                if len(sub_ops)!=0:
                    for so in sub_ops:
                       so.add_main_op(*result1.groups())
                       so.op_group = op.name
                       all_ops.append(so)

                    sub_ops.clear()

            else:
                if len(last_ops)==0:
                    last_ops.clear()
                    continue

                result2 = op2.match(l)
                result3 = op3.match(l)

                if result2:     # Operation statistic 
                    if int(result2.group(2))!=0:
                        for ops in last_ops:
                            ops.add_op2(*result2.groups())
                            all_ops.append(ops)

                    last_ops.clear()

                elif result3:   # Operation statistic 
                    if int(result3.group(2))!=0:
                        for ops in last_ops:
                            ops.add_op3(*result3.groups())
                            all_ops.append(ops)

                    last_ops.clear()

    all_ops.sort(key=lambda x: x.real_time)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Please provide input file.")
        exit()

    input_file = sys.argv[1]

    all_ops = []
    parse_log(input_file, all_ops)

    # print("######### Print Operations #########")
    # for ops in all_ops:
    #     print(ops)

    # https://stackoverflow.com/questions/34997174/how-to-convert-list-of-model-objects-to-pandas-dataframe
    df = pd.DataFrame.from_records([op.to_dict() for op in all_ops])

    df.to_csv('{}.csv'.format(input_file))
