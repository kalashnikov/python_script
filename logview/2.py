import re
import glob 
import pandas as pd
import numpy as np

from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox
from bokeh.models import ColumnDataSource, HoverTool, BoxZoomTool, ResetTool, Div
from bokeh.models.widgets import Slider, Select, TextInput
from bokeh.io import curdoc


#############  Class for operations  ############ 
class Ops:
    # match op1 to create object for each calibre operations
    def __init__(self):
        self.name = ""
        self.sub_type = "FullOp"
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
            'lvheap_used': int(self.lvheap_used),
            'lvheap_allocated': int(self.lvheap_allocated),
            'lvheap_max': int(self.lvheap_max),
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

#############  Function to parse log  ############ 
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
    sub_op2 = re.compile('WARNING:\s+CPU TIME = (\d+)  REAL TIME = (\d+).?\s? - (\S+)')

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
                last_ops.append(op)

                if len(sub_ops) != 0:
                    for so in sub_ops:
                       so.add_main_op(*result1.groups())
                       all_ops.append(so)

                    sub_ops.clear()

            else:
                if len(last_ops)==0:
                    last_ops.clear()
                    continue

                result2 = op2.match(l)
                result3 = op3.match(l)

                if result2:     # Operation statistic 
                    if int(result2.group(2)) != 0:
                        for ops in last_ops:
                            ops.add_op2(*result2.groups())
                            all_ops.append(ops)

                    last_ops.clear()

                elif result3:   # Operation statistic 
                    if int(result3.group(2)) != 0:
                        for ops in last_ops:
                            ops.add_op3(*result3.groups())
                            all_ops.append(ops)

                    last_ops.clear()

    all_ops.sort(key=lambda x: x.real_time)


#############  Parse log and prepare DataFrame  ############ 
def prepare_operations(file_name):
    all_ops = []
    parse_log(file_name, all_ops)
    operations = pd.DataFrame.from_records([op.to_dict() for op in all_ops])
    
    # Highlight low scale factor
    operations["color"] = np.where(operations["scale_factor"] < 2, "gold", "grey")
    operations["color"] = np.where(operations["scale_factor"] > 6, "greenyellow", operations["color"])
    operations["alpha"] = np.where(operations["scale_factor"] < 2, 0.9, 0.25)
    return operations


########################################################################## 
############################ Bokeh Server App ############################
########################################################################## 


#############  Start data processing  ############ 
current_file = "run_latest.log"
operations = prepare_operations(current_file)


#############  Start Bokeh Configuration  ############ 
axis_map = {
    "CPU time": "cpu_time",
    "Real time": "real_time",
    "LVHEAP used": "lvheap_used",
    "LVHEAP allocated": "lvheap_allocated",
    "Scale factor": "scale_factor",
    "Shared memory used": "shared_used",
}

desc = Div(text=open("temp-plot.html").read(), width=800)

########### Create Input controls ##########
cpu_time = Slider(title="CPU Time", value=0, start=0,
        end=operations.cpu_time.max(), step=10)
real_time = Slider(title="Real time", value=0, start=0,
        end=operations.real_time.max(), step=10)

lvheap_used = Slider(title="Used LVHEAP", value=0, start=0,
        end=operations.lvheap_used.max(), step=100)

scale_factor = Slider(title="Scale factor", value=0, start=0,
        end=operations.scale_factor.max(), step=1)

shared_used = Slider(title="Shared memory used", value=0, start=0,
        end=operations.shared_used.max(), step=1)

sub_type = Select(title="SubType", value="All",
        options= ['ALL'] + list(operations.sub_type.unique()))

file_list = Select(title="File List", value=current_file,
        options=[f for f in glob.iglob('**/*.log', recursive=True) if f.endswith('.log')])

# Filter by Operation name and SubType 
sub_type_name = TextInput(title="SubType")
op_name = TextInput(title="Operation name")

x_axis = Select(title="X Axis", options=sorted(axis_map.keys()), value="Real time")
y_axis = Select(title="Y Axis", options=sorted(axis_map.keys()), value="LVHEAP used")


############ Create Column Data Source that will be used by the plot ############  
source = ColumnDataSource(data=dict(x=[], y=[], name=[], color=[],
    alpha=[], sub_type=[], cpu_time=[], real_time=[], scale_factor=[],
    lvheap_used=[], lvheap_allocated=[], shared_used=[],
    fec=[], fgc=[], hec=[], hgc=[]))

hover = HoverTool(tooltips=[
    ("Operation Name", "@name"),
    ("SubType", "@sub_type"),
    ("CPU time / Real time", "@cpu_time / @real_time"),
    ("Scale factor", "@scale_factor"),
    ("LVHEAP: used, allocated", "@lvheap_used, @lvheap_allocated"),
    ("Shared memory used", "@shared_used"),
    ("FLAT: #edge, #geometry", "@fec, @fgc"),
    ("HIER: #edge, #geometry", "@hec, @hgc")
])

p = figure(plot_height=900, plot_width=900, title="", toolbar_location=None,
        tools=[hover, BoxZoomTool(), ResetTool()])
p.left[0].formatter.use_scientific = False
p.circle(x="x", y="y", source=source, size=10, color="color", line_color=None, fill_alpha="alpha")


#############  Control callback function ############ 
def select_operations():
    global operations
    global current_file
    
    sub_type_val = sub_type.value
    sub_type_name_val = sub_type_name.value
    op_name_val = op_name.value.strip()
    file_name = file_list.value.strip()

    # Parse log and replace data
    if (file_name != current_file):
        operations = prepare_operations(file_name)
        current_file = file_name
    
    selected = operations[
        (operations.cpu_time >= cpu_time.value) &
        (operations.real_time >= real_time.value) &
        (operations.lvheap_used >= lvheap_used.value) &
        (operations.scale_factor >= scale_factor.value) &
        (operations.shared_used >= shared_used.value)
    ]
    if (sub_type_val != "All"):
        selected = selected[selected.sub_type.str.contains(sub_type_val) == True]
    if (op_name_val != ""):
        selected = selected[selected.name.str.lower().str.contains(op_name_val.lower()) == True]
    if (sub_type_name_val != ""):
        selected = selected[selected.sub_type.str.lower().str.contains(sub_type_name_val.lower()) == True]
    return selected


def update():
    df = select_operations()
    x_name = axis_map[x_axis.value]
    y_name = axis_map[y_axis.value]

    p.xaxis.axis_label = x_axis.value
    p.yaxis.axis_label = y_axis.value
    p.title.text = "%d operations selected" % len(df)
    source.data = dict(
        x=df[x_name],
        y=df[y_name],
        name=df['name'],
        sub_type=df['sub_type'],
        color=df['color'],
        alpha=df['alpha'],
        cpu_time=df['cpu_time'],
        real_time=df['real_time'],
        scale_factor=df['scale_factor'],
        lvheap_used=df['lvheap_used'],
        lvheap_allocated=df['lvheap_allocated'],
        shared_used=df['shared_used'],
        fec=df['fec'],
        fgc=df['fgc'],
        hec=df['hec'],
        hgc=df['hgc'],
    )


#############  Control panel configuration  ############ 
controls = [file_list, cpu_time, real_time, lvheap_used, scale_factor, shared_used, sub_type, sub_type_name, op_name, x_axis, y_axis]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())


#############  Layout and finalization ############ 
# sizing_mode = 'fixed'  # 'scale_width' also looks nice with this example
sizing_mode = 'scale_width'

inputs = widgetbox(*controls, sizing_mode=sizing_mode)
l = layout([
    [desc],
    [inputs, p],
], sizing_mode=sizing_mode)

update()  # initial load of the data

curdoc().add_root(l)
curdoc().title = "Calibre transcript analystic"

