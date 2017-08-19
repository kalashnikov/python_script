import re
import glob
import pandas as pd
import numpy as np
from os.path import dirname, join

from bokeh.plotting import figure
from bokeh.palettes import Viridis11
from bokeh.layouts import layout, widgetbox, row, column
from bokeh.models import ColumnDataSource, HoverTool, BoxZoomTool, ResetTool, Div
from bokeh.models.widgets import Slider, Select, TextInput, DataTable, TableColumn, NumberFormatter
from bokeh.io import curdoc

from plotly.offline import download_plotlyjs, offline
from plotly.graph_objs import *
import plotly


#############  Class for HDB memory monitor ############ 
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


#############  Class for operations  ############ 
class Ops:
    # match op1 to create object for each calibre operations
    def __init__(self):
        self.name = ""
        self.sub_type = "FullOp"
        self.op_group = ""
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

    # For sub_op1
    def init_sub_op1(self, cpu_time, real_time, lvheap, shared, name):
        self.sub_type = name
        self.cpu_time = float(cpu_time)
        self.real_time = float(real_time)
        self.lvheap = lvheap
        self.shared = shared
        self.lvheap_used, self.lvheap_allocated, self.lvheap_max = self.lvheap.split('/')
        self.shared_used, self.shared_allocated = self.shared.split('/')

    # For sub_op2
    def init_sub_op2(self, cpu_time, real_time, name):
        self.sub_typ = name
        self.cpu_time = float(cpu_time)
        self.real_time = float(real_time)

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
        self.lvheap_used, self.lvheap_allocated, self.lvheap_max = self.lvheap.split('/')

    def add_op3(self, cpu_time, real_time, lvheap, shared, elapsed_time):
        self.cpu_time = float(cpu_time)
        self.real_time = float(real_time)
        self.lvheap = lvheap
        self.shared = shared
        self.elapsed_time = float(elapsed_time)
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
            self.cpu_time, self.real_time, 0, self.lvheap, self.shared, self.elapsed_time)


def parse_log(input_file, chart_width):
    '''
    Parse Calibre Transcript
    '''

    # HDB 0
    hdbs = []
    h = HDB()
    hdbs.append(h)
    hdb_strings = ["HDB 0"]

    all_ops  = []
    sub_ops  = []
    last_op  = ''
    last_ops = []
    hdb_ops  = ['Init']

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

    ################ START OF LOG PARSE #################
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

            if not start:
                start = True

            l = line.strip()

            if 'Operation COMPLETED on HDB' in l:
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
                # print("DEBUG: {} to {}".format(l, hdb_strings[hdb_idx]))

                for idx, hdb in enumerate(hdbs):
                    if idx == hdb_idx:
                        hdb.add_heap(hdb_str)
                    else:
                        hdb.add_dummy()

                hdb_ops.append(last_op)
                continue


            if "CPU TIME" not in line and "FEC" not in line:
                continue

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
                last_op = op.op_group = last_ops[0].name if len(last_ops)!=0 else op.name
                last_ops.append(op)

                if len(sub_ops) != 0:
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

    ################ END OF LOG PARSE #################

    # Post operations for HDB monitoring 
    lvheap_used = {}
    lvheap_allocated = {}
    lvheap_max  = {}

    lvheap_used['op'] = hdb_ops
    lvheap_allocated['op'] = hdb_ops
    lvheap_max['op'] = hdb_ops

    for idx, s in enumerate(hdb_strings):
        lvheap_used[s] = hdbs[idx].lvheap_used
        lvheap_allocated[s] = hdbs[idx].lvheap_allocated
        lvheap_max[s] = hdbs[idx].lvheap_max

    hdb1_fig, hdb1_table = gen_lvheap_chart(pd.DataFrame(lvheap_used), "LVHEAP Used", chart_width)
    hdb2_fig, hdb2_table = gen_lvheap_chart(pd.DataFrame(lvheap_allocated), "LVHEAP Allocated", chart_width)
    hdb3_fig, hdb3_table = gen_lvheap_chart(pd.DataFrame(lvheap_max), "LVHEAP Maximum", chart_width)

    operations = pd.DataFrame.from_records([op.to_dict() for op in all_ops])

    return operations, hdb1_fig, hdb2_fig, hdb3_fig, hdb1_table, hdb2_table, hdb3_table


def gen_lvheap_chart(df, fig_title, chart_width):
    '''
    Calculate 'SUM' & 'Dif' columns for HDB DataFrames
    Find all 10% difference operations
    Generate line chart
    '''

    # Pre-process data before generating chart
    df['Sum'] = df['HDB 0'] + df['HDB 1'] + df['HDB 2'] + df['HDB 3'] + df['HDB 4']
    df['Dif'] = df['Sum'].diff(1) / df['Sum'].shift(1)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.loc[df.Dif > 0.1, ['HDB 0', 'HDB 1', 'HDB 2', 'HDB 3', 'HDB 4', 'Sum', 'op']]
    label_arrays = df['op'].values # Backup label string array

    x_axis_array = np.arange(0, len(df.index))

    # Init Figure object
    fig = figure(plot_width=chart_width, title=fig_title, x_axis_label='Operations', y_axis_label='LVHEAP')

    # Adding lines
    num_lines = len(df.columns)
    palettes = Viridis11[0:num_lines]
    for idx, l in enumerate(df.columns.values):
        if l=='op':
            continue
        fig.line(x=x_axis_array, y=df[l].values, line_color=palettes[idx], line_width=3, legend=l)

    # Adjust x-axis ticks label
    # fig.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    fig.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks

    # Use integer as x-axis ticker
    fig.xaxis[0].ticker = x_axis_array

    # Override integer ticker using df.op column
    fig.xaxis[0].major_label_overrides = dict(zip(map(str, x_axis_array), df.op.values))

    # Rotate x-axis ticker label
    fig.xaxis[0].major_label_orientation = 'vertical'

    # Y-axis use normal number
    fig.left[0].formatter.use_scientific = False

    fig.legend.location = "top_left"

    return fig, gen_hdb_data_table(df, chart_width)

def gen_hdb_data_table(df, chart_width):
    '''
    Generate data into DataTable format
    '''

    data = dict(df[['op', 'HDB 0', 'HDB 1', 'HDB 2', 'HDB 3', 'HDB 4', 'Sum']])
    source = ColumnDataSource(data)

    columns = [
            TableColumn(field='op', title='Operation'),
            TableColumn(field='HDB 0', title='HDB 0'),
            TableColumn(field='HDB 1', title='HDB 1'),
            TableColumn(field='HDB 2', title='HDB 2'),
            TableColumn(field='HDB 3', title='HDB 3'),
            TableColumn(field='HDB 4', title='HDB 4'),
            TableColumn(field='Sum', title='Sum'),
    ]

    return DataTable(source=source, columns=columns, width=chart_width)


def prepare_operations(file_name, chart_width):
    '''
    Parse log and prepare DataFrame
    '''

    # Parse log
    operations, hdb1_fig, hdb2_fig, hdb3_fig, hdb1_table, hdb2_table, hdb3_table = parse_log(file_name, chart_width)

    # Calculate runtime ratio
    max_time = operations.elapsed_time.max()
    operations["runtime_ratio"] = 100 * operations["real_time"] / max_time

    # Calculate scale_factor
    operations["scale_factor"] = np.where(operations["real_time"]!=0, operations["cpu_time"] / operations["real_time"], 0)

    # Highlight low scale factor
    operations["color"] = np.where(operations["scale_factor"] < 2, "gold", "grey")
    operations["color"] = np.where(operations["scale_factor"] > 6, "greenyellow", operations["color"])
    operations["color"] = np.where(operations["runtime_ratio"] > 50, "red", operations["color"])
    operations["alpha"] = np.where(operations["scale_factor"] < 2, 0.9, 0.25)

    return operations, hdb1_fig, hdb2_fig, hdb3_fig, hdb1_table, hdb2_table, hdb3_table


def gen_plot_page(df, chart_width):
    '''
    Generate Pareto chart using Plotly
    '''

    # Uniquify by real_time and op_group page 
    df = df.loc[df.sub_type=='FullOp', ['real_time','op_group']]
    df = df.drop_duplicates()
    df = df.sort_values(by="real_time",ascending=False)

    # Prepare addition columns for Pareto Chart 
    max_time = operations.elapsed_time.max()
    df['cumulative_sum'] = df.real_time.cumsum()
    df['cumulative_perc'] = 100 * df.cumulative_sum/ max_time
    df['demarcation'] = 80

    # Filter out until 80% 
    # df = df.query('cumulative_perc < 200')

    # Show Top 20 rows 
    df = df.head(20)

    # Prepare plotly data
    trace1 = Bar(
        x=df.op_group,
        y=df.real_time,
        name='Real Time',
        marker=dict(
             color='rgb(34,163,192)'
                    )
    )
    trace2 = Scatter(
        x=df.op_group,
        y=df.cumulative_perc,
        name='Cumulative Percentage',
        yaxis='y2',
        line=dict(
            color='rgb(243,158,115)',
            width=2.4
           )
    )
    trace3 = Scatter(
        x=df.op_group,
        y=df.demarcation,
        name='80%',
        yaxis='y2',
        line=dict(
            color='rgba(128,128,128,.45)',
            dash = 'dash',
            width=1.5
           )
    )
    data = [trace1, trace2, trace3]
    layout = Layout(
        title='Smartfill run time - Pareto Chart',
        titlefont=dict(
            color='',
            family='',
            size=0
        ),
        font=Font(
            color='rgb(128,128,128)',
            family='Balto, sans-serif',
            size=12
        ),
        width=chart_width,
        height=623,
        paper_bgcolor='rgb(240, 240, 240)',
        plot_bgcolor='rgb(240, 240, 240)',
        hovermode='compare',
        margin=dict(b=250,l=60,r=60,t=65),
        showlegend=True,
           legend=dict(
              x=.83,
              y=1.3,
              font=dict(
                family='Balto, sans-serif',
                size=12,
                color='rgba(128,128,128,.75)'
            ),
        ),
        annotations=[ dict(
                      text="Cumulative Percentage",
                      showarrow=False,
                      xref="paper", yref="paper",
                      textangle=90,
                      x=1.050, y=.75,
                      font=dict(
                      family='Balto, sans-serif',
                      size=14,
                      color='rgba(243,158,115,.9)'
                ),)],
        xaxis=dict(
          tickangle=-90,
          autorange=True,
          autotick=True
        ),
        yaxis=dict(
          title='Real Time',
          autorange=True,
          tickfont=dict(
                color='rgba(34,163,192,.75)'
            ),
          autotick=True,
          titlefont=dict(
              family='Balto, sans-serif',
              size=14,
              color='rgba(34,163,192,.75)')
        ),
        yaxis2=dict(
            autorange=True,
            autotick=True,
            tickfont=dict(
                color='rgba(243,158,115,.9)'
            ),
            # tickvals = [0,20,40,60,80,100],
            overlaying='y',
            side='right'
        )
    )

    fig = dict(data=data, layout=layout)

    # Gen plotly page 
    offline.plot(fig, auto_open=False, filename="pareto_chart.html")


def gen_data_table(df, chart_width):
    '''
    Generate data into DataTable format
    '''

    # Uniquify by real_time and op_group page 
    df = df.loc[df.sub_type=='FullOp', ['op_group', 'sub_type', 'cpu_time', 'real_time', 'runtime_ratio', 'scale_factor',
        'lvheap_used', 'lvheap_allocated', 'shared_used']]
    df = df.drop_duplicates()

    # Show Top 20 rows 
    df = df.sort_values(by="real_time", ascending=False)
    df = df.head(20)

    data = dict(df[['op_group', 'sub_type', 'cpu_time', 'real_time', 'runtime_ratio', 'scale_factor',
        'lvheap_used', 'lvheap_allocated', 'shared_used']])#, 'fec', 'fgc', 'hec', 'hgc']])
    source = ColumnDataSource(data)

    columns = [
            TableColumn(field='op_group', title='Name'),
            TableColumn(field='sub_type', title='Sub-Op Type'),
            TableColumn(field='cpu_time', title='CPU time'),
            TableColumn(field='real_time', title='Real time'),
            TableColumn(field='runtime_ratio', title='Runtime ratio', formatter=NumberFormatter(format="0.00")),
            TableColumn(field='scale_factor', title='Scale factor', formatter=NumberFormatter(format="0.00")),
            TableColumn(field='lvheap_used', title='LVHEAP used'),
            TableColumn(field='lvheap_allocated', title='LVHEAP allocated'),
            TableColumn(field='shared_used', title='Shared used'),
    ]

    return DataTable(source=source, columns=columns, width=chart_width)


########################################################################## 
############################ Bokeh Server App ############################
########################################################################## 

# Layout General Setting 
widget_width = 300
sizing_mode  = 'fixed'  # 'scale_width' also looks nice with this example
chart_width  = 1150
chart_height = 800

#############  Start data processing  ############ 
current_file = "run_latest.log"
operations, hdb1_fig, hdb2_fig, hdb3_fig, hdb1_table, hdb2_table, hdb3_table = prepare_operations(current_file, chart_width)


#############  Start Bokeh Configuration  ############ 
axis_map = {
    "CPU time": "cpu_time",
    "Real time": "real_time",
    "LVHEAP used": "lvheap_used",
    "LVHEAP allocated": "lvheap_allocated",
    "Scale factor": "scale_factor",
    "Shared memory used": "shared_used",
}

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
    alpha=[], sub_type=[], cpu_time=[], real_time=[], runtime_ratio=[],
    scale_factor=[], lvheap_used=[], lvheap_allocated=[], shared_used=[],
    fec=[], fgc=[], hec=[], hgc=[]))

hover = HoverTool(tooltips=[
    ("Operation Name (SubType)", "@name (@sub_type)"),
    ("CPU time / Real time", "@cpu_time / @real_time"),
    ("Scale factor / Runtime Ratio", "@scale_factor / @runtime_ratio%"),
    ("LVHEAP: used, allocated", "@lvheap_used, @lvheap_allocated"), ("Shared memory used", "@shared_used"),
    ("FLAT: #edge, #geometry", "@fec, @fgc"),
    ("HIER: #edge, #geometry", "@hec, @hgc")
])


#############  Control callback function ############ 
def select_operations():
    global operations
    global current_file
    global chart_width

    sub_type_val = sub_type.value
    sub_type_name_val = sub_type_name.value
    op_name_val = op_name.value.strip()
    file_name = file_list.value.strip()

    # Parse log and replace data
    if (file_name != current_file):
        # Get data and reload HDB monitor chart
        operations, hdb1_fig, hdb2_fig, hdb3_fig, hdb1_table, hdb2_table, hdb3_table = prepare_operations(file_name, chart_width)
        current_file = file_name

        # Gen and reload Pareto chart
        gen_plot_page(operations, chart_width)
        l.children[1] = Div(text=open("pareto_chart.html").read(), width=chart_width)

        # Reload Datatable 
        l.children[2] = widgetbox(gen_data_table(operations, chart_width))

        # Reload HDB monitor chart & table
        l.children[3] = hdb1_fig
        l.children[4] = widgetbox(hdb1_table)
        l.children[5] = hdb2_fig
        l.children[6] = widgetbox(hdb2_table)
        l.children[7] = hdb3_fig
        l.children[8] = widgetbox(hdb3_table)

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
        runtime_ratio=df['runtime_ratio'],
        scale_factor=df['scale_factor'],
        lvheap_used=df['lvheap_used'],
        lvheap_allocated=df['lvheap_allocated'],
        shared_used=df['shared_used'],
        fec=df['fec'],
        fgc=df['fgc'],
        hec=df['hec'],
        hgc=df['hgc'],
    )

#############  Layout and finalization ############ 

# Control panel configuration 
controls = [file_list, cpu_time, real_time, lvheap_used, scale_factor, shared_used, sub_type, sub_type_name, op_name, x_axis, y_axis]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())
inputs = widgetbox(*controls, width = widget_width, height = chart_height)


#############  Chart generation  ############ 

# Gen Pareto chart
gen_plot_page(operations, chart_width) # It will generate "pareto_chart.html"
pareto_chart = Div(text=open("pareto_chart.html").read(), width=chart_width) # Load into Div

# 2D analystic figure  
p = figure(plot_height = chart_height, plot_width = (chart_width-widget_width),
        title = "", toolbar_location = None, #selectable=True,
        tools = [hover, BoxZoomTool(), ResetTool()])
p.left[0].formatter.use_scientific = False
p.circle(x="x", y="y", source=source, size=10, color="color", line_color=None, fill_alpha="alpha")

# Data Table
table = widgetbox(gen_data_table(operations, chart_width))

table_hdb1 = widgetbox(hdb1_table)
table_hdb2 = widgetbox(hdb2_table)
table_hdb3 = widgetbox(hdb3_table)

# Finalize Layout setting
l = layout([
    [inputs, p],
    [pareto_chart],
    [table],
    [hdb1_fig],
    [table_hdb1],
    [hdb2_fig],
    [table_hdb2],
    [hdb3_fig],
    [table_hdb3],
], sizing_mode=sizing_mode)

update()  # initial load of the data

curdoc().add_root(l)
curdoc().title = "Calibre Transcript Analystic"

