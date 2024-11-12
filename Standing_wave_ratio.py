# STANDING_WAVE_RATIO.PY
# Nathaniel Heatwole, PhD (heatwolen@gmail.com)
# Reproduces a cross-needle SWR/Watt meter that simultaneously displays forward power (Watts), reflected power (Watts), and standing wave ratio (SWR)
# SWR assess mismatch between a transmission line and its load, with greater values indicating less good matches (SWR >= 1)
# SWR can be computed using: forward power (emanating from the transmitter), and reflected power (not accepted by the load)

import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style
from sympy import symbols, Eq, solve
from matplotlib.backends.backend_pdf import PdfPages

time0 = time.time()
ver = ''  # version (empty or integer)

topic = 'Standing wave ratio'
topic_underscore = topic.replace(' ','_')

#--------------#
#  PARAMETERS  #
#--------------#

swr_levels = [1, 1.3, 1.5, 2, 3, 5, 'inf']  # swr curves (can include infinity, as 'inf')

needle_length = 1       # length of each needle
angle_min_degrees = 15  # needle minimum angle (relative to horizontal) (0 <= angle < 45 deg)

fwd_pwr_max = 300  # forward power maximum (watts)
ref_pwr_max = 80   # reflected power maximum (watts)
pwr_increment = 1  # power increment (watts)

# rate of non-linearly in angle-power relationship for needles (0 < c <= 1)
c = 0.8  # angle = a + b * ln[1 + (power / max power) ^ c]

# major and minor tick mark increments (for plot)
fwd_major_tick_increment = 50
fwd_minor_tick_increment = 25
ref_major_tick_increment = 20
ref_minor_tick_increment = 10

#----------#
#  CHECKS  #
#----------#

swr_levels_num = [swr for swr in swr_levels if swr != 'inf']

# check various input settings
error_msgs = []
if min(swr_levels_num) < 1:
    error_msgs.append('swr out of range')
if needle_length <= 0:
    error_msgs.append('needle length out of range')
if (not (0 <= angle_min_degrees < 45)):
    error_msgs.append('angle out of range')
if (fwd_pwr_max <= 0) or (ref_pwr_max <= 0):
    error_msgs.append('power level out of range')
if (fwd_pwr_max % pwr_increment != 0) or (ref_pwr_max % pwr_increment != 0):
    error_msgs.append('power increment out of range')
if (not (0 < c <= 1)):
    error_msgs.append('c-value out of range')

# take action (if needed)
if len(error_msgs) > 0:
    for m in range(len(error_msgs)):
        print(Fore.RED + '\033[1m' + '\n' + 'ERROR: ' + error_msgs[m].upper() + '\n' + Style.RESET_ALL)
    sys.exit('execution ceased')
del error_msgs

#----------#
#  INPUTS  #
#----------#

degrees_to_radians = (2 * math.pi) / 360

fwd_pwr_levels = list(np.arange(0, fwd_pwr_max + pwr_increment, pwr_increment))
ref_pwr_levels = list(np.arange(0, ref_pwr_max + pwr_increment, pwr_increment))

# range of angles (radians) for needle movement (relative to horizontal)
angle_min = angle_min_degrees * degrees_to_radians
angle_max = (90 * degrees_to_radians) - angle_min  # assumes needles are perpendicular when one is at its maximum and the other is at its minimum

# needle angle (relative to horizontal): angle = a + b * ln[1 + (power / max_power)^c]
a = angle_min  # because the log term approaches zero as power goes to zero
b = (angle_max - angle_min) / math.log(2)  # because angle is at its maximum when power ratio = 1

# needle distance to origin (set so when one needle is maximized, it intersects the zero of and is perpendicular to the other power curve)
x_offset_needle = (0.5 * needle_length) / np.cos(angle_min)

# needle rotation point coordinates (origin set midway between these points)
y_fwd_needle = 0
y_ref_needle = 0
x_fwd_needle = x_offset_needle
x_ref_needle = -x_offset_needle

#-------------#
#  FUNCTIONS  #
#-------------#

# convert swr to string (decimal points -> 'pt')
def swr_as_text(swr):
    return str(swr).replace('.', 'pt')

# reflection coefficient (ratio of reflected power to forward power)
def reflection_coef(swr):
    if swr == 'inf':
        return 1
    else:
        return (swr - 1)**2 / (swr + 1)**2  # see https://en.wikipedia.org/wiki/Standing_wave_ratio#Relationship_to_the_reflection_coefficient

# needle angle (radians) (relative to horizontal)
def angle(pwr_ratio):
    return a + b * math.log(1 + pwr_ratio ** c)  # power ratio is power divided by maximum power level (forward or reflected)

# point where forward power needle intersect its corresponding power curve
def fwd_coords(angle_fwd):
    x_fwd = x_fwd_needle - (needle_length * np.cos(angle_fwd))
    y_fwd = y_fwd_needle + (needle_length * np.sin(angle_fwd))
    return x_fwd, y_fwd

# point where reflected power needle intersect its corresponding power curve
def ref_coords(angle_ref):
    x_ref = x_ref_needle + (needle_length * np.cos(angle_ref))
    y_ref = y_ref_needle + (needle_length * np.sin(angle_ref))
    return x_ref, y_ref

#--------------#
#  SWR CURVES  #
#--------------#

# generate swr curves
for swr in swr_levels:
    for fwd_pwr in fwd_pwr_levels:        
        ref_pwr = fwd_pwr * reflection_coef(swr)  # reflected power
        
        # needle angles (relative to horizontal)
        angle_fwd = angle(fwd_pwr / fwd_pwr_max)
        angle_ref = angle(ref_pwr / ref_pwr_max)
 
        # point where each needle intersects its corresponding power curve
        x_fwd, y_fwd = fwd_coords(angle_fwd)
        x_ref, y_ref = ref_coords(angle_ref)

        # line connecting rotation point of each needle to particular point (watts) on its corresponding power curve
        slope_fwd = (y_fwd - y_fwd_needle) / (x_fwd - x_fwd_needle)
        slope_ref = (y_ref - y_ref_needle) / (x_ref - x_ref_needle)
        intercept_fwd = abs(slope_fwd) * x_fwd_needle
        intercept_ref = slope_ref * abs(x_ref_needle)
        
        # point where the two needles cross ('cross' in cross-needle meter)
        x, y = symbols('x y')
        eq1 = Eq(y - ((slope_fwd * x) + intercept_fwd), 0)
        eq2 = Eq(y - ((slope_ref * x) + intercept_ref), 0)
        sol_dict = solve((eq1, eq2), (x, y))  # simultaneously solve for coordinates of intersection point (2 equations / 2 unknowns)
        x_cross, y_cross = float(sol_dict[x]), float(sol_dict[y])
        
        # check inclusion criteria (if distance to rotation point of both needles does not exceed the length of the needle)
        dist_fwd_needle = math.sqrt((y_cross - y_fwd_needle) ** 2 + (x_cross - x_fwd_needle) ** 2)
        dist_ref_needle = math.sqrt((y_cross - y_ref_needle) ** 2 + (x_cross - x_ref_needle) ** 2)
        include_pt = (dist_fwd_needle <= needle_length) and (dist_ref_needle <= needle_length)
        
        # save current coordinates (if inclusion criteria met)
        if include_pt == True:
            x_name = 'x_swr_' + swr_as_text(swr)
            y_name = 'y_swr_' + swr_as_text(swr)
            try:  # case that lists have already been created
                exec(x_name + '.append(x_cross)')
                exec(y_name + '.append(y_cross)')
            except:  # case that lists are new
                exec(x_name + '= [x_cross]')
                exec(y_name + '= [y_cross]')

#---------------#
#  PLOT INPUTS  #
#---------------#

# parameters
title_size = 11
axis_label_size = 8
axis_tick_label_size = 8
legend_size = 8
line_width = 1.25
major_tick_size = 15
minor_tick_size = 15
needle_pt_size = 75

# spatial buffers (zero = no buffer)
pwr_labels_buffer = 0.04
major_tick_buffer = 0.02
x_buffer_plot_edge = 0.6
y_buffer_plot_edge = 0.1

# initalize
fwd_labels = []
ref_labels = []
x_fwd_curve = []
x_ref_curve = []
y_fwd_curve = []
y_ref_curve = []
x_fwd_offset = []
x_ref_offset = []
y_fwd_offset = []
y_ref_offset = []
x_fwd_major_ticks = []
x_ref_major_ticks = []
y_fwd_major_ticks = []
y_ref_major_ticks = []
x_fwd_minor_ticks = []
x_ref_minor_ticks = []
y_fwd_minor_ticks = []
y_ref_minor_ticks = []

# forward power curve (black semi-circle - opens to southeast) - with labels and tick marks
for f in fwd_pwr_levels:
    # curve itself
    angle_fwd = angle(f / fwd_pwr_max)
    x_fwd, y_fwd = fwd_coords(angle_fwd)
    x_fwd_curve.append(x_fwd)
    y_fwd_curve.append(y_fwd)
    # major tick marks (at min and max power levels and user-specified increments)
    if (f == 0) or (f == fwd_pwr_max) or (f % fwd_major_tick_increment == 0):
        # coordinates (x, y)
        fwd_labels.append(f)
        x_fwd_major_ticks.append(x_fwd)
        y_fwd_major_ticks.append(y_fwd)
        # offset (so power labels are not flush with the power curves)
        offset_x_fwd = -1 * major_tick_buffer * needle_length * np.cos(angle_fwd)
        offset_y_fwd = major_tick_buffer * needle_length * np.sin(angle_fwd)
        x_fwd_offset.append(offset_x_fwd)
        y_fwd_offset.append(offset_y_fwd)
    # minor tick marks (at user-specified increments) (no labels)
    elif (f % fwd_minor_tick_increment == 0):
        x_fwd_minor_ticks.append(x_fwd)    
        y_fwd_minor_ticks.append(y_fwd)

# reflected power curve (black semi-circle - opens to southwest) - with labels and tick marks
for r in ref_pwr_levels:
    # curve itself
    angle_ref = angle(r / ref_pwr_max)
    x_ref, y_ref = ref_coords(angle_ref)
    x_ref_curve.append(x_ref)
    y_ref_curve.append(y_ref)
    # major tick marks (at min and max power levels and user-specified increments)
    if (r == 0) or (r == ref_pwr_max) or (r % ref_major_tick_increment == 0):
        # coordinates (x, y)
        ref_labels.append(r)
        x_ref_major_ticks.append(x_ref)
        y_ref_major_ticks.append(y_ref)
        # offset (so power labels are not flush with the curve)
        offset_x_ref = major_tick_buffer * needle_length * np.cos(angle_ref)
        offset_y_ref = major_tick_buffer * needle_length * np.sin(angle_ref)
        x_ref_offset.append(offset_x_ref)
        y_ref_offset.append(offset_y_ref)
    # minor tick marks (at user-specified increments) (no labels)
    elif (r % ref_minor_tick_increment == 0):  
        x_ref_minor_ticks.append(x_ref)    
        y_ref_minor_ticks.append(y_ref)

# coordinates - needles at rest
y_fwd_min = y_fwd_major_ticks[0]
y_ref_min = y_ref_major_ticks[0]
x_fwd_min = x_fwd_major_ticks[0]
x_ref_min = x_ref_major_ticks[0]

# coordinates - needles maximized
y_fwd_max = y_fwd_major_ticks[-1]
y_ref_max = y_ref_major_ticks[-1]
x_fwd_max = x_fwd_major_ticks[-1]
x_ref_max = x_ref_major_ticks[-1]

#---------#
#  PLOTS  #
#---------#

# example scenario (forward power = 150 watts, SWR = 2)
fwd_pwr_ex = 150
swr_ex = 2
ref_coeff_ex = reflection_coef(swr_ex)
ref_pwr_ex = ref_coeff_ex * fwd_pwr_ex
angle_fwd_ex = angle(fwd_pwr_ex / fwd_pwr_max)
angle_ref_ex = angle(ref_pwr_ex / ref_pwr_max)
x_fwd_ex, y_fwd_ex = fwd_coords(angle_fwd_ex)
x_ref_ex, y_ref_ex = ref_coords(angle_ref_ex)

# titles
plot_title_overall = 'SWR/Watt cross-needle meter'
plot_subtitles = ['needles at rest', 'needles at maximum', 'full power, perfect match', '150 Watts forward, SWR = 2', 'needles omitted']

# generate plots
figs = []
for p in range(len(plot_subtitles)):
    # initialize plot
    exec('fig' + str(p + 1) + ' = plt.figure()')
    exec('figs.append(fig' + str(p + 1) + ')')
    
    # title
    plot_subtitle = plot_subtitles[p]
    plt.title(plot_title_overall + ' (' + plot_subtitle + ')', fontsize=title_size, fontweight='bold')

    # power curves and labels (black semi-circles)
    plt.plot(x_fwd_curve, y_fwd_curve, linewidth=line_width, color='black', zorder=5)
    plt.plot(x_ref_curve, y_ref_curve, linewidth=line_width, color='black', zorder=5)
    plt.text(x_fwd_min, y_fwd_min - (pwr_labels_buffer * needle_length), 'forward\n(watts)', va='top', ha='center', fontsize=axis_label_size)
    plt.text(x_ref_min, y_ref_min - (pwr_labels_buffer * needle_length), 'reflected\n(watts)', va='top', ha='center', fontsize=axis_label_size)
    
    # major tick marks (power curves) (with labels)
    plt.scatter(x_fwd_major_ticks, y_fwd_major_ticks, marker='*', color='black', s=major_tick_size)
    plt.scatter(x_ref_major_ticks, y_ref_major_ticks, marker='*', color='black', s=major_tick_size)
    for f in range(len(fwd_labels)):
        x_fwd_coord = x_fwd_major_ticks[f] + x_fwd_offset[f]
        y_fwd_coord = y_fwd_major_ticks[f] + y_fwd_offset[f]
        plt.annotate(fwd_labels[f], (x_fwd_coord, y_fwd_coord), va='bottom', ha='center', fontsize=axis_tick_label_size)
    for r in range(len(ref_labels)):
        x_ref_coord = x_ref_major_ticks[r] + x_ref_offset[r]
        y_ref_coord = y_ref_major_ticks[r] + y_ref_offset[r]
        plt.annotate(ref_labels[r], (x_ref_coord, y_ref_coord), va='bottom', ha='center', fontsize=axis_tick_label_size)
    
    # minor tick marks (power curves) (no labels)
    plt.scatter(x_fwd_minor_ticks, y_fwd_minor_ticks, marker='|', color='black', s=minor_tick_size)
    plt.scatter(x_ref_minor_ticks, y_ref_minor_ticks, marker='|', color='black', s=minor_tick_size)
    
    inf_label = 'inf'

    # swr curves (various colors)
    for swr in swr_levels:
        x_name = 'x_swr_' + swr_as_text(swr)
        y_name = 'y_swr_' + swr_as_text(swr)
        if swr == 'inf':
            full_name = x_name + ', ' + y_name + ', ' + 'label=inf_label' 
        else:
            full_name = x_name + ', ' + y_name + ', ' + 'label=' + str(swr)
        exec('plt.plot(' + full_name + ', linewidth=line_width, zorder=0)')

    # plots showing needles (blue lines)
    if plot_subtitle != 'needles omitted':
        # needle rotation points (large black stars)
        plt.scatter(x_fwd_needle, y_fwd_needle, marker='*', color='black', s=needle_pt_size)
        plt.scatter(x_ref_needle, y_ref_needle, marker='*', color='black', s=needle_pt_size)

        # both needles at rest ('first positions')
        if plot_subtitle == 'needles at rest':
            x_fwd_plot = [x_fwd_needle, x_fwd_min]
            y_fwd_plot = [y_fwd_needle, y_fwd_min]
            x_ref_plot = [x_ref_needle, x_ref_min]
            y_ref_plot = [y_ref_needle, y_ref_min]
        # both needles at maximum
        elif plot_subtitle == 'needles at maximum':
            x_fwd_plot = [x_fwd_needle, x_fwd_max]
            y_fwd_plot = [y_fwd_needle, y_fwd_max]
            x_ref_plot = [x_ref_needle, x_ref_max]
            y_ref_plot = [y_ref_needle, y_ref_max]
        # full power (perfect match)
        elif plot_subtitle == 'full power, perfect match':
            x_fwd_plot = [x_fwd_needle, x_fwd_max]
            y_fwd_plot = [y_fwd_needle, y_fwd_max]
            x_ref_plot = [x_ref_needle, x_ref_min]
            y_ref_plot = [y_ref_needle, y_ref_min]
        # example scenario
        elif plot_subtitle == '150 Watts forward, SWR = 2':
            x_fwd_plot = [x_fwd_needle, x_fwd_ex]
            y_fwd_plot = [y_fwd_needle, y_fwd_ex]
            x_ref_plot = [x_ref_needle, x_ref_ex]
            y_ref_plot = [y_ref_needle, y_ref_ex]

        # generate plot
        plt.plot(x_fwd_plot, y_fwd_plot, linewidth=line_width, color='blue')
        plt.plot(x_ref_plot, y_ref_plot, linewidth=line_width, color='blue')

    # finalize plot
    plt.xlim([-(x_buffer_plot_edge + 0.05) * needle_length, (x_buffer_plot_edge + 0.05) * needle_length])
    plt.ylim([-(0.5 * y_buffer_plot_edge) * needle_length, (y_buffer_plot_edge + 1) * needle_length])
    plt.axis('off')
    plt.legend(loc='upper right', fontsize=legend_size, facecolor='lightgrey', title='SWR')
    plt.show(True)

#----------#
#  EXPORT  #
#----------#

# export plots (pdf)
pdf = PdfPages(topic_underscore + '_plots' + ver + '.pdf')
for f in figs:
    pdf.savefig(f)
pdf.close()
del pdf, f, figs

###

# runtime
runtime_sec = round(time.time() - time0, 2)
if runtime_sec < 60:
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec')
else:
    runtime_min_sec = str(int(np.floor(runtime_sec / 60))) + ' min ' + str(round(runtime_sec % 60, 2)) + ' sec'
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec (' + runtime_min_sec + ')')
del time0


