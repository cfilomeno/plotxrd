import os
import pandas as pd
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, cli
from werkzeug.utils import secure_filename
import json
import jinja2
from bokeh.plotting import figure, output_file
from bokeh.io import show, save, curdoc
from bokeh.layouts import column, layout, WidgetBox
from bokeh.models import ColumnDataSource, RangeTool, BoxSelectTool, HoverTool, Button, CustomAction, CustomJS, Column
from bokeh.models.widgets import Slider, CheckboxGroup, MultiSelect, Button, TableColumn
from bokeh.resources import INLINE


from bokeh.embed import components, file_html, server_document
from bokeh.resources import CDN, INLINE

UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file', filename=filename))
    return render_template('index.html')


# @app.route('/teste')
# def teste():
#     button = Button(label='FIT', aspect_ratio=10, background='red',
#                     align='center', button_type='primary',
#                     width_policy='max')
#
#     # w1 = Button(label='FIT', aspect_ratio=10, background='red',
#     #                 align='center', button_type='primary',
#     #                 width_policy='max')
#
#
#     # Get JavaScript/HTML resources
#     script, div1 = components(WidgetBox(button, width=500))
#
#     # Get JavaScript/HTML resources
#
#     # js_resources = INLINE.render_js()
#     # css_resources = INLINE.render_css()
#     return render_template('teste.html',
#                            script=script, div1=div1)


@app.route('/<filename>')
def uploaded_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    def search_string_in_file(file_name, string_to_search):
        line_number = 0
        list_of_results = []
        with open(file_name, 'r') as read_obj:
            for line in read_obj:
                line_number += 1
                if string_to_search in line:
                    list_of_results.append((line_number, line.rstrip()))
        return list_of_results

    matched_lines = search_string_in_file(filepath, '[Data]')
    elem = ()
    for elem in matched_lines:
        break

    df = pd.read_csv(filepath, skipinitialspace=True, skiprows=elem[0])
    df.dropna(axis=1, inplace=True)
    source = ColumnDataSource(df)

    plt.figure(figsize=(10, 8), dpi=100)
    plt.plot(df['Angle'], df['Det1Disc1'], label=filename[0:-4])
    plt.legend(loc='upper left')
    generatedfilename = 'plot_' + filename[0:-4] + '.png'
    plt.savefig(os.path.join('static', generatedfilename))

    # p = figure(plot_height=700, plot_width=1000, toolbar_location="right", tools="save,xwheel_zoom,"
    #                                                                              ",reset,crosshair",
    #            x_axis_type="linear", x_axis_location="below",
    #            background_fill_color=None, x_range=(df['Angle'].iloc[0], df['Angle'].iloc[-1]))
    #
    # p.add_tools(BoxSelectTool(dimensions='width'))
    # p.add_tools(CustomAction(icon="/home/cleber/Documentos/gitHub/plotxrd/static/icon.png",
    #                          callback=CustomJS(code='alert("foo")')))
    # legend = filename[0:-4]
    #
    # p.scatter('Angle', 'Det1Disc1', source=source, legend_label=legend, color='red', size=0.2, alpha=0)
    # p.line('Angle', 'Det1Disc1', source=source, legend_label=legend, line_color='#0020C2')
    # p.yaxis.axis_label = 'I (U.A.)'
    # p.xaxis.axis_label = '2-theta (degree)'
    # # button = Button(label='FIT', aspect_ratio=10, background='red',
    # #                 align='center', button_type='primary',
    # #                 width_policy='max')
    #
    # select = figure(title='Selecionar pico',
    #                 plot_height=110, plot_width=1030, y_range=p.y_range,
    #                 x_axis_type="linear", y_axis_type=None,
    #                 tools="", toolbar_location=None, background_fill_color="#efefef")
    # range_tool = RangeTool(x_range=p.x_range)
    # range_tool.overlay.fill_color = "navy"
    # range_tool.overlay.fill_alpha = 0.2
    # select.line('Angle', 'Det1Disc1', source=df)
    # select.ygrid.grid_line_color = None
    # select.xgrid.grid_line_color = None
    # select.add_tools(range_tool)
    # select.toolbar.active_multi = range_tool

    # script, divs = components((p, select))

    return render_template('page2.html', plot=generatedfilename)
    # , script=script, div0=divs[0], div1=divs[1])


@app.route('/<filename>/fit')
def fit(filename):
    def pvoigt(x, A, mu, sigma, alpha, K):
        sigmag = sigma * math.sqrt(2 * math.log(2))
        return ((1 - alpha) * A / (sigmag * math.sqrt(2 * math.pi))) * (np.exp(-(x - mu) ** 2 / (2 * sigmag ** 2))) + (
                    alpha * A / math.pi) * (sigma / ((x - mu) ** 2 + sigma ** 2)) + K

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    def search_string_in_file(file_name, string_to_search):
        line_number = 0
        list_of_results = []
        with open(file_name, 'r') as read_obj:
            for line in read_obj:
                line_number += 1
                if string_to_search in line:
                    list_of_results.append((line_number, line.rstrip()))
        return list_of_results

    matched_lines = search_string_in_file(filepath, '[Data]')
    elem = ()
    for elem in matched_lines:
        break

    df = pd.read_csv(filepath, skipinitialspace=True, skiprows=elem[0])
    df.dropna(axis=1, inplace=True)
    source = ColumnDataSource(df)

    # p = figure(plot_height=700, plot_width=1000, toolbar_location="right", tools="save,xwheel_zoom,"
    #                                                                              ",reset,crosshair",
    #            x_axis_type="linear", x_axis_location="below",
    #            background_fill_color=None, x_range=(df['Angle'].iloc[0], df['Angle'].iloc[-1]))
    #
    # p.add_tools(BoxSelectTool(dimensions='width'))
    # p.add_tools(CustomAction(icon="/home/cleber/Documentos/gitHub/plotxrd/static/icon.png",
    #                          callback=CustomJS(code='alert("foo")')))
    # # legend = filename[0:-4]
    #
    # # p.scatter('Angle', 'Det1Disc1', source=source, legend_label=legend, color='red', size=0.2, alpha=0)
    # # p.line('Angle', 'Det1Disc1', source=source, legend_label=legend, line_color='#0020C2')
    # p.yaxis.axis_label = 'I (U.A.)'
    # p.xaxis.axis_label = '2-theta (degree)'

    point_a = 60
    point_b = 65

    afit = df[(df['Angle'] >= point_a) & (df['Angle'] <= point_b)]
    afit.reset_index(drop=True, inplace=True)

    guess = [max(afit['Det1Disc1']), np.mean(afit['Angle']), 0.5, 0, min(afit['Det1Disc1'])]

    n = len(afit['Angle'])
    y = np.empty(n)

    for i in range(n):
        y[i] = pvoigt(afit['Angle'][i], guess[0], guess[1], guess[2], guess[3], guess[4])

    a = afit['Angle'].values
    s = afit['Det1Disc1'].values
    c, cov = curve_fit(pvoigt, a, s, guess)

    for i in range(n):
        y[i] = pvoigt(afit['Angle'][i], c[0], c[1], c[2], c[3], c[4])

    # plt.scatter(df['Angle'], df['Int'], alpha=0.5, color='black')
    # p.line('Angle', y, legend_label='Cu2teste', line_color='red')
    plt.plot(df['Angle'], df['Det1Disc1'], label=filename[0:-4], marker=',')
    plt.plot(afit['Angle'], y, 'r')
    textstr = '\n'.join((
        r'A = %.2f' % (c[0],),
        r'$\mu$ = %.2f' % (c[1],),
        r'$\sigma$ = %.2f' % (c[2],),
        r'$\alpha$ = %.2f' % (c[3],),
        r'offset = %.2f' % (c[4],)))

    plt.legend()
    pos_x = max(df['Angle'])*0.8
    pos_y = max(df['Det1Disc1'])*0.7
    plt.text(pos_x, pos_y, textstr, fontsize=10)
    generatedfilename = 'fit_' + filename[0:-4] + '.png'
    plt.savefig(os.path.join('static', generatedfilename))
    return render_template('page3.html', fit=generatedfilename, amp=c[0], media=c[1], sigma=c[2], alpha=c[3], offset=c[4])

    # script, div = components(p)

    # print(
    #     f'Amplitude: {c[0]:3.3f},\n Média mu: {c[1]:3.3f},\n Sigma: {c[2]:3.3f},\n Alpha: {c[3]:3.3f},\n K (offset): {c[4]:3.3f}')

    # return render_template('page3.html', script=script, div=div,
    #                        amp=c[0], media=c[1], sigma=c[2], alpha=c[3], offset=c[4])


if __name__ == "__main__":
    app.run(debug=True)
