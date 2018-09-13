import base64
import io
import subprocess
import flask
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import numpy as np
import pandas as pd
import os

app = dash.Dash()
server = app.server
app.title = 'AmrutaInc'
app.scripts.config.serve_locally = True

app.layout = html.Div([

        html.A(html.Img(src="/static/Smiley.png", alt='Smiley',
        style={

            'width': '30%',
            'height': '50px'
            }), href='http://google.com/'),

            html.H2('GEICO'),

    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={

            'width': '30%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'solid',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'float': 'left'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),


    html.A(html.Button('Download',
            style={

                'width': '30%',
                'height': '62px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'solid',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px',
                'float': 'left'
                },),
                id='download-link',
                download="datadictionary_fraud_freetrial.csv",
                href="",
                target="_blank",
                n_clicks=0
        ),

    html.Div(id='output-data-upload'),
    html.Div(dt.DataTable(rows=[{}]), style={'display': 'none'})

])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            df.to_csv(filename)
            p = subprocess.Popen("python initial_modeling.py "+filename,shell=False,cwd=os.getcwd())
            p.wait()
            print("script run")
            r = pd.DataFrame(np.random.uniform(low=0, high=1, size=len(df)), columns=["Uniformed Score"])
            temp = pd.read_csv("predictions.csv")
            # To Display all the columns
            # temp = pd.concat([r, temp, df],axis=1)
            # To display only target and score columns
            temp = pd.concat([r,temp],axis=1)
            print("concat")
            return html.Div([html.H5(filename),
            dt.DataTable(rows=temp.to_dict(orient='records'),row_selectable=True,),
            html.P('Do you agree?',
            style={
                'textAlign': 'center',
            }),
            dcc.RadioItems(
            options=[
                {'label': 'Yes', 'value': 'Yes'},
                {'label': ' No ', 'value': 'No'},
            ],
            value='No',
            style={
                'textAlign': 'center',
            }
            )

    ])
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            print("in elif")
            df = pd.read_excel(io.BytesIO(decoded))
            df.to_csv(filename)
            r = pd.DataFrame(np.random.uniform(low=0, high=1, size=len(df)), columns=["Uniformed Score"])

            p = subprocess.Popen("python initial_modeling.py "+filename,shell=False,cwd=os.getcwd())
            p.wait()

            print("script run")
            temp = pd.read_csv("predictions.csv")
            # To Display all the columns
            # temp = pd.concat([r, temp, df],axis=1)
            # To display only target and score columns
            temp = pd.concat([r,temp],axis=1)

            return html.Div([html.H5(filename),
            dt.DataTable(rows=temp.to_dict(orient='records'),row_selectable=True,),
            html.P('Do you agree?',
            style={
                'textAlign': 'center',
            }),
            dcc.RadioItems(
            options=[
                {'label': 'Yes', 'value': 'Yes'},
                {'label': ' No ', 'value': 'No'},
            ],
            value='No',
            style={
                'textAlign': 'center',
            }
            )

    ])

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename'),
               Input('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

@server.route('/favicon.ico')
def favicon():
    return flask.send_from_directory(os.path.join(server.root_path, 'static'),
                                     'Robsonbillponte-Sinem-File-Downloads.ico')

@app.callback(
    Output('download-link', 'href'),
    [Input('download-link', 'n_clicks')])

def generate_report_url(n_clicks):

    return '/dash/urldownload'

@app.server.route('/dash/urldownload')

def generate_report_url():

    return flask.send_file('datadictionary_fraud_freetrial.csv', attachment_filename = 'datadictionary_fraud_freetrial.csv', as_attachment = True)


if __name__ == '__main__':
    app.run_server(debug=True)
