import os
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pickle

# Load the pre-generated charts
with open('missed_slot_over_time_chart.pkl', 'rb') as f:
    missed_slot_over_time_charts = pickle.load(f)
    
with open('time_in_slot_scatter_chart.pkl', 'rb') as f:
    time_in_slot_scatter_charts = pickle.load(f)
    
with open('last_updated.txt', 'r') as f:
    last_updated = f.read().strip()


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        </script>
        <meta charset="UTF-8">
        <meta name="twitter:card" content="summary_large_image">
        <meta name="twitter:site" content="@nero_ETH">
        <meta name="twitter:title" content="Ethereum Timing Dashboard">
        <meta name="twitter:description" content="Selected comparative visualizations on timing games on Ethereum.">
        <meta name="twitter:image" content="https://raw.githubusercontent.com/nerolation/censorship.pics/main/assets/censorship.jpg">
        <meta property="og:title" content="Timings.pics" relay="" api="" dashboard="">
        <meta property="og:site_name" content="timings.pics">
        <meta property="og:url" content="timings.pics">
        <meta property="og:description" content="Selected comparative visualizations on timings on Ethereum.">
        <meta property="og:type" content="website">
        <link rel="shortcut icon" href="https://raw.githubusercontent.com/nerolation/timings.pics/main/assets/timings.jpg">
        <meta property="og:image" content="https://raw.githubusercontent.com/nerolation/timings.pics/main/assets/timings.jpg">
        <meta name="description" content="Selected comparative visualizations on reorged blocks on Ethereum.">
        <meta name="keywords" content="Ethereum, Timings, Dashboard">
        <meta name="author" content="Toni Wahrstätter">
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
app.scripts.append_script({"external_url": "update_window_width.js"})
app.clientside_callback(
    "window.dash_clientside.update_window_size",
    Output('window-size-store', 'data'),
    Input('window-size-trigger', 'n_intervals')
)

app.title = 'Timings.pics'
server = app.server

app.layout = html.Div([
    dbc.Container([
        dbc.Row(html.H1("Ethereum Timing Dashboard", style={'textAlign': 'center', 'marginTop': '20px', 'color': '#ffffff', 'fontFamily': 'Ubuntu Mono, monospace', 'fontWeight': 'bold'}), className="mb-4"),    
        dbc.Row([
            dbc.Col(html.Div([
               html.H6([
                    "Built with 🤍 by ",
                    html.A("Toni Wahrstätter", href="https://twitter.com/nero_eth", target="_blank", 
                           style={'color': '#ffffff'})
                ], style={'color': '#ffffff'}),
                
                html.H5("What are timing games?", style={'marginTop': '20px', 'color': '#ffffff'}),
                html.P("Timing games involve validators delaying their block proposals to increase their MEV rewards. This tactic gives builders extra time to enhance their blocks, leading to more MEV for the validator. As a result, those entities enganging in such games might be able to offer higher APYs, attracting more stakers and users. This site aims to keep you informed about the latest in timing games and how they're affecting the consensus layer.", style={'color': '#ffffff'}),
            ]), width=6),
            dbc.Col(html.Div([
                html.H6(f"Last time updated: {last_updated}", style={'color': '#ffffff'}),
                html.H5("Why does it matter?", style={'marginTop': '20px', 'color': '#ffffff'}),
                html.P("Timing games present a challenge to decentralization and censorship resistance. They favor sophisticated entities like staking pools or centralized exchanges, who have the resources and size to experiment with risky strategies and ensure their late blocks remain on the canonical chain. These tactics aren't feasible for solo stakers, who can't afford the risk of missing a slot or engaging in such experimentation.", style={'color': '#ffffff'}),
            ]), width=6),
        ]),
        dbc.Row(dcc.Interval(id='window-size-trigger', interval=1000, n_intervals=0, max_intervals=1)),
        dcc.Store(id='window-size-store', data={'width': 800}),
        dbc.Checklist(
            id='entity-selector',
            options=[{'label': entity, 'value': entity} for entity in missed_slot_over_time_charts.keys()],
            value=['Coinbase'],  # Default value
            switch=True,  # Use toggle switch style checkboxes
            inline=True,  # Align checkboxes horizontally
            className='my-2',  # Add margin
            style={
                'fontSize': '16px',  # Larger font size for readability
                'color': 'white',  # Text color
            }
        ),
        html.Div(id='charts-container', style={'backgroundColor': '#0a0a0a'}),
        html.Div([
            dbc.Row(
                dbc.Col(html.Div([
                    html.H6("Additional resources:", style={'color': '#ffffff'}),
                    html.Ul([
                        html.Li([
                            html.A("Timing Games: Implications and Possible Mitigations", href="https://ethresear.ch/t/timing-games-implications-and-possible-mitigations/17612", target="_blank", style={'color': '#4E9CAF'}),
                            " by Caspar and Mike"
                        ]),
                        html.Li([
                            html.A("Time to Bribe: Measuring Block Construction Market", href="https://arxiv.org/abs/2305.16468", target="_blank", style={'color': '#4E9CAF'}),
                            " by Toni et al."
                        ]),
                        html.Li([
                            html.A("Time is Money: Strategic Timing Games in Proof-of-Stake Protocols", href="https://arxiv.org/abs/2305.09032", target="_blank", style={'color': '#4E9CAF'}),
                            " by Caspar et al."
                        ]),
                        html.Li([
                            html.A("Time, slots, and the ordering of events in Ethereum Proof-of-Stake", href="https://www.paradigm.xyz/2023/04/mev-boost-ethereum-consensus", target="_blank", style={'color': '#4E9CAF'}),
                            " by Georgios and Mike"
                        ]),
                        html.Li([
                            html.A("Time is Money (Youtube)", href="https://www.youtube.com/watch?v=gsFU-inKRQ8", target="_blank", style={'color': '#4E9CAF'}),
                            " by Caspar"
                        ])
                    ], style={'paddingLeft': '20px', 'color': '#ffffff'})

                ]), width=12),
            )
        ], style={'backgroundColor': '#0a0a0a', 'padding': '20px', 'marginTop': '5vh'}, id='additional-resources')
        
    ], fluid=True, style={"maxWidth": "960px", 'backgroundColor': '#0a0a0a'})
], id='main-div', style={
    "display": "flex",
    "flexDirection": "column",
    "justifyContent": "center",
    "alignItems": "center",
    "minHeight": "100vh",
    'backgroundColor': '#0a0a0a',
})

@app.callback(
    Output('charts-container', 'children'),
    [Input('entity-selector', 'value')]
)
def update_charts(selected_entities):
    max_y_value_missed_slot = 0 
    
    for entity in selected_entities:
        missed_slot_fig = missed_slot_over_time_charts.get(entity)
        if missed_slot_fig:
            all_y_values = [y for trace in missed_slot_fig.data for y in trace.y]
            current_max_y = max(all_y_values, default=0)
            max_y_value_missed_slot = max(max_y_value_missed_slot, current_max_y)

    rows = []

    # Process the entities in reverse order so the latest selection is at the top
    for entity in reversed(selected_entities):
        cols = []

        # Create a header for the entity and append it to the rows list
        entity_header = dbc.Row(
            dbc.Col(
                html.H4(f"{entity}", style={
                        'textAlign': 'center',
                        'paddingTop': '1vh',
                        'paddingBottom': '1vh',
                        'width':"100%",
                        'color': '#ffffff',
                        "backgroundColor": "#333333",
                        "marginTop": "5vh",
                        "borderRadius": "10px"  # This adds rounded corners. Adjust as necessary.
                    }
                ),
                width=12
            ),
            className='mb-2'
        )
        rows.append(entity_header)
        
        # Retrieve the corresponding figures for the entity
        time_in_slot_scatter_fig = time_in_slot_scatter_charts.get(entity)
        missed_slot_fig = missed_slot_over_time_charts.get(entity)

        # Update the missed slot over time chart with the new y-axis range
        if missed_slot_fig:
            missed_slot_fig.update_layout(yaxis=dict(range=[0, max_y_value_missed_slot]))
            cols.append(dbc.Col(dcc.Graph(figure=missed_slot_fig), width=6, md=6))

        # Add the time in slot scatter chart
        if time_in_slot_scatter_fig:
            cols.append(dbc.Col(dcc.Graph(figure=time_in_slot_scatter_fig), width=6, md=6))

        # Add the current set of charts (and their header) to the rows
        rows.append(dbc.Row(cols))

    return rows




if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
