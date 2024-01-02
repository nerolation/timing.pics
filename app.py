import os
import dash
from dash import dcc
from dash import html
from dash import callback_context
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pickle
#from flask_caching import Cache
import pandas as pd

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://use.fontawesome.com/releases/v5.8.1/css/all.css'])

#cache = Cache(app.server, config={
#    'CACHE_TYPE': 'simple'  # Use 'filesystem' or 'redis' for persistent caching
#})

#@cache.memoize(timeout=86400)  # Cache for one day (86400 seconds)
def load_data():
    original_marker_sizes = {}
    
    with open('missed_slot_over_time_chart.pkl', 'rb') as f:
        missed_slot_over_time_charts = pickle.load(f)
    
    with open('time_in_slot_scatter_chart.pkl', 'rb') as f:
        time_in_slot_scatter_charts = pickle.load(f)
        
    with open('gamer_bars.pkl', 'rb') as f:
        gamer_bars = pickle.load(f)
    
    with open('missed_slot_bars.pkl', 'rb') as f:
        missed_slot_bars = pickle.load(f)
        
    with open('gamer_advantage_lines.pkl', 'rb') as f:
        gamer_advantage_lines = pickle.load(f)
    
    with open('gamer_advantage_avg.pkl', 'rb') as f:
        gamer_advantage_avg = pickle.load(f)
        
    with open('missed_market_share_chart.pkl', 'rb') as f:
        missed_market_share_chart = pickle.load(f)
        
        
    for entity, fig in time_in_slot_scatter_charts.items():
        # Initialize the list for this entity
        for trace in fig.data:
            # Check if this is a scatter trace with a marker size attribute
            if 'marker' in trace and 'size' in trace.marker:
                original_marker_sizes[entity] = trace.marker.size
            else:
                pass
            # If not, append None or some other placeholder to maintain index alignment
                #original_marker_sizes[entity].append(None)

    
    return (
        missed_slot_over_time_charts, 
        time_in_slot_scatter_charts, 
        original_marker_sizes, 
        gamer_bars, 
        missed_slot_bars,
        gamer_advantage_lines,
        gamer_advantage_avg,
        missed_market_share_chart
    )

missed_slot_over_time_charts, time_in_slot_scatter_charts, original_marker_sizes, gamer_bars, missed_slot_bars, gamer_advantage_lines, gamer_advantage_avg, missed_market_share_chart = load_data()

reduced_size_markers = {}
for i, j in original_marker_sizes.items():
    reduced_size_markers[i] = j*0.6

    
with open('last_updated.txt', 'r') as f:
    last_updated = f.read().strip()


MAX_SELECTIONS = 4

KNOWN_CEX=[i for i in missed_slot_over_time_charts.keys()  if i in ['Coinbase', 'Kraken', 'Binance', 'Bitpanda', 'Bitstamp', 'Bitcoin suisse', "Upbit", "Coinspot"]][0:MAX_SELECTIONS]
KNOWN_LST=[i for i in missed_slot_over_time_charts.keys() if i in ['Lido', 'Rocketpool', 'Staked.us', "Figment", "Kiln", "Okx","P2p.org", "Stakefish", "Frax finance"]][0:MAX_SELECTIONS]


def update_figure_layout(fig, width, entity, marker=False):
    if width <= 800:
        fig.update_layout(
            font=dict(size=8),
            margin=dict(l=0, r=30, t=20, b=20), 
            xaxis_tickfont=dict(size=9),
            yaxis_tickfont=dict(size=9), 
            height=230
        )
        
    else:
        fig.update_layout(
            font=dict(size=16), 
            margin={"t":70,"b":0,"r":50,"l":0},
            xaxis_tickfont=dict(size=16),
            yaxis_tickfont=dict(size=16),
            height=250
        )
    if marker:
        for trace in fig.data:
            if 'marker' in trace:
                if "size" in trace['marker']:
                    trace['marker']['size'] = reduced_size_markers[entity] if width <= 800 else original_marker_sizes[entity]

    return fig

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <!-- Google tag (gtag.js) -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-GR9E3MCG52"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());

          gtag('config', 'G-GR9E3MCG52');
        </script>
        <meta charset="UTF-8">
        <meta name="twitter:card" content="summary_large_image">
        <meta name="twitter:site" content="@nero_ETH">
        <meta name="twitter:title" content="Ethereum Timing Dashboard">
        <meta name="twitter:description" content="Selected comparative visualizations on block proposal timings and missed slots on Ethereum.">
        <meta name="twitter:image" content="https://raw.githubusercontent.com/nerolation/timing.pics/main/assets/timinggames_og_image.jpg">
        <meta property="og:title" content="Timing.pics" relay="" api="" dashboard="">
        <meta property="og:site_name" content="timing.pics">
        <meta property="og:url" content="timing.pics">
        <meta property="og:description" content="Selected comparative visualizations on block proposal timings and missed slots on Ethereum.">
        <meta property="og:type" content="website">
        <meta property="og:image" content="https://raw.githubusercontent.com/nerolation/timing.pics/main/assets/timinggames_og_image.jpg">
        <meta name="description" content="Selected comparative visualizations on block proposal timings and missed slots on Ethereum.">
        <meta name="keywords" content="Ethereum, Timings, DotPics, Dashboard">
        <meta name="author" content="Toni Wahrstätter">
        <link rel="shortcut icon" href="https://raw.githubusercontent.com/nerolation/timing.pics/main/assets/favicon.png">
        {%metas%}
        <title>{%title%}</title>
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

app.title = 'Timing.pics'
server = app.server

app.layout = html.Div([
    dbc.Container([
        dbc.Row(html.H1("Ethereum Timing Dashboard", style={'textAlign': 'center', 'marginTop': '18px', 'color': '#ffffff', 'fontFamily': 'Ubuntu Mono, monospace', 'fontWeight': 'bold'}), className="mb-4"),    
        dbc.Row([
            dbc.Col(html.Div([
               html.H6([
                    "Built with 🤍 by ",
                    html.A("Toni Wahrstätter", href="https://twitter.com/nero_eth", target="_blank", 
                           style={'color': '#ffffff'})
                ],className='evensmallerfont', style={'color': '#ffffff'}),
                
                html.H5("What are timing games?", className='smallerfont', style={'marginTop': '20px', 'color': '#ffffff'}),
                html.P("Timing games involve validators delaying their block proposals to increase their MEV rewards. This tactic gives builders extra time to enhance their blocks, leading to more MEV for the validator. As a result, those entities enganging in such games might be able to offer higher APYs, attracting more stakers and users. This site aims to keep you informed about the latest in timing games and how they're affecting the consensus layer.", className='smallerfont', style={'color': '#ffffff'}),
            ]), width=6),
            dbc.Col(html.Div([
                html.H6(f"Last time updated: {last_updated}", className='evensmallerfont', style={'color': '#ffffff'}),
                html.H5("Why does it matter?", style={'marginTop': '18px', 'color': '#ffffff'}, className='smallerfont'),
                html.P("Timing games present a challenge to decentralization and censorship resistance. They favor sophisticated entities like staking pools or centralized exchanges, who have the resources and size to experiment with risky strategies and ensure their late blocks remain on the canonical chain. These tactics aren't feasible for solo stakers, who can't afford the risk of missing a slot or engaging in such experimentation.", style={'color': '#ffffff'}, className='smallerfont'),
            ]), width=6),
        ]),
        dbc.Row(dcc.Interval(id='window-size-trigger', interval=1000, n_intervals=0, max_intervals=1)),
        dcc.Store(id='window-size-store', data={'width': 800}),
        dbc.Tabs([
            dbc.Tab(
                label="General Info Charts", 
                children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='chart3', figure=gamer_advantage_lines), width={"size": 8, "offset": 0}, lg=8, xl=8),
                        dbc.Col(dcc.Graph(id='chart4', figure=gamer_advantage_avg), width={"size": 4, "offset": 0}, lg=4, xl=4),
                    ], justify="center"),
                    dbc.Row([
                    
                        dbc.Col(dcc.Graph(id='chart5', figure=missed_market_share_chart), width={"size": 12, "offset": 0}, lg=12, xl=12),
                    ], justify="center"),  
                    dbc.Row([
                        # The chart takes up the full width on extra small to small screens,
                        # and an appropriate fraction of the width on larger screens
                        dbc.Col(dcc.Graph(id='chart1', figure=gamer_bars), width={"size": 6, "offset": 0}, lg=5, xl=5),

                        # Second chart does the same
                        dbc.Col(dcc.Graph(id='chart2', figure=missed_slot_bars), width={"size": 6, "offset": 0}, lg=5, xl=5),
                    ], justify="center"), 
                       
                     
                ], 
                tab_style={"margin": "10px", "padding": "10px", "fontWeight": "bold"},
                tab_class_name='custom-tab',
                label_style={"color": "#ffffff"},
                active_label_style={"color": "#000000"}
            ),
            dbc.Tab(
                label="Per Validator", 
                children=[
                    dcc.Store(id='selected_order', storage_type='session'),
                    dbc.Checklist(
                        id='entity-selector',
                        options=[
                            {'label': 'Largest CEXs', 'value': 'All CEX'},
                            {'label': 'Largest Pools', 'value': 'All LSTs'}
                        ] + 
                        [{'label': entity.split("<")[0], 'value': entity} for entity in missed_slot_over_time_charts.keys()],
                        value=['Coinbase'],  # Default value
                        switch=True,
                        inline=True,
                        className='my-2 smallerfont',
                        style={
                            'color': 'white',  # Text color
                        }
                    ),
                    dcc.Loading(
                        id="loading-1",
                        type="default",  # You can change the spinner type here (options: 'graph', 'cube', 'circle', 'dot', and 'default')
                        children=html.Div(id='charts-container', style={'backgroundColor': '#0a0a0a'})
                    ),
                ],
                tab_style={"margin": "10px", "padding": "10px", "fontWeight": "bold"},
                tab_class_name='custom-tab',
                label_style={"color": "#ffffff"},
                active_label_style={"color": "#000000"}
            )
        ], style={'fontSize': '18px', 'fontFamily': 'Ubuntu Mono, monospace'}),

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
                            html.A("The cost of artificial latency in the PBS context", href="https://ethresear.ch/t/the-cost-of-artificial-latency-in-the-pbs-context/17847s", target="_blank", style={'color': '#4E9CAF'}),
                            " by Chorus One"
                        ]),
                        html.Li([
                            html.A("Empirical analysis of the impact of block delays on the consensus layer", href="https://ethresear.ch/t/empirical-analysis-of-the-impact-of-block-delays-on-the-consensus-layer/17888", target="_blank", style={'color': '#4E9CAF'}),
                            " by Kiln"
                        ]),
                        html.Li([
                            html.A("Time, slots, and the ordering of events in Ethereum Proof-of-Stake", href="https://www.paradigm.xyz/2023/04/mev-boost-ethereum-consensus", target="_blank", style={'color': '#4E9CAF'}),
                            " by Georgios and Mike"
                        ]),
                        html.Li([
                            html.A("P2P Presentation on Timing Games (Youtube)", href="https://youtu.be/J_N13erDWKw?t=1061", target="_blank", style={'color': '#4E9CAF'}),
                            " by P2P"
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
    [Input('window-size-store', 'data'),
     Input('entity-selector', 'value')]
)
def update_charts(window_size_data, selected_entities):
    if window_size_data is None:
        raise dash.exceptions.PreventUpdate

    # Determine the triggering input
    ctx = callback_context
    if not ctx.triggered:
        trigger_id = 'No input yet'
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    width = window_size_data['width']
    max_y_value_missed_slot = 0
    max_y_value_missed_slot = 0 
    
    
    if 'All CEX' in selected_entities:
        for i in reversed(KNOWN_LST):
            if i in selected_entities:
                selected_entities.remove(i)
        selected_entities.extend(KNOWN_CEX)  # Replace with actual values if different

    if 'All LSTs' in selected_entities:
        for i in reversed(KNOWN_CEX):
            if i in selected_entities:
                selected_entities.remove(i)
        selected_entities.extend(KNOWN_LST) 
    
    for entity in selected_entities:
        missed_slot_fig = missed_slot_over_time_charts.get(entity)
        if missed_slot_fig:
            all_y_values = [y for trace in missed_slot_fig.data for y in trace.y]
            all_y_values = [0 if pd.isna(x) else x for x in all_y_values]
            current_max_y = max(all_y_values, default=0)
            max_y_value_missed_slot = max(max_y_value_missed_slot, current_max_y)

    rows = []

    # Process the entities in reverse order so the latest selection is at the top
    last_added_marker_shrink = True
    for entity in reversed(selected_entities):
        cols = []
        entity_name = entity.split(' <')[0]
        info_symbol = html.Span(
            className="fas fa-info-circle", 
            id=f"tooltip-target-{entity_name}",
            style={
                'textAlign': 'center',
                'marginLeft': '10px',
                'cursor': 'pointer'
            }
        )
        entity_header = dbc.Row(
            dbc.Col(
                html.H4([entity_name, info_symbol if entity_name == "Solo Stakers" else None], style={
                        'textAlign': 'center',
                        'paddingTop': '1vh',
                        'paddingBottom': '1vh',
                        'width':"100%",
                        'color': '#ffffff',
                        "backgroundColor": "#333333",
                        "marginTop": "5vh",
                        "borderRadius": "10px"
                    }
                ),
                width=12
            ),
            className='mb-2'
        )
        rows.append(entity_header)
        if entity_name == "Solo Stakers":
            tooltip = dbc.Tooltip(
                "Solo Stakers are identified through various heuristics applied in a highly conservative manner. Consequently, the displayed number of Solo Stakers may represent a lower bound, potentially underestimating the actual count.",
                target=f"tooltip-target-{entity_name}",
                placement="top",
            )
            rows.append(tooltip)  

        # Retrieve the corresponding figures for the entity
        time_in_slot_scatter_fig = time_in_slot_scatter_charts.get(entity)
        missed_slot_fig = missed_slot_over_time_charts.get(entity)

        # Update the missed slot over time chart with the new y-axis range
        if missed_slot_fig:
            missed_slot_fig.update_layout(yaxis=dict(range=[0, max_y_value_missed_slot]))
            missed_slot_fig = update_figure_layout(missed_slot_fig, width, entity)
            cols.append(dbc.Col(dcc.Graph(figure=missed_slot_fig), width=6, md=6))

        # Add the time in slot scatter chart
        if time_in_slot_scatter_fig:
            time_in_slot_scatter_fig = update_figure_layout(time_in_slot_scatter_fig, width, entity, marker=last_added_marker_shrink)
            cols.append(dbc.Col(dcc.Graph(figure=time_in_slot_scatter_fig), width=6, md=6))

        # Add the current set of charts (and their header) to the rows
        rows.append(dbc.Row(cols))
        last_added_marker_shrink = False

    return rows


@app.callback(
    [Output('entity-selector', 'value'),
     Output('selected_order', 'data')],  # Additional output for the order of selections
    [Input('entity-selector', 'value')],
    [State('selected_order', 'data')]  # Include the previous order of selections
)
def set_checklist_values(selected_entities, selected_order):
    if not selected_entities:
        raise PreventUpdate

    all_selected_entities = selected_entities.copy()  # Copy the list to avoid modifying the input directly

    # Handle 'All CEX' and 'All LSTs' special cases
    if 'All CEX' in selected_entities:
        all_selected_entities = [entity for entity in all_selected_entities if entity not in KNOWN_LST + ['All LSTs']]
        for cex in reversed(KNOWN_CEX):
            if cex not in all_selected_entities:
                all_selected_entities.append(cex)

    if 'All LSTs' in selected_entities:
        all_selected_entities = [entity for entity in all_selected_entities if entity not in KNOWN_CEX + ['All CEX']]
        for lst in reversed(KNOWN_LST):
            if lst not in all_selected_entities:
                all_selected_entities.append(lst)

    all_selected_entities = [entity for entity in all_selected_entities if entity not in ['All CEX', 'All LSTs']]

    # Initialize selected_order if it's None
    if selected_order is None:
        selected_order = []

    # Update the order of selections
    new_order = []
    for entity in reversed(all_selected_entities):
        if entity in selected_order:
            selected_order.remove(entity)
        new_order.append(entity)
    new_order = new_order+selected_order

    # Limit the number of selections
    if len(new_order) > MAX_SELECTIONS:
        deselected_count = len(new_order) - MAX_SELECTIONS
        deselected = new_order[-deselected_count:]
        all_selected_entities = [entity for entity in all_selected_entities if entity not in deselected]
        new_order = new_order[:len(new_order) - deselected_count]
        
    

    return all_selected_entities, new_order
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
