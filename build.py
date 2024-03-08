#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import argparse

from datetime import datetime

def set_google_credentials(CONFIG, GOOGLE_CREDENTIALS):
    try:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS']
    except:
        print(f"setting google credentials as global variable...")
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CONFIG \
        + GOOGLE_CREDENTIALS or input("No Google API credendials file provided." 
        + "Please specify path now:\n")
        
set_google_credentials("./config/","google-creds.json")

TESTING = False
def set_up_testing_env(arguments):
    global TESTING
    try:
        get_ipython().__class__.__name__
        sys.argv = arguments
        TESTING = True
        print(f"manually set attributes: {str(arguments)}")
        print(f"testing: {str(TESTING)}")
    except:
        pass

if not os.path.isdir("data"):
    os.mkdir("data")
    
set_up_testing_env([""])

parser = argparse.ArgumentParser(description="Timing.pics Builder")

parser.add_argument("-r", "--replace",
                    action="store_true",
                    help="Replace locally stored dataframes with freshley synced one")
args = parser.parse_args()
    
print(f"Overwriting local dfs: {str(args.replace)}")
OVERWRITE_KNOWN_DATAFRAMES = args.replace


# In[ ]:


def slot_to_time(slot):
    timestamp = 1606824023 + slot * 12
    dt_object = datetime.utcfromtimestamp(timestamp)
    formatted_time = dt_object.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time

def slot_to_hour_minute(slot):
    timestamp = 1606824023 + slot * 12
    dt_object = datetime.utcfromtimestamp(timestamp)
    formatted_time = dt_object.strftime("%Y-%m-%d %H:%M")
    return formatted_time

def slot_to_day(slot):
    timestamp = 1606824023 + slot * 12
    dt_object = datetime.utcfromtimestamp(timestamp)
    formatted_time = dt_object.strftime("%Y-%m-%d")
    return formatted_time

def slot_to_month(slot):
    timestamp = 1606824023 + slot * 12
    dt_object = datetime.utcfromtimestamp(timestamp)
    formatted_time = dt_object.strftime("%Y-%m")
    return formatted_time

def slot_to_week_of_year(slot):
    timestamp = 1606824023 + slot * 12
    dt_object = datetime.utcfromtimestamp(timestamp)
    formatted_week = dt_object.strftime("%Y-W%U")
    return formatted_week

def get_dt_from_ts(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

def get_time_in_curr_slot(ts, slot):
    return (ts - (1654824023000 + (slot-1-4e6)*12000) - 12000) / 1000 

# Constants calculated once
offset = 1654824023000 - (4e6 - 1) * 12000 - 12000

def get_time_in_curr_slot2(ts, slot):
    return (ts - (offset + slot * 12000)) / 1000

def only_last_x_d(df, x):
    df['day'] = pd.to_datetime(df['day'])
    max_day = df['day'].max()
    thirty_days_ago = max_day - pd.Timedelta(days=x)
    filtered_df = df[(df['day'] > thirty_days_ago) & (df['day'] <= max_day)]
    return filtered_df


# In[ ]:


#try: 
#    dfbids = pd.read_parquet("data/dfbids.parquet")
#except:
#    dfbids = pd.DataFrame([0], columns=["slot"])


# In[ ]:


#OVERWRITE_KNOWN_DATAFRAMES = True


# In[ ]:


def load_new_df_entries():
    print("processing df")
    try:
        if OVERWRITE_KNOWN_DATAFRAMES:
            raise
        df = pd.read_parquet("data/df.parquet")
        print(f"{len(df.slot.unique())} unique validators locally found")
    except:
        df = pd.DataFrame([0], columns=["slot"])
    query = f"""SELECT distinct A.*, B.label FROM 
    (SELECT slot, validator_pubkey, block_number FROM (
      SELECT slot, validator_pubkey, block_number from `ethereum-data-nero.ethdata.beaconchain`
      where slot > {df.slot.max()}
      order by slot desc limit 2592000)) A
    LEFT JOIN
    (
      SELECT pubkey, label FROM `ethereum-data-nero.ethdata.beaconchain_validators_db`
    ) B on A.validator_pubkey = B.pubkey
    order by slot desc
    """
    df2 = pd.read_gbq(query)
    print(f"{len(df2.slot.unique())} unique validators loaded from bq")
    
    df2 = df2[df2["slot"] != 0]
    df = pd.concat([df,df2], ignore_index=True).drop_duplicates()
    df = df[df["slot"] != 0]
    print(f"{len(df.slot.unique())} unique validators in dataset")
    return df

def load_new_dfbids_entries():
    print("processing dfbids")
    try:
        if OVERWRITE_KNOWN_DATAFRAMES:
            raise
        df = pd.read_parquet("data/dfbids.parquet")
        print(f"{len(df.slot.unique())} unique validators locally found")
    except:
        df = pd.DataFrame([0], columns=["slot"])
    query = f"""SELECT distinct A.*, B.timestamp_ms FROM
    (SELECT validator, slot, block_hash, value_gwei, proposer_pubkey as pubkey 
    FROM `ethereum-data-nero.ethdata.ethereum_slot_db` WHERE TIMESTAMP_TRUNC(date, DAY) > TIMESTAMP("2023-06-01") 
    and slot > {df.slot.max()} and validator not like "0x%"
    ) A LEFT JOIN 
    (
      SELECT DISTINCT timestamp_ms, block_hash FROM (
      SELECT min(timestamp_ms) timestamp_ms, block_hash 
      FROM `ethereum-data-nero.eth.mevboost_all_bids` where slot > {df.slot.max()} group by block_hash 
      UNION ALL
      SELECT min(timestamp_ms) timestamp_ms, block_hash 
      FROM `ethereum-data-nero.eth.mevboost_all_bids_archive_0` where slot > {df.slot.max()} group by block_hash 
    )
    )B on A.block_hash = B.block_hash
    """
    df2 = pd.read_gbq(query)
    df2 = df2[df2["slot"] != 0]
    print(f"{len(df2.slot.unique())} unique validators loaded from bq")
    df = pd.concat([df,df2], ignore_index=True).drop_duplicates()
    df = df[df["slot"] != 0]
    print(f"{len(df.slot.unique())} unique validators in dataset")
    return df

def load_seen_blocks_dfs():
    print("processing seen slots (pace)")
    try:
        if OVERWRITE_KNOWN_DATAFRAMES:
            raise
        df = pd.read_parquet("data/seenblocks.parquet")
        print(f"{len(df.slot.unique())} unique blocks locally found")
    except:
        df = pd.DataFrame([0], columns=["slot"])
        
    df2 = pd.read_gbq(f"""
        SELECT distinct cl_client, slot from ethereum-data-nero.ethdata.beaconchain_pace where slot > {df.slot.max()}
        """)
    df2 = df2[df2["slot"] != 0]
    df = pd.concat([df,df2], ignore_index=True).drop_duplicates()
    df = df[df["slot"] != 0]
    print(f"{len(df.slot.unique())} unique blocks in dataset")
    return df


def load_mevboost_blocks_dfs():
    print("processing mevboost slots")
    try:
        if OVERWRITE_KNOWN_DATAFRAMES:
            raise
        df = pd.read_parquet("data/mevboostblocks.parquet")
        if len(df) == 0:
            raise
        print(f"{len(df.slot.unique())} unique blocks locally found")
    except:
        df = pd.DataFrame([0], columns=["slot"])
        
    df2 = pd.read_gbq(f"""SELECT distinct slot, builder FROM `ethereum-data-nero.ethdata.ethereum_slot_db`  
WHERE TIMESTAMP_TRUNC(date, DAY) >= TIMESTAMP("2023-06-01") and slot > {df.slot.min()} 
and builder is not null""")
    df2 = df2[df2["slot"] != 0]
    df = pd.concat([df,df2], ignore_index=True).drop_duplicates()
    df = df[df["slot"] != 0]
    print(f"{len(df.slot.unique())} unique mevboost blocks in dataset")
    return df


df = load_new_df_entries()
df.to_parquet("data/df.parquet", index=None)

dfbids = load_new_dfbids_entries()
dfbids.to_parquet("data/dfbids.parquet", index=None)

dfs = load_seen_blocks_dfs()
dfs.to_parquet("data/seenblocks.parquet", index=None)

mdf = load_mevboost_blocks_dfs()
mdf.to_parquet("data/mevboostblocks.parquet", index=None)

with open("last_updated.txt", "w") as file:
    file.write(slot_to_hour_minute(max(df.slot)))


# In[ ]:


gaming_pubkeys = []
def preprocess_gamers_nongamers(dfbids2, quantile=0.90):
    global gaming_pubkeys
    gg = dfbids2.groupby("pubkey")["seconds in slot"].median().reset_index().sort_values("seconds in slot", ascending=False)
    gg = gg[gg["seconds in slot"] > gg["seconds in slot"].quantile(quantile)]
    gaming_pubkeys = gg["pubkey"].tolist()
    dfbids2["gamer"] = dfbids2['pubkey'].isin(gg['pubkey'])
    dfbids2 = dfbids2[["day", "gamer", "slot", "value_gwei"]].groupby(["day", "gamer"]).agg({"slot": "count", "value_gwei": "median"}).reset_index()
    dfbids2["value_gwei"] = dfbids2["value_gwei"] / 1e9
    dfbids2['day'] = pd.to_datetime(dfbids2['day'])
    gamers = dfbids2[dfbids2['gamer']]
    non_gamers = dfbids2[~dfbids2['gamer']]
    gamers_weekly = gamers.resample('W', on='day').mean().reset_index()
    non_gamers_weekly = non_gamers.resample('W', on='day').mean().reset_index()
    return (
        gamers_weekly,
        non_gamers_weekly,
        gamers,
        non_gamers,
    )
#(
#    gamers,
#    non_gamers,
#    gamers_weekly,
#    non_gamers_weekly,
#) = preprocess_gamers_nongamers(dfbids


total_stake = 32

def get_apy(df):
    df["yearly_value"] = df["value_gwei"]*2.92
    df["clreward"] = 0.9597986027554286
    df["total"] = df["value_gwei"] + df["clreward"]
    df["apy"] = (df["total"] / total_stake) * 100
    return df

def prep_df_for_missed_bar_chart(df):
    gg = df[["label", "slot", "missed"]].groupby("label").agg({"slot": "count", "missed": "sum"}).reset_index().sort_values("slot", ascending=False).reset_index(drop=True)
    gg["total slots"] = len(range(df.slot.min(), df.slot.max()))
    gg["label_slot_share"] = gg["slot"] / gg["total slots"]
    gg = gg.rename(columns={"slot": "label_slots", "missed": "label_missed"})
    gg["actual missed"] = gg["label_missed"] / gg["label_slots"]
    gg["potential missed"] = gg["label_slot_share"]
    gg = gg[["label", "actual missed", "potential missed"]]
    gg["label"]= gg["label"].apply(lambda x: x.split(" <")[0])    
    return gg


# In[ ]:


if TESTING:
    dfbids = pd.read_parquet("data/dfbids.parquet")
    df = pd.read_parquet("data/df.parquet")    
    dfs = pd.read_parquet("data/seenblocks.parquet")    
    mdf = pd.read_parquet("data/mevboostblocks.parquet")


# In[ ]:


unique_solostakers = 0
solo_staker_string = ""
def preprocess_df(df, dfbids):
    print("preprocessing df")
    global unique_solostakers, solo_staker_string
    df["day"] = df["slot"].apply(lambda x: slot_to_day(x))
    #df["month"] = df["slot"].apply(lambda x: slot_to_month(x))
    #df["month"] = df["slot"].apply(lambda x: slot_to_month(x))
    #df["week"] = df["slot"].apply(lambda x: slot_to_week_of_year(x))
    #df["time"] = df["slot"].apply(lambda x: slot_to_time(x))
    df["label"] = df["label"].apply(lambda x: x[0].upper() + x[1:].lower() if x != None else x)
    df["missed"] = df["block_number"].apply(lambda x: 1 if x == 0 else 0)
    df = df[df["slot"] >= min(dfbids.slot)]  
    df = df[df["slot"] <= max(dfbids.slot)]  
    
    if "solo_staker" not in df.columns:
        df["solo_staker"] = df.label.apply(lambda x: 1 if x is not None and x.endswith(".eth") else 0)
    df.loc[df['solo_staker'] == 1, 'label'] = 'Solo Stakers'


    unique_solostakers = len(df[df["label"] == "Solo Stakers"].validator_pubkey.unique())
    solo_staker_string = f"Solo Stakers <sub>({unique_solostakers} unique)</sub>"
    df.replace("Solo Stakers", solo_staker_string, inplace=True)
    
    largest = df.dropna().groupby("label")["slot"].count().reset_index().sort_values("slot", ascending=False).label[0:30].tolist()
    df["label"] = df["label"].apply(lambda x: "`Other`" if x not in largest else x)
    largest.remove(solo_staker_string)
    df.set_index("label", inplace=True)
    df = df.loc[[solo_staker_string]+largest]
    df.reset_index(inplace=True)
    return df

def preprocess_dfbids(dfbids, aggregate=True):
    global unique_solostakers, solo_staker_string
    print("preprocessing dfbids")
    dfbids.drop_duplicates(inplace=True)
    dfbids.dropna(inplace=True)
    print("slot to day...")
    dfbids["day"] = dfbids["slot"].apply(lambda x: slot_to_day(x))
    print("slot to month...")
    dfbids["month"] = dfbids["slot"].apply(lambda x: slot_to_month(x))
    dfbids["validator"] = dfbids["validator"].apply(lambda x: x[0].upper() + x[1:].lower() if x != None else x)
    print("handle solo stakers...")
    if "solo_staker" not in dfbids.columns:
        dfbids["solo_staker"] = dfbids.validator.apply(lambda x: 1 if x is not None and x.endswith(".eth") else 0)

    #unique_solostakers = len(df[df["label"] == "Solo Stakers"].validator_pubkey.unique())    
    #solo_staker_string = f"Solo Stakers <sub>({unique_solostakers} unique)</sub>"
    dfbids.replace("Solo Stakers", solo_staker_string, inplace=True)
    if not aggregate:
        print("aggregation off")
        #dfbids.reset_index(inplace=True)
        pass
    else:
        dfbids.loc[dfbids['solo_staker'] == 1, 'validator'] = solo_staker_string#'Solo Stakers'
        print(f"exclude small entities... {len(dfbids)}")
        largest = dfbids.dropna().groupby("validator")["slot"].count().reset_index().sort_values("slot", ascending=False).validator[0:20].tolist()
        dfbids["validator"] = dfbids["validator"].apply(lambda x: "`Other`" if x not in largest else x)
        dfbids.set_index("validator", inplace=True)
        dfbids = dfbids.loc[[solo_staker_string]+largest]
        if solo_staker_string in largest:
            largest.remove(solo_staker_string)
            
        dfbids.reset_index(inplace=True)
        dfbids=dfbids[dfbids["validator"] != "`Other`"]
        print(f"exclude small entities finished. {len(dfbids)}")
    print("retrieve seconds in slot...")
    #dfbids["seconds in slot"] = dfbids[["timestamp_ms", "slot"]].apply(lambda x: get_time_in_curr_slot(*x), axis=1)
    offset = 1654824023000 - (4e6 - 1) * 12000 - 12000
    dfbids["seconds in slot"] = (dfbids["timestamp_ms"] - (offset + dfbids["slot"] * 12000)) / 1000    #identical = dfbids['seconds in slot'].equals(dfbids['seconds in slot2'])
    #print("retrieve seconds in slot...")
    #dfbids["seconds in slot2"] = dfbids[["timestamp_ms", "slot"]].apply(lambda x: get_time_in_curr_slot2(*x), axis=1)

    #identical = dfbids['seconds in slot'].equals(dfbids['seconds in slot2'])
    #print("The columns are identical:", identical)
    print(str(len(dfbids[dfbids["validator"] == solo_staker_string])), " solo stakers")
    return dfbids



df = preprocess_df(df, dfbids)
dfbids2 = preprocess_dfbids(dfbids.copy(), False)
dfbids = preprocess_dfbids(dfbids)
print("preprocessing done")


# In[ ]:


DAYS = 14

def generate_gamer_advantage_avg(dfbids):
    dfbids2=dfbids.copy()
    gg = dfbids2.groupby("pubkey")["seconds in slot"].median().reset_index().sort_values("seconds in slot", ascending=False)
    gg = gg[gg["seconds in slot"] > gg["seconds in slot"].quantile(0.90)]
    dfbids2["gamer"] = dfbids2['pubkey'].isin(gg['pubkey']).copy()
    categories = ['Non-Gamers', 'Gamers']
    nongamer, gamer = dfbids2.groupby("gamer").value_gwei.median()/1e9
    values = [round(nongamer,4), round(gamer,4)]

    # Create the bar chart
    fig = go.Figure(data=[
        go.Bar(x=categories, y=values, text=values, textposition='auto', marker_color=["rgba(0, 128, 0, 0.7)","rgba(255,0,0,0.7)"],
              hovertemplate="%{y:,2f} ETH on avg for %{x:,2f} <extra></extra>", textfont=dict(color="white") )
    ])

    # Update the layout for a more infographic style to match your existing chart
    fig.update_layout(
        title=f'Average Earnings</br></br><sub>last {DAYS} days</sub>',
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#0a0a0a",
        margin={"t": 90, "b": 0, "r": 50, "l": 0},
        title_font_color='white',
        font=dict(family="Ubuntu Mono", color='white', size=16),
        showlegend=False,
        dragmode=False,
        xaxis=dict(
            title=None,
            color='white',
            showline=True,
            showgrid=False,
            fixedrange=True,
            linecolor='white',
            linewidth=2,
            ticks='outside',
            tickfont=dict(size=18, color='white')
        ),
        yaxis=dict(
            title='ETH',
            color='white',
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.3)',
            showline=True,
            fixedrange=True,
            showticklabels=True,
            linecolor='white',
            linewidth=2,
            ticks='outside',
            tickfont=dict(size=16, color='white')
        ),
        height=400,
        #width=300
    )

    # Add annotation to emphasize the difference
    fig.add_annotation(
        x=0.5,
        y=max(values),
        xref="paper",
        yref="y",
        text=f"{(values[1] - values[0]) / values[0] * 100:.2f}% higher",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40,
        font=dict(family="Ubuntu Mono", color='white', size=18)
    )
    return True, fig

if TESTING:
    fig=generate_gamer_advantage_avg(only_last_x_d(dfbids, DAYS))[1]
    # Show the plot
    fig.show()


# In[ ]:


def postprocess_dfbids(dfbids):
    print(len(dfbids))

    def weighted_sample(df, n_samples=None, weight_column='seconds in slot'):
        """Performs weighted sampling from a dataframe."""
        # Calculate weights as the inverse of the 'seconds in slot' values
        weights = 1 / df[weight_column]
        # Normalize weights to sum to 1
        weights /= weights.sum()
        # Determine the number of samples, defaulting to the size of the dataframe if not provided
        n_samples = n_samples or len(df)
        # Perform weighted random sampling
        return df.sample(n=n_samples, weights=weights, replace=False)

    def reduce_datapoints(gg):
        gg = gg[gg["seconds in slot"] > 0].sort_values("seconds in slot").reset_index(drop=True)

        high = float(gg.iloc[len(gg.index)//30*29]["seconds in slot"])
        print(high)

        low = gg[gg["seconds in slot"] <= min(1.2, high)].copy()
        mid = gg[(gg["seconds in slot"] > min(1.2, high)) & (gg["seconds in slot"] < high)].copy()
        high = gg[gg["seconds in slot"] >= high].copy()

        # Apply weighted sampling to each subset if they exceed the size threshold
        if len(low) > 100000:
            low = weighted_sample(low, n_samples=len(low)//2)
        elif len(low) > 80000:
            low = weighted_sample(low, n_samples=len(low)//2) 
        elif len(low) > 50000:
            low = weighted_sample(low, n_samples=len(low)//2) 
        elif len(low) > 30000:
            low = weighted_sample(low, n_samples=len(low)//2) 
        if len(mid) > 100000:
            mid = weighted_sample(mid, n_samples=len(mid)//2)
        elif len(mid) > 80000:
            mid = weighted_sample(mid, n_samples=len(mid)//2)
        elif len(mid) > 30000:
            mid = weighted_sample(mid, n_samples=len(mid)//2)
        elif len(mid) > 50000:
            mid = weighted_sample(mid, n_samples=len(mid)//2) 

        # For the 'high' subset, you might want to keep all data points 
        # or apply a different sampling strategy due to their rarity and importance.

        return pd.concat([low, mid, high], ignore_index=True)    

    # Main process to reduce data points for each validator
    newdf = pd.DataFrame(columns=dfbids.columns)
    hover_threshold = 1.2
    for validator in dfbids.validator.unique():
        print(validator)
        gg = dfbids[dfbids["validator"] == validator].copy()
        gg['hovertext'] = gg.apply(lambda x: f"Slot: {x['slot']}<br>Sec. in Slot: {x['seconds in slot']:.4}" if x["seconds in slot"] > hover_threshold else '', axis=1)
        
        #if len(gg) > 10:
        #    #gg = reduce_datapoints(gg)
        #    #gg = gg.drop_duplicates(subset=["seconds in slot"], keep="last")   
        #    gg = gg.sample(frac=1).drop_duplicates(subset=["seconds in slot"]).reset_index(drop=True)
        newdf = pd.concat([newdf.copy(), gg], ignore_index=True)
    newdf = newdf.sample(frac=1).drop_duplicates(subset=["validator","seconds in slot"]).reset_index(drop=True)
    print(len(dfbids))
    dfbids = newdf.copy()
    print(len(dfbids))
    return dfbids

dfbids = postprocess_dfbids(dfbids.copy())
df2 = prep_df_for_missed_bar_chart(only_last_x_d(df.copy(), 60))
print("postprocessing done")


# In[ ]:


def get_avg_reorged_of_missed(_df):
    df = _df.copy()
    df = df[df["block_number"] == 0]  
    df["reorged"] = df["slot"].isin(dfs[dfs["cl_client"] != "missed"].slot.tolist())
    df = df.groupby(["label", "reorged"])["slot"].count().unstack().reset_index()
    df.columns = ["label", "n reorged", "reorged"]
    df["avg_reorg"] = df["reorged"]/(df["n reorged"]+ df["reorged"])
    df["label"] = df["label"].apply(lambda x: x.split(" <")[0])
    df.fillna(0, inplace=True)
    df = df[["label", "avg_reorg"]].set_index("label").to_dict()["avg_reorg"]
    return df

def get_avg_mevboost_of_missed(_df):
    df = _df.copy()
    df["mevboost"] = df["slot"].isin(mdf.slot)
    mevboost_validators = set(df[df["mevboost"] == True].validator_pubkey.tolist())
    print(len(mevboost_validators))
    df["mevboost"] = df["validator_pubkey"].isin(mevboost_validators)
    df = df[df["block_number"] == 0]  
    df = df.groupby(["mevboost", "label"])["slot"].count().reset_index().pivot(index="label", columns="mevboost").reset_index()
    df.columns = ["label", "non", "yes"]
    df["avg_mevboost"] = df["yes"]/(df["non"]+ df["yes"])
    df["label"] = df["label"].apply(lambda x: x.split(" <")[0])
    df.fillna(0, inplace=True)
    df = df[["label", "avg_mevboost"]].set_index("label").to_dict()["avg_mevboost"]
    return df


# In[ ]:


def generate_gamer_advantage_lines(gamers_weekly, non_gamers_weekly):
    fig = go.Figure()

    gamers_weekly['rolling_apy'] = gamers_weekly['apy'].rolling(window=2).mean()
    non_gamers_weekly['rolling_apy'] = non_gamers_weekly['apy'].rolling(window=2).mean()

    # Calculate the cumulative sum for gamers and non-gamers
    gamers_weekly['cumulative_value_gwei'] = gamers_weekly['value_gwei'].cumsum()
    non_gamers_weekly['cumulative_value_gwei'] = non_gamers_weekly['value_gwei'].cumsum()

    # Add line traces for gamers and non-gamers daily values
    fig.add_trace(go.Scatter(
        x=gamers_weekly['day'], 
        y=gamers_weekly['rolling_apy'],
        mode='lines',  # This makes it a line chart
        name='Gamers', 
        line=dict(color='#FF0000'),
        hoverinfo='x+y',
        customdata=["Gamer" if i else "Non-Gamer" for i in gamers_weekly['gamer']],
        hovertemplate="%{customdata}: %{y:,.2f} APY<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=non_gamers_weekly['day'], 
        y=non_gamers_weekly['rolling_apy'],
        mode='lines',  # This makes it a line chart
        name='Non-Gamers', 
        line=dict(color='#008000'),
        hoverinfo='x+y',
        customdata=["Gamer" if i else "Non-Gamer" for i in non_gamers_weekly['gamer']],
        hovertemplate="%{customdata}: %{y:,.2f} APY<extra></extra>"
    ))

    # Add line traces for gamers and non-gamers cumulative values
    fig.add_trace(go.Scatter(
        x=gamers_weekly['day'], 
        y=gamers_weekly['cumulative_value_gwei'],
        mode='lines',  # This makes it a line chart
        name='Gamers', 
        line=dict(color='#FF0000', dash='dot'),  # dash type can be 'solid', 'dot', 'dash', 'longdash', etc.
        visible=False,
        hoverinfo='x+y',
        customdata=["Gamer" if i else "Non-Gamer" for i in gamers_weekly['gamer']],
        hovertemplate="%{customdata}: %{y:,.2f} ETH<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=non_gamers_weekly['day'], 
        y=non_gamers_weekly['cumulative_value_gwei'],
        mode='lines',  # This makes it a line chart
        name='Non-Gamers', 
        line=dict(color='#008000', dash='dot'),
        visible=False,
        hoverinfo='x+y',
        customdata=["Gamer" if i else "Non-Gamer" for i in non_gamers_weekly['gamer']],
        hovertemplate="%{customdata}: %{y:,.2f} ETH<extra></extra>"
    ))

    for trace in fig.data:
        trace.hoverlabel = dict(font_size=14, font_family='Ubuntu Mono')

    # Add titles and labels
    fig.update_layout(
        barmode='group', 
        title='Profitability',
        xaxis_title=None, 
        yaxis_title='APY',
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#0a0a0a",
        dragmode=False,
        xaxis=dict(
            fixedrange=True,
            gridcolor="rgba(255,255,255,0.3)",
        ),
        yaxis=dict(
            fixedrange=True,
            gridcolor="rgba(255,255,255,0.3)",
        ),
        height=400,
        #width=600,
        hovermode="x unified",
        margin={"t": 70, "b": 0, "r": 50, "l": 0},
        title_font_color='white',
        font=dict(family="Ubuntu Mono", color='white', size=14),
        legend=dict(
            x=0.8,  # X position of the legend (fraction of the total width)
            y=1,    # Y position of the legend (fraction of the total height)
            traceorder="normal",
            font=dict(
                family="Ubuntu Mono",
                size=12,
                color="white"
            ),
            bgcolor="#0a0a0a",
            #bordercolor="White",
            #borderwidth=1
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"visible": [True, True, False, False]},
                              {"yaxis.title": "APY",
                               "title": "Timing games profitability",
                               "legend.x": 0.8}],  # Update this line
                        label="Daily View",
                        method="update"  # Change to 'update'
                    ),
                    dict(
                        args=[{"visible": [False, False, True, True]},
                              {"yaxis.title": "ETH",
                               "title": "Cumulative Proposer Revenues",
                               "legend.x": 0.05}],  # Update this line
                        label="Cumulative View",
                        method="update"  # Change to 'update'
                    )
                ]),
                pad={"r": 0, "t": 0, "l": 0, "b": 0},
                showactive=False,
                x=0.6,
                xanchor="left",
                y=1.2,
                yanchor="top",
                bgcolor="#0a0a0a",
                bordercolor="#ffffff",
                
                font = dict(color='#ffffff'),
            ),
        ],
        
    )
    return True, fig



if TESTING:
    _, _, gamers_weekly, non_gamers_weekly = preprocess_gamers_nongamers(dfbids, 0.85)
    gamers_weekly, non_gamers_weekly = only_last_x_d(gamers_weekly, 60), only_last_x_d(non_gamers_weekly, 60)
    gamer_weekely= get_apy(gamers_weekly)
    non_gamers_weekly= get_apy(non_gamers_weekly)

    fig=generate_gamer_advantage_lines(gamers_weekly, non_gamers_weekly)[1]
    # Show the plot
    fig.show()


# In[ ]:


order = []

def generate_gaming_slot_bars(dfbids2):
    global order
    ff = dfbids2[["slot", "validator", "value_gwei", "pubkey", "seconds in slot", "solo_staker"]].copy()
    ff.loc[ff['solo_staker'] == 1, 'validator'] = 'Solo Stakers'
    largest = ff.dropna().groupby("validator")["slot"].count().reset_index().sort_values("slot", ascending=False).validator[0:20].tolist()
    ff = ff[ff["validator"].isin(largest)]
    ff["game"] = ff["seconds in slot"] > ff["seconds in slot"].quantile(0.90)
    ff = ff[["validator", "slot", "game"]].groupby(["validator", "game"]).count().reset_index()
    sum_slots = ff.groupby(['validator', 'game'])['slot'].sum().reset_index()
    pivot_slots = sum_slots.pivot(index='validator', columns='game', values='slot').reset_index()
    pivot_slots.columns = ['validator', 'slots_false', 'slots_true']
    pivot_slots['avg_true_vs_false'] = pivot_slots['slots_true'] / pivot_slots['slots_false']
    order = pivot_slots.sort_values("avg_true_vs_false", ascending=False).validator.tolist()
    ff.set_index("validator", inplace=True)
    ff = ff.loc[order]
    ff.reset_index(inplace=True)
    
    validators = ff['validator'].unique()
    # Create subplots without shared_xaxes
    fig = make_subplots(rows=len(validators), cols=1, vertical_spacing=0.01)

    for idx, validator in enumerate(validators):
        data = ff[ff['validator'] == validator]

        # Calculate the total slots for the validator
        total_slots = data['slot'].sum()

        # Separate data for gaming and not gaming, then calculate percentages
        gaming_slots = data[data['game'] == True]['slot'].sum()
        not_gaming_slots = data[data['game'] == False]['slot'].sum()

        gaming_percentage = (gaming_slots / total_slots) * 100 if total_slots else 0
        not_gaming_percentage = (not_gaming_slots / total_slots) * 100 if total_slots else 0

        # Add trace for not gaming
        fig.add_trace(
            go.Bar(
                x=[not_gaming_percentage],
                y=[validator],
                orientation='h',
                name='Not Gaming',
                marker=dict(color='rgba(0, 128, 0, 0.7)'),
                hovertemplate="%{x:.2f}% not playing<extra></extra>"  # Custom hover template
            ), row=idx+1, col=1
        )

        # Add trace for gaming
        fig.add_trace(
            go.Bar(
                x=[gaming_percentage],
                y=[validator],
                orientation='h',
                name='Gaming',
                marker=dict(color='rgba(255,0,0,0.7)'),
                hovertemplate="%{x:.2f}% playing timing games<extra></extra>"  # Custom hover template
            ), row=idx+1, col=1
        )

        # Update xaxis properties only for the last subplot
        if idx == len(validators) - 1:
            fig.update_xaxes(title='% of Total Slots',showgrid=False, row=idx+1, col=1)
        else:
            fig.update_xaxes(showticklabels=False, row=idx+1, col=1)
            
        fig.update_xaxes(
            range=[-18, 101],  # This adds space to the beginning and end of the x-axis. Adjust as necessary.
            domain=[0, 0.9],# Light color for the gridlines
            row=idx+1, 
            col=1,
            showgrid=False,
            fixedrange=True
        )
        fig.update_yaxes(
            showgrid=False,
            fixedrange=True
        )

    fig.update_layout(
        title='Timing Games - Share per Validator',
        barmode='stack',
        hovermode="y unified",
        showlegend=False,
        bargap=0,
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#0a0a0a",
        margin={"t": 70, "b": 0, "r": 50, "l": 0},
        title_font_color='white',
        font=dict(family="Ubuntu Mono", color='white', size=15),
        #width=480
        height=500,
        xaxis=dict(showgrid=False, fixedrange=True),
        yaxis=dict(showgrid=False, fixedrange=True),
        dragmode=False
        
    )

    return True, fig


def generate_missed_slot_bars(df):
    
    ee = df.groupby(["label", "missed"])["slot"].count().unstack().reset_index().rename(columns={0: "not missed", 1: "missed"})
    ee["label"] = ee["label"].apply(lambda x: x.split(" <")[0]) 
    ee["avg"] = ee["missed"]/(ee["missed"] + ee["not missed"])
    ee.set_index("label", inplace=True)
    
    ee = ee.loc[order]
    ee.reset_index(inplace=True)
    
    validators = ee['label'].unique()
    fig = make_subplots(rows=len(validators), cols=1, vertical_spacing=0.01)

    for idx, validator in enumerate(validators):
        data = ee[ee['label'] == validator]

        # Calculate the total slots for the validator
        total_slots = data['missed'].sum() + data['not missed'].sum()

        # Calculate percentages
        missed_percentage = (data['missed'].sum() / total_slots) * 100 if total_slots else 0
        not_missed_percentage = (data['not missed'].sum() / total_slots) * 100 if total_slots else 0

        # Add trace for missed slots (place it before the not missed slots)
        fig.add_trace(
            go.Bar(
                x=[missed_percentage],
                y=[validator],
                orientation='h',
                name='Missed',
                marker=dict(color='rgba(255,0,0,0.7)'),
                hovertemplate="%{x:.2f}% missed slots<extra></extra>"  # Custom hover template
            ), row=idx+1, col=1
        )

        # Add trace for not missed slots
        fig.add_trace(
            go.Bar(
                x=[not_missed_percentage],
                y=[validator],
                orientation='h',
                name='Not Missed',
                marker=dict(color='rgba(0, 128, 0, 0.7)'),
                hovertemplate="%{x:.2f}% not missed slots<extra></extra>"  # Custom hover template
            ), row=idx+1, col=1
        )

        # Update xaxis properties only for the last subplot
        if idx == len(validators) - 1:
            fig.update_xaxes(title='% of Total Slots', range=[0, 5], showgrid=False,  gridcolor='rgba(255, 255, 255, 0.3)', row=idx+1, col=1)
        else:
            fig.update_xaxes(showticklabels=False, range=[0, 5], row=idx+1, col=1)  # Hide tick labels for non-last subplots

        fig.update_xaxes(
            range=[-0.9, 6],  # This adds space to the beginning and end of the x-axis. Adjust as necessary.
            domain=[0, 0.9],# Light color for the gridlines
            row=idx+1, 
            col=1,
            showgrid=False,
            fixedrange=True
        )
        fig.update_yaxes(
            showgrid=False,
            fixedrange=True
        )
    # Update layout to match existing style
    fig.update_layout(
        title='Missed Slots - Share per Validator',
        barmode='stack',
        hovermode="y unified",
        showlegend=False,
        dragmode=False,
        bargap=0,
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#0a0a0a",
        margin={"t": 70, "b": 0, "r": 50, "l": 0},
        title_font_color='white',
        font=dict(family="Ubuntu Mono", color='white', size=15),
        #width=480,
        height=500,
        xaxis=dict(showgrid=False, fixedrange=True),
        yaxis=dict(showgrid=False, fixedrange=True)
    )
    
    return True, fig

if TESTING:
    fig1 = generate_gaming_slot_bars(dfbids2)[1]
    fig = generate_missed_slot_bars(df)[1]
    fig1.show() 
    fig.show() 


# In[ ]:


df = df[df["slot"] >= min(dfbids.slot)]  
df = df[df["slot"] <= max(dfbids.slot)]  


# In[ ]:


def generate_missed_slot_over_time_chart(df, entity):
    gg = df[df["label"] == entity]
    missed = gg[gg["block_number"] == 0]
    non_missed = gg[gg["block_number"] != 0]
    missed = missed.groupby(["day", "label"])["slot"].count().reset_index().rename(columns={"slot": "missed"})
    non_missed = non_missed.groupby(["day", "label"])["slot"].count().reset_index()
    gg = pd.merge(non_missed, missed, on=["day", "label"], how="left")
    gg["missed_per"] = gg["missed"] / gg["slot"] * 100
    gg['day'] = pd.to_datetime(gg['day'])  # Convert the 'day' column to datetime

    # Calculate weekly average
    gg.set_index('day', inplace=True)
    weekly_avg = gg['missed_per'].resample('W').mean()  # 'W' stands for weekly
        
    weekly_labels = weekly_avg.index.strftime("%Y-%m-%d")


    x_min = gg.index.min()
    x_max = gg.index.max()

    # Create a stacked bar chart
    fig = go.Figure()

    gg.reset_index(inplace=True)  # Reset the index for plotting
    gg["daystr"] = gg["day"].apply(lambda x: str(x)[0:10])
    fig.add_trace(go.Bar(
        x=gg['day'],
        y=gg['missed_per'],
        name="Missed %",
        customdata=gg[['daystr', 'missed_per']]
    ))

    # Add a line chart for weekly average with correct custom data
    fig.add_trace(go.Scatter(
        x=weekly_avg.index,
        y=weekly_avg,
        mode='lines',
        name='Weekly avg.',
        line=dict(color='yellow', width=2, dash='dash'),
        customdata=list(zip(weekly_labels, weekly_avg))  # Updated custom data for weekly average
    ))
    
    # Update hover templates
    bar_hovertemplate = "Date: %{customdata[0]}<br>Missed slots: %{customdata[1]:.2f}%<extra></extra>"
    line_hovertemplate = "Week of: %{customdata[0]}<br>Avg. Missed slots: %{customdata[1]:.2f}%<extra></extra>"
    fig.data[0].hovertemplate = bar_hovertemplate
    fig.data[1].hovertemplate = line_hovertemplate
    fig.update_traces(hoverlabel=dict(font_size=16, font_family="Ubuntu Mono"))  # Adjust font size and family

    # Update the layout for the bar chart
    fig.update_layout(
        title=f'Missed Slots {entity}',
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#0a0a0a",
        margin={"t": 70, "b": 0, "r": 50, "l": 0},
        title_font_color='white',
        font=dict(family="Ubuntu Mono", color='white', size=15),
        showlegend=False,
        dragmode = False,
        xaxis=dict(
            title=None,
            color='white',
            showline=True,
            showgrid=True,
            fixedrange=True,
            gridcolor='rgba(255, 255, 255, 0.3)',
            showticklabels=True,
            linecolor='white',
            linewidth=2,
            ticks='outside',
            tickfont=dict(size=14, color='white'),
            range=[x_min, x_max],
            tickmode='linear',
            tick0=x_min,
            dtick=1209600000  # Two weeks in milliseconds
        ),
        yaxis=dict(
            title="% missed",
            color='white',
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.3)',
            showline=True,
            fixedrange=True,
            showticklabels=True,
            linecolor='white',
            linewidth=2,
            ticks='outside',
            tickfont=dict(size=14, color='white')
        ),
        height=250,
    )
    return True, fig

if TESTING:
    fig = generate_missed_slot_over_time_chart(df,  "Lido")[1]
    fig.show()


# In[ ]:


def generate_missed_market_share_chart(gg):

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=gg['label'],
        y=gg['actual missed']*100,
        name='Missed Slots',
        marker=dict(color='#FF0000'),
        hovertemplate="%{y:.2f}% missed slots<extra></extra>"
    ))

    # Add 'potential missed' bars with a different pattern (using opacity)
    fig.add_trace(go.Bar(
        x=gg['label'],
        y=gg['potential missed']*100,
        name='Market Share',

        marker=dict(color='rgba(0, 128, 0, 0.7)', pattern_shape="+",),  
        hovertemplate="%{y:.2f}% market share<extra></extra>"
    ))

    # Update layout to match the style of your previous chart
    fig.update_layout(
        title='Missed Slots vs Market Share<br><sub>last 60 days</sub>',
        xaxis_title=None,
        yaxis_title='%',
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#0a0a0a",
        dragmode=False,
        yaxis=dict(type="log", gridcolor="rgba(255,255,255,0.3)", fixedrange=True),
        xaxis=dict(fixedrange=True),
        font=dict(family="Ubuntu Mono", color='white', size=14),
        barmode='group',
        height=600,
        bargap=0.4,
        legend=dict(x=0.82, y=1.2, traceorder="normal", font=dict(family="Ubuntu Mono", size=14, color="white"), bgcolor="#0a0a0a", bordercolor="White", borderwidth=1)
    )
    return True, fig

if TESTING:
    fig = generate_missed_market_share_chart(df2)[1]
    fig.show()


# In[ ]:


def generate_missed_reorged_chart(gg, reorg_ratio):

    # Calculate the missed slots due to reorgs and non-reorgs
    missed_due_to_reorg = gg['actual missed'] * gg['label'].map(reorg_ratio)
    missed_due_to_non_reorg = gg['actual missed'] - missed_due_to_reorg

    # Initialize the figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=gg['label'],
        y=missed_due_to_reorg*100,
        name='Proposed but Reorged',
        marker=dict(color='#FF0000'),
        hovertemplate="%{y:.2f}% missed slots due to reorgs<extra></extra>"
    ))


    # Add trace for missed slots due to non-reorgs
    fig.add_trace(go.Bar(
        x=gg['label'],
        y=missed_due_to_non_reorg*100,
        name='Truly Missed',
        marker=dict(color='rgba(0, 128, 0, 0.7)'),
        hovertemplate="%{y:.2f}% missed slots due to non-reorgs<extra></extra>"
    ))

    # Add trace for missed slots due to reorgs
    
    # Update layout
    fig.update_layout(
        title='Missed vs Reorged <span style="font-size:12px;">last 60 days</span>',
        xaxis_title=None,
        dragmode=False,
        yaxis_title='%',
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#0a0a0a",
        yaxis=dict(gridcolor="rgba(255,255,255,0.3)", fixedrange=True),
        xaxis=dict(fixedrange=True),
        font=dict(family="Ubuntu Mono", color='white', size=14),
        barmode='stack',  # Change to stack to stack bars
        height=400,
        bargap=0.4,
        legend=dict(x=0.82, y=1.3, traceorder="normal", font=dict(family="Ubuntu Mono", size=14, color="white"), bgcolor="#0a0a0a", bordercolor="White", borderwidth=1)
    )
    return True, fig

# Use the function with your data and reorg_ratio
if TESTING:
    reorg_ratio = get_avg_reorged_of_missed(df)
    fig = generate_missed_reorged_chart(df2.iloc[:25], reorg_ratio)[1]
    fig.show()


# In[ ]:


def generate_missed_mevboost_chart(gg, mevboost_ratio):

    # Calculate the missed slots due to reorgs and non-reorgs
    missed_due_mevboost = gg['actual missed'] * gg['label'].map(mevboost_ratio)
    missed_due_non_mevboost = gg['actual missed'] - missed_due_mevboost

    # Initialize the figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=gg['label'],
        y=missed_due_mevboost*100,
        name='Missed (MEV-boost)',
        marker=dict(color='#FF0000'),
        hovertemplate="%{y:.2f}% missed slots of MEV-Boost validators<extra></extra>"
    ))


    # Add trace for missed slots due to non-reorgs
    fig.add_trace(go.Bar(
        x=gg['label'],
        y=missed_due_non_mevboost*100,
        name='Missed (Non-MEV-Boost)',
        marker=dict(color='rgba(0, 128, 0, 0.7)'),
        hovertemplate="%{y:.2f}% missed slots of Vanilla Builders<extra></extra>"
    ))

    # Add trace for missed slots due to reorgs
    
    # Update layout
    fig.update_layout(
        title='Missed Slots: MEV-Boost vs Vanilla Builders <span style="font-size:12px;">last 60 days</span>',
        xaxis_title=None,
        dragmode=False,
        yaxis_title='%',
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#0a0a0a",
        yaxis=dict(gridcolor="rgba(255,255,255,0.3)", fixedrange=True),
        xaxis=dict(fixedrange=True),
        font=dict(family="Ubuntu Mono", color='white', size=14),
        barmode='stack',  # Change to stack to stack bars
        height=400,
        bargap=0.4,
        legend=dict(x=0.82, y=1.3, traceorder="normal", font=dict(family="Ubuntu Mono", size=14, color="white"), bgcolor="#0a0a0a", bordercolor="White", borderwidth=1)
    )
    return True, fig

# Use the function with your data and reorg_ratio
if TESTING:
    mevboost_ratio = get_avg_mevboost_of_missed(df)
    fig = generate_missed_mevboost_chart(df2.iloc[:25], mevboost_ratio)[1]
    fig.show()


# In[ ]:


def generate_time_in_slot_scatter(dfbids, entity):
    gg = dfbids[dfbids["validator"] == entity]
    gg = gg[gg["seconds in slot"] >= 0]
    if len(gg) == 0:
        return False, []

    # Assuming 'gg' is your pandas DataFrame
    gg['day'] = pd.to_datetime(gg['day'])  # Convert the 'day' column to datetime
    gg["daystr"] = gg["day"].apply(lambda x: str(x)[0:10])
    
    x_min = gg['day'].min()
    x_max = gg['day'].max()
    
    min_size = 2
    max_size = 5
    gg['normalized_size'] = (gg['seconds in slot'] - gg['seconds in slot'].min()) / (gg['seconds in slot'].max() - gg['seconds in slot'].min())
    gg['scaled_size'] = gg['normalized_size'] * (max_size - min_size) + min_size
    gg['scaled_size'] = gg['scaled_size'].apply(lambda x: max([x,5]))

    opacity = np.where(gg['seconds in slot'] < 2, 0.6, 0.8)  # 0.5 opacity for dots with 'seconds in slot' < 2, 1 for others


    fig = go.Figure(data=go.Scattergl(
        x=gg['day'],
        y=gg['seconds in slot'],
        mode='markers',
        marker=dict(
            color=gg['seconds in slot'],
            colorscale='RdYlBu_r',  # Red-Yellow-Blue reversed scale
            size=gg['scaled_size'],  # Use the scaled sizes
            sizemode='diameter',
            line=dict(width=0),
            opacity=opacity
        ),
    ))

    # Set the y-axis to a logarithmic scale and define the range
    # Starting from a small number close to zero, like 0.01
    fig.update_layout(plot_bgcolor="#0a0a0a", paper_bgcolor = "#0a0a0a")
    
    fig.update_traces(hoverlabel=dict(font_size=16, font_family="Ubuntu Mono"))  # Adjust font size and family

    fig.update_traces(customdata=gg[['hovertext']], hovertemplate="%{customdata[0]}<extra></extra>")
    
    fig.update_layout(
        title=f"Bids selected by {entity}",
        title_font_color='white',
        font_color='white',
        font=dict(family="Ubuntu Mono", color='white', size=15),
        coloraxis_showscale=False,
        margin={"t":70,"b":0,"r":50,"l":0},
        dragmode = False,
        xaxis=dict(
            title=None,
            color='white',
            showline=True,
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.3)',        
            showticklabels=True,
            linecolor='white',
            linewidth=2,
            ticks='outside',
            fixedrange=True,
            tickfont=dict(
                size=14,
                color='white'
            ),
            range=[x_min, x_max],
            tickmode='linear',
            tick0=x_min,
            dtick=1209600000
        ),
        yaxis=dict(
            title="seconds in slot",
            color='white',
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.3)',
            showline=True,
            showticklabels=True,
            linecolor='white',
            linewidth=2,
            ticks='outside',
            fixedrange=True,
            tickfont=dict(
                size=14,
                color='white'
            )
        ),
        yaxis_range=[0.01, 3],
        height=250
    )
    return True, fig

if TESTING:
    fig = generate_time_in_slot_scatter(dfbids, "Lido")
    fig[1].show()


# In[ ]:


missed_slot_over_time_chart = {}
generate_time_in_slot_scatter_chart = {}

found3, gaming_slot_bars = generate_gaming_slot_bars(dfbids2)
found4, missed_slot_bars = generate_missed_slot_bars(df)

_, _, gamers_weekly, non_gamers_weekly = preprocess_gamers_nongamers(dfbids, 0.85)
gamers_weekly, non_gamers_weekly = only_last_x_d(gamers_weekly, 60), only_last_x_d(non_gamers_weekly, 60)
gamer_weekely= get_apy(gamers_weekly)
non_gamers_weekly= get_apy(non_gamers_weekly)

found5, gamer_advantage_lines = generate_gamer_advantage_lines(gamers_weekly, non_gamers_weekly)
found6, gamer_advantage_avg = generate_gamer_advantage_avg(dfbids)
found7, missed_market_share_chart = generate_missed_market_share_chart(df2)

reorg_ratio = get_avg_reorged_of_missed(df)
found8, missed_reorged_chart = generate_missed_reorged_chart(df2.iloc[:25], reorg_ratio)

mevboost_ratio = get_avg_mevboost_of_missed(df)
found9, missed_mevboost_chart = generate_missed_mevboost_chart(df2.iloc[:25], mevboost_ratio)
    
    
entities = df.label.unique()
for entity in entities[:20]:
    found1, chart1 = generate_missed_slot_over_time_chart(df, entity)
    found2, chart2 = generate_time_in_slot_scatter(dfbids, entity)
    if found1 and found2:
        print(entity)
        missed_slot_over_time_chart[entity] = chart1
        generate_time_in_slot_scatter_chart[entity] = chart2

                
with open('missed_slot_over_time_chart.pkl', 'wb') as f:
    pickle.dump(missed_slot_over_time_chart, f)
    
with open('time_in_slot_scatter_chart.pkl', 'wb') as f:
    pickle.dump(generate_time_in_slot_scatter_chart, f)
    
with open('gamer_bars.pkl', 'wb') as f:
    pickle.dump(gaming_slot_bars, f)
    
with open('missed_slot_bars.pkl', 'wb') as f:
    pickle.dump(missed_slot_bars, f)
    
with open('gamer_advantage_lines.pkl', 'wb') as f:
    pickle.dump(gamer_advantage_lines, f)
    
with open('gamer_advantage_avg.pkl', 'wb') as f:
    pickle.dump(gamer_advantage_avg, f)

with open('missed_market_share_chart.pkl', 'wb') as f:
    pickle.dump(missed_market_share_chart, f)

with open('missed_reorged_chart.pkl', 'wb') as f:
    pickle.dump(missed_reorged_chart, f)
    
with open('missed_mevboost_chart.pkl', 'wb') as f:
    pickle.dump(missed_mevboost_chart, f)


# In[ ]:


def curtime() -> str:
    return datetime.strftime(datetime.now(), "%m-%d|%H:%M:%S")

with open("logs.txt", "a") as file:
    file.write(f"{curtime()} | timing_games build successful")

