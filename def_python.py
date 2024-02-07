import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import math

months_abbrev = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def import_csv(folder,filename):
    df = pd.read_csv(f'{folder}{filename}.csv', encoding='utf-8', encoding_errors='replace')
    return df

def custom_slice(value,extract):
    if '-' in value:
        if value.index('-') != 3:
            return value
        elif extract == 'rack':
            return value[:3]
        elif extract == 'rack-bay':
            return value[:7]
        elif extract == 'level':
            return value[-2:]
    else:
        return 'dummy'

def each_user_activity_month_chart(df,month_serie):
    for i, m in enumerate(month_serie):
        x = df[df['Comp_Month'] == m]
        x = x.pivot_table(index=['Comp.by'], columns='Activity', values='TO Number.', aggfunc='count')
        x = x.fillna(0)
        title = f'Count PO of each activity, each user in {months_abbrev[m-1]}'
        fig = plt.figure(figsize=(25,10),dpi=300)
        fig.suptitle(title)
        fig.subplots_adjust(hspace=0.8, wspace=0.2)
        ax = x.plot.bar(ax=plt.axes(), edgecolor = "white",linewidth=1.0)
        plt.show()

def each_user_activity_chart(df):
    x = df.pivot_table(index=['Comp.by'], columns='Activity', values='TO Number.', aggfunc='count')
    x = x.fillna(0)
    title = f'Count PO of each activity, each user'
    fig = plt.figure(figsize=(25,10),dpi=300)
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.8, wspace=0.2)
    ax = x.plot.bar(ax=plt.axes(), edgecolor = "white",linewidth=1.0)
    plt.show()

def get_newcol_created_datetime(df,user):
    df = df[df['Comp.by'] == user].sort_values('Comp._Datetime')
    # Calculate the time difference between consecutive rows
    time_diff = df['Comp._Datetime'].diff()
    # Create a mask for the first row of each shift
    mask = (time_diff.dt.total_seconds() / 3600) > 10

    # Calculate the beginning of the hour for each row
    beginning_of_hour = df['Comp._Datetime'].dt.floor('H')
    # Create the 'created_datetime' column and set it to NaN
    df['created_datetime'] = pd.NaT
    # Create the 'created_datetime' column and set the beginning of the hour for the first row of each day
    df.loc[~mask, 'created_datetime'] = df['Comp._Datetime'].shift(1)[~mask]
    df.loc[mask, 'created_datetime'] = beginning_of_hour[mask]
    df.loc[df.index[0], 'created_datetime'] = beginning_of_hour.iloc[0]
    return df

def add_newcol_created_datetime(df,user_unique):
    for i, user in enumerate(user_unique):
        if i == 0:
            df_x = get_newcol_created_datetime(df,user)
        else:
            df_y = get_newcol_created_datetime(df,user)
            df_x = pd.concat([df_x, df_y], axis=0, ignore_index = False)
    df = df_x.sort_index()
    return df

###############################################
## Log section

def user_log(df,user):
    user_log = df[df['User Name'] == user].sort_values(by='Datetime')
    user_log['Trace\Log Text'] = user_log['Trace\Log Text'].str.lower()
    user_log = user_log.reset_index(drop=True)
    return(user_log)

def save_user_log(df,userlist):
    for user in userlist:
        User_log = user_log(df,user)
        filename = f'Data/Users_log_noPy/{user}_log.csv'
        User_log.to_csv(filename)

# Get the first login timestamp of each day
def first_TimeOfDay(user_log):
    # Find 'Log Event' == 'PROGRAM_START' first occurance of each day
    sys_start_rows = user_log[user_log['Log Event'] == 'PROGRAM_START']
    first_sys_start_each_day = sys_start_rows.groupby(user_log['Datetime'].dt.date).first().reset_index(drop=True)
    first_sys_start_each_day = first_sys_start_each_day.loc[:,['Date', 'Time', 'User Name', 'Datetime']]
    return(first_sys_start_each_day)

def Only0_To_1Sequence(df):
    # Excluded TO_log column that begin with '8'
    df = df[~df['TO_log'].astype(str).str.startswith('8')]

    mask1 = (df['status'] == 0) & (df['status'].shift(-1) == 1)
    mask2 = (df['status'] == 1) & (df['status'].shift(1) == 0)
    df = df[mask1 | mask2]
    return df


###############################################
## Graph function
def create_graph(graph_pattern=1):
    node_labels = None
    node_colors = None
    if graph_pattern==1 or graph_pattern==2:
        edges = pd.read_csv(f'Data\graph\edgelist_graph{graph_pattern}.csv')
        pos_info = pd.read_csv(f'Data\graph\position_graph{graph_pattern}.csv')
    else:
        return print('There is no graph_pattern which you have defined.')

    pos = {row['node']: (row['x'], row['y']) for _, row in pos_info.iterrows()}
    graph = nx.Graph()
    graph = nx.from_pandas_edgelist(
        edges,
        edge_key="edge_key",
        edge_attr=["weight"],
        create_using=nx.MultiGraph(),
        )
    if graph_pattern==1:
        nx.draw(graph, 
            pos,
            with_labels=True,
            node_color="red",
            node_size=20,
            font_color="black",
            font_size=5,
            font_family="Times New Roman", 
            font_weight="bold",
            #width=5
            )
    else:
        node_labels = {row['node']: row['label'] if not pd.isna(row['label']) else None for _, row in pos_info.iterrows() if not pd.isna(row['label'])}
        node_colors = {row['node']: row['color'] for _, row in pos_info.iterrows()}

        nx.draw(graph, 
        pos,
        labels=node_labels,
        node_color=[node_colors[node] for node in graph.nodes()],
        node_size=20,
        font_color="black",
        font_size=5,
        font_family="Times New Roman", 
        font_weight="bold",
        #width=5
        )

    plt.margins(0.2)
    dpi = 300
    plt.savefig(f'Data\graph\graph{graph_pattern}.png', dpi=dpi)
    plt.show()
    return graph,pos,node_labels,node_colors

def get_shortest_path (graph, sequence_of_nodes, same_I_O = True, back_to_out = True): #sequence_of_nodes = start to last node

    # Initialize variables to keep track of the total weight and the current node
    total_weight = 0
    current_node = sequence_of_nodes[0]
    path_sequence = []

    # Iterate through the sequence of nodes to find the shortest path step by step
    for next_node in sequence_of_nodes[1:]:
        shortest_path = nx.shortest_path(graph, source=current_node, target=next_node, weight='weight') #method string, optional (default = ‘dijkstra’)
        path_sequence.append(shortest_path)
        path_weight = 0

        for i in range(len(shortest_path) - 1):
            try:
                edge_key = shortest_path[i] + '_to_' + shortest_path[i + 1]
                path_weight += graph[shortest_path[i]][shortest_path[i + 1]][edge_key]['weight']
            except:
                edge_key = shortest_path[i+1] + '_to_' + shortest_path[i]
                path_weight += graph[shortest_path[i]][shortest_path[i + 1]][edge_key]['weight']

        total_weight += path_weight
        current_node = next_node

    if same_I_O == True:
        if back_to_out == True:
            # Calculate the total weight of the path from the last node back to the starting node
            last_to_start_path = nx.shortest_path(graph, source=current_node, target=sequence_of_nodes[0], weight='weight')
            if sequence_of_nodes[-1] != 'I_O':
                path_sequence.append(last_to_start_path)

            path_weight = 0
            for i in range(len(last_to_start_path) - 1):
                try:
                    edge_key = last_to_start_path[i] + '_to_' + last_to_start_path[i+1]
                    path_weight += graph[last_to_start_path[i]][last_to_start_path[i+1]][edge_key]['weight']
                except:
                    edge_key = last_to_start_path[i+1] + '_to_' + last_to_start_path[i]
                    path_weight += graph[last_to_start_path[i]][last_to_start_path[i+1]][edge_key]['weight']
        elif back_to_out == False:
            return total_weight,path_sequence
    else:
        print('need to add code for the exit node when it is not the same as start node!')

    total_weight += path_weight

    print(f"Total Weight: {total_weight}")
    print(path_sequence)
    return total_weight,path_sequence

def visualize_travel(graph, position, node_labels, node_colors, total_weight, path_sequence, graph_pattern=1):
    path_sequence_flat = [item for sublist in path_sequence for item in sublist]
    #dijkstra_subgraph = graph.subgraph(path_sequence_flat)
    if graph_pattern==1:
        nx.draw(graph, 
                position,
                with_labels=True,
                node_color="red",
                node_size=20,
                font_color="black",
                font_size=5,
                font_family="Times New Roman", 
                font_weight="bold",
                #width=5
                )
    elif graph_pattern==2:
        nx.draw(graph, 
        position,
        labels=node_labels,
        node_color=[node_colors[node] for node in graph.nodes()],
        node_size=20,
        font_color="black",
        font_size=5,
        font_family="Times New Roman", 
        font_weight="bold",
        #width=5
        )
    else:
        return print('There is no graph_pattern which you have defined.')
        
    #dijkstra_edges = list(dijkstra_subgraph.edges())
    dijkstra_edges = []

    # Iterate through the nodes and add edges
    for i in range(len(path_sequence_flat) - 1):
        dijkstra_edges.append((path_sequence_flat[i], path_sequence_flat[i+1]))

    nx.draw_networkx_edges(graph, position, edgelist=dijkstra_edges, edge_color='blue', width=4)

    bottom_start_x = -3
    bottom_start_y = -10
    space_btw_line = 2.2
    fontsize = 8

    bottom_left_text1 = 'Total weight = '+str(total_weight)+' m'
    bottom_left_text2 = f'Total targeted visited bins including I/O point = {len(path_sequence)}'
    plt.text(bottom_start_x, bottom_start_y, bottom_left_text1, fontsize=fontsize, color="black")
    plt.text(bottom_start_x, bottom_start_y+space_btw_line, bottom_left_text2, fontsize=fontsize, color="black")
    
    plt.margins(0.2)
    dpi = 300
    plt.savefig(f'Data\graph\graph{graph_pattern}_with_dijkstra_path.png', dpi=dpi)
    plt.show()