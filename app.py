import dash
from dash import Dash, html, dcc, exceptions
import dash_daq as daq
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import joblib
from functools import lru_cache
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import gaussian_kde
import base64
import io
import json
from datetime import datetime


# Cached data loading
@lru_cache(maxsize=None)
def load_data():
    return pd.read_csv("test.csv")


# Load data and model
test = load_data()
TARGET = 'Survived'
labels = ['0% to <10%', '10% to <20%', '20% to <30%', '30% to <40%', '40% to <50%',
          '50% to <60%', '60% to <70%', '70% to <80%', '80% to <90%', '90% to <100%']
pipeline = joblib.load("model.pkl")

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = 'Titanic Analysis'


# -------------------------
# Helper Functions for Data Filters
# -------------------------
def create_dropdown_options(series):
    """Create dropdown options from a pandas Series."""
    return [{'label': i, 'value': i} for i in sorted(series.unique())]


def create_dropdown_value(series):
    """Return a sorted list of unique values from the Series."""
    return sorted(series.unique().tolist())


def create_slider_marks(values):
    """Create slider marks given a list of values."""
    return {i: {'label': str(i)} for i in values}


def status_in_class(df, klass, status):
    """Return the number of passengers in a class that survived or perished."""
    class_count = 0
    status_count = 0

    for srv, kls in zip(df.Survived, df.Class):
        if kls == klass:  # the class that we want?
            class_count += 1

            if srv == status:  # survived or perished in that class?
                status_count += 1

    return class_count, status_count


def get_feature_importance(model):
    """Get feature importance from the trained model."""
    try:
        return model.feature_importances_
    except:
        return None



def get_passenger_stories():
    """Return interesting passenger stories for the stories section."""
    return [
        {
            'name': 'Margaret "Molly" Brown',
            'class': 'First',
            'survived': True,
            'story': 'Known as "The Unsinkable Molly Brown", she helped organize the lifeboat evacuation and survived the disaster. She took command of Lifeboat 6 and helped row for 7 hours.',
            'age': 45,
            'gender': 'Female'
        },
        {
            'name': 'John Jacob Astor IV',
            'class': 'First',
            'survived': False,
            'story': 'One of the wealthiest passengers aboard, he helped his pregnant wife into a lifeboat but perished himself. His body was recovered with $2,440 in his pocket.',
            'age': 47,
            'gender': 'Male'
        },
        {
            'name': 'Rose Amélie Icard',
            'class': 'First',
            'survived': True,
            'story': 'A French maid who survived by clinging to a piece of debris in the freezing water for hours before being rescued.',
            'age': 38,
            'gender': 'Female'
        },
        {
            'name': 'Benjamin Guggenheim',
            'class': 'First',
            'survived': False,
            'story': 'A wealthy businessman who famously said "We are dressed in our best and are prepared to go down like gentlemen" before the ship sank.',
            'age': 46,
            'gender': 'Male'
        },
        {
            'name': 'Isidor Straus',
            'class': 'First',
            'survived': False,
            'story': 'Co-owner of Macy\'s department store, he refused to board a lifeboat while women and children were still on board. His wife Ida chose to stay with him.',
            'age': 67,
            'gender': 'Male'
        },
        {
            'name': 'Eva Hart',
            'class': 'Second',
            'survived': True,
            'story': 'A 7-year-old girl who survived with her mother. Her father perished. She later became one of the last living survivors and spoke publicly about the disaster.',
            'age': 7,
            'gender': 'Female'
        },
        {
            'name': 'Charles Joughin',
            'class': 'Third',
            'survived': True,
            'story': 'The ship\'s baker who survived by staying in the water for hours. He claimed the alcohol he consumed helped him stay warm.',
            'age': 32,
            'gender': 'Male'
        },
        {
            'name': 'Masabumi Hosono',
            'class': 'Second',
            'survived': True,
            'story': 'The only Japanese passenger aboard. He survived but faced criticism in Japan for not going down with the ship.',
            'age': 42,
            'gender': 'Male'
        },
        {
            'name': 'Violet Jessop',
            'class': 'Crew',
            'survived': True,
            'story': 'A stewardess who survived the Titanic disaster and later survived the sinking of the Britannic in 1916. She was known as "Miss Unsinkable".',
            'age': 24,
            'gender': 'Female'
        },
        {
            'name': 'Thomas Andrews',
            'class': 'First',
            'survived': False,
            'story': 'The ship\'s designer who was last seen in the first-class smoking room, staring at a painting. He helped many passengers into lifeboats.',
            'age': 39,
            'gender': 'Male'
        }
    ]


def save_prediction_to_csv(user_input, prediction, probability):
    """Save user prediction data to a CSV file for tracking and analysis."""
    try:
        # Create prediction record with timestamp
        prediction_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'passenger_class': user_input['Class'],
            'gender': user_input['Gender'],
            'age': user_input['Age'],
            'siblings_spouses': user_input['Sibsp'],
            'parents_children': user_input['Parch'],
            'fare': user_input['Fare'],
            'embark_town': user_input['Embark town'],
            'who': user_input['Who'],
            'adult_male': user_input['Adult male'],
            'deck': user_input['Deck'],
            'alone': user_input['Alone'],
            'predicted_survival': prediction,
            'survival_probability': probability,
            'prediction_confidence': abs(probability - 0.5) * 2  # Distance from 0.5
        }
        
        # Try to read existing file or create new one
        try:
            existing_data = pd.read_csv('user_predictions.csv')
            new_data = pd.concat([existing_data, pd.DataFrame([prediction_record])], ignore_index=True)
        except FileNotFoundError:
            new_data = pd.DataFrame([prediction_record])
        
        # Save to CSV
        new_data.to_csv('user_predictions.csv', index=False)
        return True
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return False


# -------------------------
# Helper Functions for Graphs
# -------------------------
def style_layout(title, bgcolor='rgba(255,242,204,100)'):
    """Return a standardized layout for graphs."""
    return dict(
        title=title,
        font_family='Tahoma',
        plot_bgcolor=bgcolor
    )


def create_age_density_plot(df, view_mode='combined'):
    """
    Create density plots for passenger ages by gender.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'Age' and 'Gender' columns
    view_mode : str
        'combined' for overlaid plots, 'separated' for side-by-side subplots
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Density plot figure
    """
    # Filter out missing ages
    df_age = df[df['Age'].notna()].copy()
    
    if df_age.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No age data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(**style_layout('Distribution of Passenger Ages'))
        return fig
    
    # Separate by gender
    male_data = df_age[df_age['Gender'] == 'Male']['Age'].values
    female_data = df_age[df_age['Gender'] == 'Female']['Age'].values
    all_data = df_age['Age'].values
    
    # Count passengers with known age
    n_male = len(male_data)
    n_female = len(female_data)
    n_all = len(all_data)
    
    # Create age range for plotting
    age_min = max(0, df_age['Age'].min() - 5)
    age_max = df_age['Age'].max() + 5
    age_range = np.linspace(age_min, age_max, 500)
    
    # Colors matching the reference images
    color_male = 'rgb(100, 120, 150)'  # Blue-grey
    color_female = 'rgb(200, 120, 80)'  # Orange-brown
    color_all = 'rgba(200, 200, 200, 0.5)'  # Light gray
    
    if view_mode == 'separated':
        # Create subplots side by side
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Male passengers', 'Female passengers'),
            shared_yaxes=True,
            shared_xaxes=True
        )
        
        # Calculate densities for male subplot
        if n_male > 1:  # Need at least 2 points for KDE
            kde_male = gaussian_kde(male_data)
            density_male = kde_male(age_range)
            # Scale so area equals number of passengers
            # The area under a KDE curve is approximately 1, so we multiply by n_male
            scaled_density_male = density_male * n_male
        elif n_male == 1:
            # Single point - create a simple spike
            scaled_density_male = np.zeros_like(age_range)
            idx = np.argmin(np.abs(age_range - male_data[0]))
            scaled_density_male[idx] = n_male * 10  # Scale for visibility
        else:
            scaled_density_male = np.zeros_like(age_range)
        
        # Calculate densities for female subplot
        if n_female > 1:  # Need at least 2 points for KDE
            kde_female = gaussian_kde(female_data)
            density_female = kde_female(age_range)
            # Scale so area equals number of passengers
            scaled_density_female = density_female * n_female
        elif n_female == 1:
            # Single point - create a simple spike
            scaled_density_female = np.zeros_like(age_range)
            idx = np.argmin(np.abs(age_range - female_data[0]))
            scaled_density_female[idx] = n_female * 10  # Scale for visibility
        else:
            scaled_density_female = np.zeros_like(age_range)
        
        # Add all passengers background (gray) to both subplots
        if n_all > 1:
            kde_all = gaussian_kde(all_data)
            density_all = kde_all(age_range)
            scaled_density_all = density_all * n_all
            
            # Add to male subplot
            fig.add_trace(
                go.Scatter(
                    x=age_range, y=scaled_density_all,
                    fill='tozeroy',
                    mode='lines',
                    name='all passengers',
                    line=dict(color=color_all, width=1),
                    showlegend=True,
                    legendgroup='all'
                ),
                row=1, col=1
            )
            
            # Add to female subplot
            fig.add_trace(
                go.Scatter(
                    x=age_range, y=scaled_density_all,
                    fill='tozeroy',
                    mode='lines',
                    name='all passengers',
                    line=dict(color=color_all, width=1),
                    showlegend=False,  # Don't show again
                    legendgroup='all'
                ),
                row=1, col=2
            )
        
        # Add male distribution
        if n_male > 0:
            fig.add_trace(
                go.Scatter(
                    x=age_range, y=scaled_density_male,
                    fill='tozeroy',
                    mode='lines',
                    name='males',
                    line=dict(color=color_male, width=2),
                    showlegend=True,
                    legendgroup='male'
                ),
                row=1, col=1
            )
        
        # Add female distribution
        if n_female > 0:
            fig.add_trace(
                go.Scatter(
                    x=age_range, y=scaled_density_female,
                    fill='tozeroy',
                    mode='lines',
                    name='females',
                    line=dict(color=color_female, width=2),
                    showlegend=True,
                    legendgroup='female'
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_xaxes(
            title_text="passenger age (years)", 
            row=1, col=1,
            showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)'
        )
        fig.update_xaxes(
            title_text="passenger age (years)", 
            row=1, col=2,
            showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)'
        )
        fig.update_yaxes(
            title_text="scaled density", 
            row=1, col=1,
            showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)'
        )
        fig.update_yaxes(
            row=1, col=2,
            showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)'
        )
        
        fig.update_layout(
            title=f'Age Distribution by Gender (n={n_all})',
            font_family='Tahoma',
            plot_bgcolor='white',
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.25,
                xanchor='center',
                x=0.5
            ),
            height=500,
            margin=dict(b=80)  # Add bottom margin to accommodate legend and x-axis label
        )
        
    else:  # combined view
        fig = go.Figure()
        
        # Add male distribution
        if n_male > 1:  # Need at least 2 points for KDE
            kde_male = gaussian_kde(male_data)
            density_male = kde_male(age_range)
            # Scale so area equals number of passengers
            scaled_density_male = density_male * n_male
            
            fig.add_trace(
                go.Scatter(
                    x=age_range, y=scaled_density_male,
                    fill='tozeroy',
                    mode='lines',
                    name='male',
                    line=dict(color=color_male, width=2),
                    fillcolor=color_male
                )
            )
        elif n_male == 1:
            # Single point - create a simple spike
            scaled_density_male = np.zeros_like(age_range)
            idx = np.argmin(np.abs(age_range - male_data[0]))
            scaled_density_male[idx] = n_male * 10  # Scale for visibility
            fig.add_trace(
                go.Scatter(
                    x=age_range, y=scaled_density_male,
                    fill='tozeroy',
                    mode='lines',
                    name='male',
                    line=dict(color=color_male, width=2),
                    fillcolor=color_male
                )
            )
        
        # Add female distribution (stacked/overlaid)
        if n_female > 1:  # Need at least 2 points for KDE
            kde_female = gaussian_kde(female_data)
            density_female = kde_female(age_range)
            # Scale so area equals number of passengers
            scaled_density_female = density_female * n_female
            
            fig.add_trace(
                go.Scatter(
                    x=age_range, y=scaled_density_female,
                    fill='tozeroy',
                    mode='lines',
                    name='female',
                    line=dict(color=color_female, width=2),
                    fillcolor=color_female
                )
            )
        elif n_female == 1:
            # Single point - create a simple spike
            scaled_density_female = np.zeros_like(age_range)
            idx = np.argmin(np.abs(age_range - female_data[0]))
            scaled_density_female[idx] = n_female * 10  # Scale for visibility
            fig.add_trace(
                go.Scatter(
                    x=age_range, y=scaled_density_female,
                    fill='tozeroy',
                    mode='lines',
                    name='female',
                    line=dict(color=color_female, width=2),
                    fillcolor=color_female
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f'Age Distribution by Gender (n={n_all})',
            xaxis_title='age (years)',
            yaxis_title='scaled density',
            font_family='Tahoma',
            plot_bgcolor='white',
            showlegend=True,
            legend=dict(
                title_text='gender',
                yanchor='top',
                y=0.99,
                xanchor='right',
                x=0.99
            ),
            height=500,
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)')
        )
    
    return fig


# -------------------------
# Division 1: Header, Sample Record & Controls
# -------------------------
def create_container1():
    """Creates the header section with title, description, image, and sample record table with controls."""
    return html.Div([
        # Left side: Header, description, and image
        html.Div([
            html.H1("Titanic Analysis"),
            html.P("Explore the Titanic dataset, analyze passenger survival, and predict survival for new passengers."),
            html.Img(src="assets/titanic-sinking.png", className="header-image")
        ], className="container1-left"),

        # Right side: Sample record table with interactive controls
        html.Div([
            dcc.Graph(id='table'),
            html.Div([
                html.Label("Survival status", className='other-labels'),
                daq.BooleanSwitch(id='target-toggle', className='toggle', on=True, color="#FFBD59"),

                html.Label("Sort probability in ascending order", className='other-labels'),
                daq.BooleanSwitch(id='sort-toggle', className='toggle', on=True, color="#FFBD59"),

                html.Label("Number of records", className='other-labels'),
                dcc.Slider(
                    id='n-slider', min=5, max=20, step=1, value=10,
                    marks=create_slider_marks([5, 10, 15, 20])
                ),
            ], id="table-controls", className="table-controls")
        ], className="container1-right")
    ], id="container1", className="container1")


# -------------------------
# Division 2: Data Filters & Graphs
# -------------------------
def create_container2():
    """Creates the data filters and multiple graphs for analysis."""
    return html.Div([
        # Filters section
        html.Div([
            html.H3("Data Filters"),
            html.Label("Passenger Class", className='dropdown-labels'),
            dcc.Dropdown(
                id='class-dropdown',
                className='dropdown',
                multi=True,
                options=create_dropdown_options(test['Class']),
                value=create_dropdown_value(test['Class'])
            ),
            html.Label("Gender", className='dropdown-labels'),
            dcc.Dropdown(
                id='gender-dropdown',
                className='dropdown',
                multi=True,
                options=create_dropdown_options(test['Gender']),
                value=create_dropdown_value(test['Gender'])
            ),
            html.Label("Age Distribution View", className='dropdown-labels'),
            dcc.RadioItems(
                id='age-view-toggle',
                options=[
                    {'label': 'Combined', 'value': 'combined'},
                    {'label': 'Separated', 'value': 'separated'}
                ],
                value='combined',
                className='radio-items'
            )
        ], id="filters", className="filters"),

        # Graphs section
        html.Div([
            dcc.Loading(
                id="loading-age-histogram",
                type="circle",
                children=dcc.Graph(id="age-histogram")
            ),
            dcc.Loading(
                id="loading-class-barplot",
                type="circle",
                children=dcc.Graph(id="class-barplot")
            ),
            dcc.Loading(
                id="loading-probability-histogram",
                type="circle",
                children=dcc.Graph(id="histogram")
            ),
            dcc.Loading(
                id="loading-survival-barplot",
                type="circle",
                children=dcc.Graph(id="barplot")
            ),
            dcc.Loading(
                id="loading-scatter",
                type="circle",
                children=dcc.Graph(id="age-fare-scatter")
            )
        ], id="graphs", className="graphs")
    ], id="container2", className="container2")


# -------------------------
# Division 3: Predict Survival
# -------------------------
def create_container3():
    """Creates the survival prediction section with a form and result display."""
    return html.Div([
        html.H3("Predict Survival"),
        html.Div([
            html.Label("Passenger Class"),
            dcc.Dropdown(
                id="predict-class",
                options=create_dropdown_options(test['Class']),
                value=create_dropdown_value(test['Class'])[0]
            ),
            html.Label("Gender"),
            dcc.Dropdown(
                id='predict-gender',
                options=[{'label': g, 'value': g} for g in ['Male', 'Female']],
                value='Male'
            ),
            html.Label("Age"),
            dcc.Slider(
                id='predict-age', min=0, max=80, step=1, value=30,
                marks=create_slider_marks([0, 20, 40, 60, 80])
            ),
            html.Label("Siblings/Spouses Aboard"),
            dcc.Slider(
                id='predict-sibsp', min=0, max=8, step=1, value=0,
                marks=create_slider_marks([0, 2, 4, 6, 8])
            ),
            html.Label("Parents/Children Aboard"),
            dcc.Slider(
                id='predict-parch', min=0, max=6, step=1, value=0,
                marks=create_slider_marks([0, 2, 4, 6])
            ),
            html.Label("Fare"),
            dcc.Slider(
                id='predict-fare', min=0, max=500, step=10, value=50,
                marks=create_slider_marks([0, 100, 200, 300, 400, 500])
            ),
            html.Label("Embark Town"),
            dcc.Dropdown(
                id='predict-embark',
                options=create_dropdown_options(test['Embark town']),
                value=create_dropdown_value(test['Embark town'])[0]
            ),
            html.Button(id='predict-button', children="Predict", n_clicks=0, className="predict-button"),
            html.Div(id='predict-result', className="predict-result")
        ], id="predict-form", className="predict-form"),
    ], id="container3", className="container3")


# -------------------------
# Division 4: Feature Importance & User Prediction History
# -------------------------
def create_container4():
    """Creates the feature importance and user prediction history section."""
    return html.Div([
        html.H3("Model Analysis & User Predictions"),
        html.Div([
            # Feature Importance
            html.Div([
                html.H4("Feature Importance"),
                dcc.Loading(
                    id="loading-feature-importance",
                    type="circle",
                    children=dcc.Graph(id="feature-importance-plot")
                )
            ], className="feature-importance-section"),
            
            # User Prediction History
            html.Div([
                html.H4("User Prediction History"),
                html.P("Track all predictions made by users of this application:"),
                dcc.Loading(
                    id="loading-prediction-history",
                    type="circle",
                    children=dcc.Graph(id="prediction-history-table")
                ),
                html.Div(id="prediction-stats", className="prediction-stats")
            ], className="prediction-history-section")
        ], className="model-analysis-grid")
    ], id="container4", className="container4")





# -------------------------
# Division 5: Passenger Stories & Timeline
# -------------------------
def create_container5():
    """Creates the passenger stories and timeline section."""
    return html.Div([
        html.H3("Passenger Stories & Timeline"),
        html.Div([
            # Passenger Stories
            html.Div([
                html.H4("Notable Passengers"),
                html.Div(id="passenger-stories", className="passenger-stories")
            ], className="stories-section"),
            
            # Interactive Timeline
            html.Div([
                html.H4("Titanic Timeline"),
                html.Div(id="titanic-timeline", className="titanic-timeline")
            ], className="timeline-section")
        ], className="stories-timeline-grid")
    ], id="container5", className="container5")


# -------------------------
# Division 6: Export & Download
# -------------------------
def create_container6():
    """Creates the export and download functionality section."""
    return html.Div([
        html.H3("Export Data"),
        html.Div([
            html.Label("Select data to export:"),
            dcc.Checklist(
                id='export-options',
                options=[
                    {'label': 'Filtered Data', 'value': 'filtered'},
                    {'label': 'User Predictions History', 'value': 'predictions'}
                ],
                value=['filtered']
            ),
            html.Button(id='download-button', children="Download Data", className="download-button"),
            dcc.Download(id="download-dataframe-csv")
        ], className="export-section")
    ], id="container6", className="container6")


# -------------------------
# App Layout: Combining All Divisions
# -------------------------
app.layout = html.Div(
    id="container",
    children=[
        dcc.Store(id='app-load-trigger', data={'loaded': True}),
        create_container1(),
        create_container2(),
        create_container3(),
        create_container4(),
        create_container5(),
        create_container6()
    ]
)


# -------------------------
# Callback: Update Visualizations (Graphs and Table)
# -------------------------
@app.callback(
    [Output('age-histogram', 'figure'),
     Output('class-barplot', 'figure'),
     Output('histogram', 'figure'),
     Output('barplot', 'figure'),
     Output('age-fare-scatter', 'figure'),
     Output('table', 'figure'),
     Output('class-dropdown', 'value'),
     Output('gender-dropdown', 'value')],
    [Input('class-dropdown', 'value'),
     Input('gender-dropdown', 'value'),
     Input('target-toggle', 'on'),
     Input('sort-toggle', 'on'),
     Input('n-slider', 'value'),
     Input('age-view-toggle', 'value')]
)
def update_output(class_value, gender_value, target, ascending, n, age_view_mode):
    """
    Update all graphs and the sample records table based on filter selections.
    """
    dff = test.copy()

    # Apply filters immediately
    # Filter by 'Passenger Class' if not empty
    if class_value:
        dff = dff[dff['Class'].isin(class_value)]

    # Filter by 'Gender' if not empty
    if gender_value:
        dff = dff[dff['Gender'].isin(gender_value)]

    # If no data remains after filtering, display empty figures with a warning
    if dff.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title_text="No data available for the selected filters")
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, class_value, gender_value

    # --- Age Density Plot ---
    age_histogram = create_age_density_plot(dff, view_mode=age_view_mode)

    # --- Class Barplot ---
    count_1st, survived_1st = status_in_class(dff, 'First', 'Yes')
    count_2nd, survived_2nd = status_in_class(dff, 'Second', 'Yes')
    count_3rd, survived_3rd = status_in_class(dff, 'Third', 'Yes')
    pct_survived_1st = 100 * survived_1st / count_1st if count_1st else 0
    pct_perished_1st = 100 - pct_survived_1st if count_1st else 0
    pct_survived_2nd = 100 * survived_2nd / count_2nd if count_2nd else 0
    pct_perished_2nd = 100 - pct_survived_2nd if count_2nd else 0
    pct_survived_3rd = 100 * survived_3rd / count_3rd if count_3rd else 0
    pct_perished_3rd = 100 - pct_survived_3rd if count_3rd else 0

    class_barplot = go.Figure(data=[
        go.Bar(name='Survived', x=['1st', '2nd', '3rd'],
               y=[pct_survived_1st, pct_survived_2nd, pct_survived_3rd],
               texttemplate='%{y:.2f}%', textposition='auto', marker_color='#3BA27A'),
        go.Bar(name='Perished', x=['1st', '2nd', '3rd'],
               y=[pct_perished_1st, pct_perished_2nd, pct_perished_3rd],
               texttemplate='%{y:.2f}%', textposition='auto', marker_color='#FFBD59')
    ])
    class_barplot.update_layout(**style_layout(f'Passenger Survival Percentage by Class (n={len(dff)})'),
                                barmode='stack')

    # --- Probability Histogram ---
    histogram = px.histogram(
        dff, x='Probability', color=TARGET, marginal="box", nbins=30,
        opacity=0.6, color_discrete_sequence=['#FFBD59', '#3BA27A']
    )
    histogram.update_layout(**style_layout(f'Distribution of Predicted Probabilities (n={len(dff)})'))
    histogram.update_yaxes(title_text="Count")

    # --- Survival Rate Barplot by Binned Probabilities ---
    grouped = dff.groupby('Binned probability', as_index=False, observed=True)['Target'].mean()
    barplot = px.bar(
        grouped, x='Binned probability', y='Target', color_discrete_sequence=['#3BA27A']
    )
    barplot.update_layout(**style_layout(f'Survival Rate by Binned Probabilities (n={len(dff)})'),
                          xaxis={'categoryarray': labels})
    barplot.update_yaxes(title_text="Percentage Survived")

    # --- Age vs Fare Scatter Plot ---
    scatter = px.scatter(
        dff, x='Age', y='Fare', color=TARGET,
        title=f'Age vs Fare by Survival Status (n={len(dff)})', opacity=0.8,
        color_discrete_sequence=['#FFBD59', '#3BA27A']
    )
    scatter.update_layout(font_family='Tahoma')

    # --- Sample Records Table ---
    # Filter table data based on survival toggle
    if target:
        dff_table = dff[dff['Target'] == 1]
    else:
        dff_table = dff[dff['Target'] == 0]

    # When the second toggle is updated or if the slider is updated
    dff_table = dff_table.sort_values('Probability', ascending=ascending).head(n)
    columns = ['Age', 'Gender', 'Class', 'Embark town', TARGET, 'Probability']
    table = go.Figure(data=[go.Table(
        header=dict(
            values=columns,
            fill_color='#23385c',
            line_color='white',
            align='center',
            font=dict(color='white', size=13)
        ),
        cells=dict(
            values=[dff_table[c] for c in columns],
            format=["d", "", "", "", "", ".2%"],
            fill_color=[['white', '#FFF2CC'] * (len(dff_table) - 1)],
            align='center'
        )
    )])
    table.update_layout(title_text=f'Sample Records (n={len(dff_table)})', font_family='Tahoma')

    return age_histogram, class_barplot, histogram, barplot, scatter, table, class_value, gender_value


# -------------------------
# Callback: Predict Survival
# -------------------------
@app.callback(
    Output('predict-result', 'children'),
    Input('predict-button', 'n_clicks'),
    [State('predict-class', 'value'),
     State('predict-gender', 'value'),
     State('predict-age', 'value'),
     State('predict-sibsp', 'value'),
     State('predict-parch', 'value'),
     State('predict-fare', 'value'),
     State('predict-embark', 'value')]
)
def predict_survival(n_clicks, pclass, gender, age, sibsp, parch, fare, embark):
    """
    Predicts survival based on user inputs.
    Validates inputs and returns the prediction and probability.
    """
    if n_clicks == 0:
        return ""

    # Validate inputs
    if None in [age, gender, sibsp, parch, fare, pclass, embark]:
        return html.Div("Please fill all required fields", className="error-message")

    try:
        # Compute extra features
        who = "Child" if age < 18 else ("Man" if gender == "Male" else "Woman")
        adult_male = (gender == "Male" and age >= 18)
        deck = "Missing"
        alone = (sibsp + parch) == 0

        # Prepare input data in expected format
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Sibsp': [sibsp],
            'Parch': [parch],
            'Fare': [fare],
            'Class': [pclass],
            'Who': [who],
            'Adult male': [adult_male],
            'Deck': [deck],
            'Embark town': [embark],
            'Alone': [alone]
        })

        # Predict survival and probability using the trained model
        prediction = pipeline.predict(input_data)[0]
        proba = pipeline.predict_proba(input_data)[0][1]

        # Save prediction to CSV
        save_prediction_to_csv(input_data.iloc[0].to_dict(), prediction, proba)

        return html.Div([
            html.H4("Prediction:"),
            html.P(f"Survived: {'Yes' if prediction else 'No'}"),
            html.P(f"Probability: {proba * 100:.2f}%"),
            html.P("✅ Prediction saved to database", style={'color': '#3BA27A', 'font-size': '12px', 'margin-top': '10px'})
        ])
    except Exception as e:
        return html.Div([
            html.H4("Prediction Error"),
            html.P(str(e))
        ], className="error-message")


# -------------------------
# Callback: Feature Importance Plot
# -------------------------
@app.callback(
    Output('feature-importance-plot', 'figure'),
    Input('app-load-trigger', 'data')
)
def update_feature_importance(_):
    """Update feature importance plot."""
    feature_importance = get_feature_importance(pipeline.named_steps['model'])
    
    if feature_importance is None:
        # Create a placeholder figure
        fig = go.Figure()
        fig.add_annotation(
            text="Feature importance not available for this model type",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title="Feature Importance")
        return fig
    
    # Get feature names (assuming they match the input columns)
    feature_names = ['Age', 'Gender', 'Sibsp', 'Parch', 'Fare', 'Class', 'Who', 'Adult male', 'Deck', 'Embark town', 'Alone']
    
    # Create feature importance plot
    fig = go.Figure(data=[
        go.Bar(
            x=feature_names,
            y=feature_importance,
            marker_color='#3BA27A',
            text=[f'{val:.3f}' for val in feature_importance],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Features",
        yaxis_title="Importance Score",
        font_family='Tahoma',
        plot_bgcolor='rgba(255,242,204,100)'
    )
    
    return fig








# -------------------------
# Callback: Passenger Stories
# -------------------------
@app.callback(
    Output('passenger-stories', 'children'),
    Input('app-load-trigger', 'data')
)
def update_passenger_stories(_):
    """Update passenger stories section."""
    stories = get_passenger_stories()
    
    stories_html = []
    for story in stories:
        survival_color = '#3BA27A' if story['survived'] else '#FFBD59'
        survival_text = 'Survived' if story['survived'] else 'Perished'
        
        stories_html.append(html.Div([
            html.H5(story['name']),
            html.P(f"Class: {story['class']} | Age: {story['age']} | Gender: {story['gender']}"),
            html.P(f"Status: {survival_text}", style={'color': survival_color, 'font-weight': 'bold'}),
            html.P(story['story'], className="story-text")
        ], className="story-card"))
    
    return html.Div(stories_html, className="stories-container")


# -------------------------
# Callback: Titanic Timeline
# -------------------------
@app.callback(
    Output('titanic-timeline', 'children'),
    Input('app-load-trigger', 'data')
)
def update_titanic_timeline(_):
    """Update Titanic timeline section."""
    timeline_events = [
        {
            'date': 'April 10, 1912',
            'time': '12:00 PM',
            'event': 'Titanic departs Southampton, England',
            'description': 'The RMS Titanic begins its maiden voyage with 2,224 passengers and crew. Captain Edward Smith commands the ship.'
        },
        {
            'date': 'April 10, 1912',
            'time': '6:30 PM',
            'event': 'Near collision with SS New York',
            'description': 'Titanic\'s wake causes the SS New York to break its moorings, narrowly avoiding a collision.'
        },
        {
            'date': 'April 11, 1912',
            'time': '11:30 AM',
            'event': 'Arrives in Queenstown, Ireland',
            'description': 'Picks up 123 additional passengers and 1,385 bags of mail before heading to New York.'
        },
        {
            'date': 'April 11, 1912',
            'time': '1:30 PM',
            'event': 'Departs Queenstown',
            'description': 'Titanic sets sail for New York, beginning the transatlantic crossing.'
        },
        {
            'date': 'April 12-13, 1912',
            'time': 'All day',
            'event': 'Smooth sailing across the Atlantic',
            'description': 'The ship enjoys calm seas and good weather, with passengers enjoying the luxurious amenities.'
        },
        {
            'date': 'April 14, 1912',
            'time': '9:00 AM',
            'event': 'Ice warnings received',
            'description': 'Multiple ice warnings are received from other ships, but the Titanic maintains its speed.'
        },
        {
            'date': 'April 14, 1912',
            'time': '11:40 PM',
            'event': 'Iceberg collision',
            'description': 'Titanic strikes an iceberg on the starboard side, causing fatal damage to 6 watertight compartments.'
        },
        {
            'date': 'April 14, 1912',
            'time': '11:50 PM',
            'event': 'Water begins flooding',
            'description': 'Water begins flooding the forward compartments, causing the ship to tilt forward.'
        },
        {
            'date': 'April 15, 1912',
            'time': '12:05 AM',
            'event': 'First lifeboat launched',
            'description': 'Lifeboat 7 is launched with only 28 people (capacity: 65). Women and children first policy begins.'
        },
        {
            'date': 'April 15, 1912',
            'time': '12:45 AM',
            'event': 'First distress signal sent',
            'description': 'The first CQD distress signal is sent. Later changed to the new SOS signal.'
        },
        {
            'date': 'April 15, 1912',
            'time': '1:15 AM',
            'event': 'Ship begins to list',
            'description': 'The Titanic begins to list heavily to port as water continues to flood the ship.'
        },
        {
            'date': 'April 15, 1912',
            'time': '2:05 AM',
            'event': 'Last lifeboat launched',
            'description': 'Collapsible D, the last lifeboat, is launched. Many passengers remain on board.'
        },
        {
            'date': 'April 15, 1912',
            'time': '2:17 AM',
            'event': 'Final radio message',
            'description': 'The final radio message is sent: "Come as quickly as possible, old man; the engine-room is filling up to the boilers."'
        },
        {
            'date': 'April 15, 1912',
            'time': '2:20 AM',
            'event': 'Titanic sinks',
            'description': 'The ship breaks apart and sinks, taking over 1,500 lives. The stern section sinks first.'
        },
        {
            'date': 'April 15, 1912',
            'time': '3:30 AM',
            'event': 'Carpathia arrives',
            'description': 'The RMS Carpathia arrives at the scene and begins rescuing survivors from lifeboats.'
        },
        {
            'date': 'April 18, 1912',
            'time': '9:30 PM',
            'event': 'Carpathia arrives in New York',
            'description': 'The Carpathia arrives in New York with 705 survivors, ending the rescue operation.'
        }
    ]
    
    timeline_html = []
    for event in timeline_events:
        timeline_html.append(html.Div([
            html.H5(f"{event['date']} - {event['time']}"),
            html.H6(event['event']),
            html.P(event['description'])
        ], className="timeline-event"))
    
    return html.Div(timeline_html, className="timeline-container")


# -------------------------
# Callback: Prediction History Table
# -------------------------
@app.callback(
    [Output('prediction-history-table', 'figure'),
     Output('prediction-stats', 'children')],
    [Input('app-load-trigger', 'data'),
     Input('predict-button', 'n_clicks')]
)
def update_prediction_history(_, predict_clicks):
    """Update prediction history table and statistics."""
    try:
        predictions_df = pd.read_csv('user_predictions.csv')
        
        if predictions_df.empty:
            # Create empty table
            fig = go.Figure()
            fig.add_annotation(
                text="No predictions made yet. Make your first prediction!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="User Prediction History")
            
            stats = html.Div([
                html.H4("Prediction Statistics"),
                html.P("No predictions available yet.")
            ])
            
            return fig, stats
        
        # Create table with recent predictions
        recent_predictions = predictions_df.tail(10)  # Show last 10 predictions
        
        # Format the data for display
        display_data = recent_predictions[['timestamp', 'passenger_class', 'gender', 'age', 
                                         'predicted_survival', 'survival_probability']].copy()
        display_data['survival_probability'] = display_data['survival_probability'].apply(lambda x: f"{x:.1%}")
        display_data['predicted_survival'] = display_data['predicted_survival'].apply(lambda x: 'Yes' if x else 'No')
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Timestamp', 'Class', 'Gender', 'Age', 'Predicted', 'Probability'],
                fill_color='#23385c',
                line_color='white',
                align='center',
                font=dict(color='white', size=13)
            ),
            cells=dict(
                values=[display_data[col] for col in display_data.columns],
                fill_color=[['#f8f9fa', '#e9ecef'] * (len(display_data) - 1)],
                align='center',
                font=dict(color='black', size=12)
            )
        )])
        fig.update_layout(title=f"Recent Predictions (Last 10 of {len(predictions_df)} total)")
        
        # Calculate statistics
        total_predictions = len(predictions_df)
        survival_rate = predictions_df['predicted_survival'].mean() * 100
        avg_confidence = predictions_df['prediction_confidence'].mean() * 100
        most_common_class = predictions_df['passenger_class'].mode().iloc[0] if not predictions_df['passenger_class'].mode().empty else 'N/A'
        
        stats = html.Div([
            html.H4("Prediction Statistics"),
            html.P(f"Total Predictions: {total_predictions}"),
            html.P(f"Average Predicted Survival Rate: {survival_rate:.1f}%"),
            html.P(f"Average Prediction Confidence: {avg_confidence:.1f}%"),
            html.P(f"Most Common Passenger Class: {most_common_class}")
        ])
        
        return fig, stats
        
    except FileNotFoundError:
        # Create empty table if file doesn't exist
        fig = go.Figure()
        fig.add_annotation(
            text="No predictions made yet. Make your first prediction!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title="User Prediction History")
        
        stats = html.Div([
            html.H4("Prediction Statistics"),
            html.P("No predictions available yet.")
        ])
        
        return fig, stats


# -------------------------
# Callback: Download Data
# -------------------------
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-button", "n_clicks"),
    [State('export-options', 'value'),
     State('class-dropdown', 'value'),
     State('gender-dropdown', 'value')],
    prevent_initial_call=True
)
def download_data(n_clicks, export_options, class_value, gender_value):
    """Download filtered data as CSV."""
    if not export_options:
        return None
    
    # Apply filters to get filtered data
    dff = test.copy()
    if class_value:
        dff = dff[dff['Class'].isin(class_value)]
    if gender_value:
        dff = dff[dff['Gender'].isin(gender_value)]
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if 'filtered' in export_options:
        filename = f"titanic_filtered_data_{timestamp}.csv"
        return dcc.send_data_frame(dff.to_csv, filename)
    
    if 'predictions' in export_options:
        try:
            predictions_df = pd.read_csv('user_predictions.csv')
            filename = f"user_predictions_history_{timestamp}.csv"
            return dcc.send_data_frame(predictions_df.to_csv, filename)
        except FileNotFoundError:
            # Create empty predictions file if it doesn't exist
            empty_df = pd.DataFrame(columns=[
                'timestamp', 'passenger_class', 'gender', 'age', 'siblings_spouses',
                'parents_children', 'fare', 'embark_town', 'who', 'adult_male',
                'deck', 'alone', 'predicted_survival', 'survival_probability', 'prediction_confidence'
            ])
            filename = f"user_predictions_history_{timestamp}.csv"
            return dcc.send_data_frame(empty_df.to_csv, filename)
    
    return None


if __name__ == '__main__':
    app.run(debug=True)