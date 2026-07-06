# File: LocalityHeatmap.py
"""
Locality-wise health condition heatmap for Help the Greens.
Shows disease concentration by geographic location.
"""

import folium
from folium.plugins import HeatMap, MarkerCluster  # FastMarkerCluster removed (unused)
import pandas as pd
import sqlite3
import html
import streamlit as st
from streamlit_folium import st_folium


def get_locality_stats(db_file):
    """
    Fetches health statistics grouped by locality/coordinates.
    Returns: DataFrame with locality, lat, lon, health distribution
    """
    try:
        conn = sqlite3.connect(db_file)
        query = """
        SELECT 
            latitude, longitude,
            COUNT(*) as total_scans,
            SUM(CASE WHEN health='Healthy' THEN 1 ELSE 0 END) as healthy_count,
            SUM(CASE WHEN health='Stressed' THEN 1 ELSE 0 END) as stressed_count,
            SUM(CASE WHEN health='Diseased' THEN 1 ELSE 0 END) as diseased_count
        FROM survey
        WHERE latitude != 0 AND longitude != 0
        GROUP BY ROUND(latitude, 2), ROUND(longitude, 2)
        """
        df = pd.read_sql(query, conn)
        conn.close()

        df['health_rate'] = (df['healthy_count'] / df['total_scans'] * 100).round(2)
        df['disease_rate'] = (df['diseased_count'] / df['total_scans'] * 100).round(2)

        return df
    except Exception as e:
        print(f"Error fetching locality stats: {e}")
        return pd.DataFrame()


def create_disease_heatmap(db_file, center=None):
    """
    Creates a Folium heatmap showing disease concentration.
    Green = healthy areas, Red = high disease concentration.
    center: [lat, lon] — defaults to centre of India if not supplied.
    """
    if center is None:          # safe mutable-default fix
        center = [20.0, 78.0]

    df = get_locality_stats(db_file)
    if df.empty:
        return None

    # Auto-centre on actual data instead of the hard-coded default when possible
    map_center = [df['latitude'].mean(), df['longitude'].mean()]

    m = folium.Map(location=map_center, zoom_start=6, tiles='OpenStreetMap')

    # Heatmap layer — intensity driven by disease_rate (0-100)
    heat_data = [
        [row['latitude'], row['longitude'], row['disease_rate'] / 100.0]  # normalize to 0-1
        for _, row in df.iterrows()
    ]
    HeatMap(
        heat_data,
        name='Disease Concentration',
        min_opacity=0.3,
        radius=30,
        blur=15,
        max_zoom=13,
        gradient={0.0: '#2E8B57', 0.5: '#FFD700', 1.0: '#DC2626'}
    ).add_to(m)

    # Circle markers with health-rate colour coding
    for _, row in df.iterrows():
        if row['total_scans'] <= 0:
            continue
        if row['health_rate'] >= 70:
            color = 'green'
        elif row['health_rate'] >= 40:
            color = 'orange'
        else:
            color = 'red'

        popup_text = (
            f"<b>Location: ({row['latitude']:.2f}, {row['longitude']:.2f})</b><br>"
            f"Total Scans: {int(row['total_scans'])}<br>"
            f"<span style='color:#15803d'>✅ Healthy: {int(row['healthy_count'])} ({row['health_rate']:.1f}%)</span><br>"
            f"<span style='color:#f59e0b'>⚠️ Stressed: {int(row['stressed_count'])}</span><br>"
            f"<span style='color:#dc2626'>❌ Diseased: {int(row['diseased_count'])} ({row['disease_rate']:.1f}%)</span>"
        )
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8 + (row['total_scans'] / 5),
            popup=folium.Popup(popup_text, max_width=300),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2,
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def create_health_distribution_chart(db_file):
    """
    Renders locality health statistics directly into the Streamlit page.
    No map — just metrics, a bar chart, and a data table.
    """
    df = get_locality_stats(db_file)   # ← no 'center' variable here at all

    if df.empty:
        st.info("No data available for locality analysis.")
        return

    df['Locality'] = df.apply(
        lambda x: f"({x['latitude']:.2f}, {x['longitude']:.2f})", axis=1
    )

    st.subheader("📊 Health Statistics by Locality")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Localities", len(df))
        st.metric("Total Scans", int(df['total_scans'].sum()))
    with col2:
        st.metric("Avg Health Rate", f"{df['health_rate'].mean():.1f}%")
        st.metric("Diseased Locations", len(df[df['disease_rate'] > 50]))

    st.subheader("🎯 Health Rate by Location")
    chart_data = (
        df.sort_values('health_rate', ascending=False)
        .head(10)[['Locality', 'health_rate', 'disease_rate']]
        .set_index('Locality')
    )
    st.bar_chart(chart_data)

    st.subheader("📍 Detailed Locality Data")
    display_df = df[[
        'latitude', 'longitude', 'total_scans',
        'healthy_count', 'stressed_count', 'diseased_count',
        'health_rate', 'disease_rate'
    ]].copy()
    display_df.columns = [
        'Latitude', 'Longitude', 'Total Scans',
        'Healthy', 'Stressed', 'Diseased', 'Health %', 'Disease %'
    ]
    st.dataframe(display_df.sort_values('Disease %', ascending=False), width='stretch')


def create_clustered_map(db_file, center=None):
    """
    Clustered marker map — available for future use.
    """
    if center is None:
        center = [20.0, 78.0]

    conn = sqlite3.connect(db_file)
    query = """
    SELECT latitude, longitude, tree_name, health, confidence, timestamp
    FROM survey
    WHERE latitude != 0 AND longitude != 0
    ORDER BY timestamp DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        return None

    map_center = [df['latitude'].mean(), df['longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=6, tiles='CartoDB positron')
    marker_cluster = MarkerCluster(name='Tree Clusters').add_to(m)

    for _, row in df.iterrows():
        color = {'Healthy': 'green', 'Stressed': 'orange'}.get(row['health'], 'red')
        popup = (
            f"<b>{html.escape(str(row['tree_name']))}</b><br>"
            f"Status: {html.escape(str(row['health']))}<br>"
            f"Confidence: {int(row['confidence'])}%<br>"
            f"Date: {html.escape(str(row['timestamp'])[:10])}"
        )
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup, max_width=250),
            icon=folium.Icon(color=color, icon='leaf'),
            tooltip=f"{row['tree_name']} - {row['health']}",
        ).add_to(marker_cluster)

    folium.LayerControl().add_to(m)
    return m