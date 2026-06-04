import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from deep_translator import GoogleTranslator
import nltk
from numpy import random
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import numpy as np
from streamlit_extras.stylable_container import stylable_container

# ==============================================================

nltk.download("stopwords")


def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("reco.css")


@st.cache_data
def load_csv():
    return pd.read_csv("movies_ml_v5.csv")


@st.cache_data
def prep_X(dataFrame):
    X = dataFrame["NLP"]
    tfidf = TfidfVectorizer(stop_words="english")
    X_tfidf = tfidf.fit_transform(X)
    return X_tfidf


# ==============================================================
#   DEF DES FONCTOINS AFFICHAGE
# ==============================================================


def hours(x):
    h = 0
    while x > 60:
        x = x - 60
        h += 1
    return str(f"{h} H {x} min")


def translate(x):
    return GoogleTranslator(source="en", target="fr").translate(x)


def compte_film():
    liste_film = []
    for i in range(len(df)):
        liste_film.append(df["primaryTitle"][i])
        if df["originalTitle"][i] not in liste_film:
            liste_film.append(df["originalTitle"][i])
    return liste_film


# ==============================================================
#       MACHINE LEARNING
# ==============================================================


def reco(title, df, nbr):

    match = df[df["originalTitle"] == title]

    if match.empty:
        return st.markdown(
            "<div class='title-center'>Aucun Resultat</div>", unsafe_allow_html=True
        )

    idx = match.index[0]

    similarity = cosine_similarity(X_tfidf[idx : idx + 1], X_tfidf)

    sim_scores = list(enumerate(similarity[0]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    liste_sim = sim_scores[1 : nbr + 1]

    return df.iloc[[i[0] for i in liste_sim]][
        [
            "originalTitle",
            "poster_url",
            "startYear",
            "runtimeMinutes",
            "averageRating",
            "genres",
            "overview",
        ]
    ]


# ==============================================================
#       init
# ==============================================================

df = load_csv()

X_tfidf = prep_X(df)

liste_films = df["originalTitle"].tolist()

# ==============================================================
#       SIDE BAR
# ==============================================================

with st.sidebar:
    menu = option_menu(
        menu_title=None,
        options=["Recomandation", "A la Une", "KPI"],
        icons=["film", "star-fill", "gear"],
    )

# ==============================================================
#       PAGE RECO DE SIDE BAR
# ==============================================================
st.set_page_config(
    page_title="Reco",
    layout="wide"
    )
if menu == "Recomandation":
    # ==============================================================
    #   VREIF / INIT  MEMOIRE
    # ==============================================================

    if "movies_infos" not in st.session_state:
        st.session_state.movies_infos = None

    if "avis_status" not in st.session_state:
        st.session_state.avis_status = False

    if "page" not in st.session_state:
        st.session_state.page = "reco"

    if "target_movies" not in st.session_state:
        st.session_state.target_movies = random.randint(30000)

    # ==============================================================
    #   PAGE DE RECO
    # ==============================================================

    if st.session_state.page == "reco":
        st.markdown(
            "<div class='title-left'>Netflix Reco</div>", unsafe_allow_html=True
        )
        st.markdown(
            "<div class='title-center'>Nos Recommandations</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='question-title'>Quel est le dernier film que vous avez aimé ?</div>",
            unsafe_allow_html=True,
        )

        target = st.selectbox("", liste_films, index=st.session_state.target_movies)
        st.session_state.target_movies = liste_films.index(target)

        df_reco = reco(target, df, 20)

        cols = st.columns(4)

        for i, (_, row) in enumerate(df_reco.iterrows()):
            with cols[i % 4]:
                with stylable_container(
                    key=f"carte_{i}",
                    css_styles="""
                    {
                        background-color: #080a0e;
                        border-radius: 12px;
                        padding: 20px;
                        transition: all 0.2s ease;
                    }"""):
                    
                        if pd.notna(row["poster_url"]):
                            st.image(row["poster_url"])
                        else:
                            st.image("logo.png")
                        st.markdown(
                            "<div style='height: 20px;'></div>",
                            unsafe_allow_html=True
                        )
                        if pd.notna(row["originalTitle"]):
                            st.markdown(
                                f"<div class='movies_title'>{row['originalTitle']}</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"<div class='movies_title'>{row['primaryTitle']}</div>",
                                unsafe_allow_html=True,
                            )
                        
                        st.markdown(
                            "<div style='height: 30px;'></div>",
                            unsafe_allow_html=True
                            )

                        if st.button("Voir", key=f"btn_{i}", use_container_width=True):

                            st.session_state.movies_infos = row
                            st.session_state.page = "infos"
                            st.rerun()

    # ==============================================================
    #    PAGE AVIS
    # ==============================================================

    if st.session_state.page == "avis":

        infos = st.session_state.movies_infos

        st.markdown(
            f"<div class='avis_title'>Qu'avez vous pensé de {infos['originalTitle']} ?</div>",
            unsafe_allow_html=True,
        )
        st.text_area(label="", max_chars=300)
        st.markdown(
            f"<div class='slider_avis_title'>Comment le sitier vous ?</div>",
            unsafe_allow_html=True,
        )
        st.select_slider(
            "", ["Mauvais", "Sans plus", "Bon", "Tres bon", "Excellent"], value="Bon"
        )

        if st.button("Valider", use_container_width=True):
            st.session_state.avis_status = False
            st.session_state.page = "infos"
            st.rerun()

    # ==============================================================
    #    PAGE INFO FILM SI MEMOIRE
    # ==============================================================

    if st.session_state.page == "infos":

        infos = st.session_state.movies_infos

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if pd.notna(infos["poster_url"]):
                st.image(infos["poster_url"])
            else:
                st.image("logo.png")

        st.markdown(
            f"<div class='spe_title'>{infos['originalTitle']}</div>",
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            st.markdown(
                f"<div class='date'>Année: {infos['startYear']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='time'>Durée: {hours(infos['runtimeMinutes'])}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='note'>Note: {infos['averageRating']}</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div style='height: 10px;'></div>",
                unsafe_allow_html=True)

            if st.button("Donnez votre avis", use_container_width=True):
                st.session_state.avis_status = True
                st.session_state.page = "avis"
                st.rerun()

        with col3:
            st.markdown(
                f"<div class='genres'>{translate(infos['genres'])}</div>",
                unsafe_allow_html=True,
            )

        if pd.notna(infos["overview"]):
            st.markdown(
                f"<div class='description'>{translate(infos['overview'])}</div>",
                unsafe_allow_html=True,
            )
        if pd.isna(infos["overview"]):
            st.markdown(
                f"<div class='spe_title'>Pas de description</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "<div style='height: 30px;'></div>",
            unsafe_allow_html=True)
        if st.button("Retour", use_container_width=True):
            st.session_state.movies_infos = None
            st.session_state.page = "reco"
            st.rerun()

        # ==============================================================
        #    RECO DU FILM INFOS
        # ==============================================================

        df_reco_infos = reco(infos["originalTitle"], df, 5)

        cols = st.columns(5)

        for i, (_, row) in enumerate(df_reco_infos.iterrows()):
            with cols[i]:
                with stylable_container(
                        key=f"carte_{i}",
                        css_styles="""
                        {
                            background-color: #080a0e;
                            border-radius: 12px;
                            padding: 20px;
                            transition: all 0.2s ease;
                        }"""):
                    if pd.notna(row["poster_url"]):
                        st.image(row["poster_url"])
                    else:
                        st.image("logo.png")

                    if pd.notna(row["originalTitle"]):
                        st.markdown(
                            f"<div class='movies_title'>{row['originalTitle']}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<div class='movies_title'>{row['primaryTitle']}</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        "<div style='height: 10px;'></div>",
                        unsafe_allow_html=True)

                    if st.button("Voir", key=f"btn_{i}", use_container_width=True):

                        st.session_state.movies_infos = row
                        st.session_state.page = "infos"
                        st.rerun()


# ==============================================================
#       PAGE A LA UNE DE SIDE BAR
# ==============================================================

if menu == "A la Une":

    # ==============================================================
    #   VREIF / INIT  MEMOIRE PAGE A LA UNE
    # ==============================================================
    if "movies_infos" not in st.session_state:
        st.session_state.movies_infos = None

    if "avis_status" not in st.session_state:
        st.session_state.avis_status = False

    if "page2" not in st.session_state:
        st.session_state.page2 = "top"

    if "cat" not in st.session_state:
        st.session_state.cat = 0

    # ==============================================================
    #    PAGE A LA UNE
    # ==============================================================

    if st.session_state.page2 == "top":

        # ==============================================================
        #    BDD  TOP FILM
        # ==============================================================

        df_top = df[(df["popularity"] > 25) & (df["startYear"] > 2015)].sort_values(
            "averageRating", ascending=False
        )

        #   CREATION DATA FRAME TOP FILM FR
        df_top_fr = df_top[df_top["production_countries"] == "['FR']"].head(10)

        # ==============================================================
        #    TOP DES FILMS EN FRANCE
        # ==============================================================

        st.markdown(
            f"<div class= 'title-center'>Top des films<div>", unsafe_allow_html=True
        )
        st.markdown(
            f"<div class= 'title-left'>Top des films en France<div>",
            unsafe_allow_html=True,
        )

        with stylable_container(
            key=f"carte",
            css_styles="""
            {
                background-color: #420f0f;
                border-radius: 12px;
                padding: 20px;
                transition: all 0.2s ease;
            }"""):
            cols = st.columns(5)
            for i, (_, row) in enumerate(df_top_fr.iterrows()):
                with cols[i % 5]:
                    if pd.notna(row["poster_url"]):
                        st.image(row["poster_url"])
                    else:
                        st.image("logo.png")
                    with stylable_container(
                        key=f"carte_{i}",
                        css_styles="""
                        {
                            background-color: #000000;
                            border-radius: 12px;
                            padding: 20px;
                            transition: all 0.2s ease;
                        }"""):
                        st.markdown(
                            f"<div class='movies_title'>{row['originalTitle']}</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            "<div style='height: 10px;'></div>",
                            unsafe_allow_html=True)

                        if st.button("Voir", key=f"btn_{i}", use_container_width=True):

                            st.session_state.movies_infos = row
                            st.session_state.page2 = "infos"
                            st.rerun()

        # ==============================================================
        #    TOP DES PAR CAT
        # ==============================================================

        liste_genres = df_top["main_genre"].unique().tolist()

        st.markdown(
            "<div style='height: 50px;'></div>",
            unsafe_allow_html=True)

        st.markdown(
            f"<div class= 'title-left'>Top des films par genres<div>",
            unsafe_allow_html=True,
        )

        target = st.selectbox("", liste_genres, index=st.session_state.cat)
        st.session_state.cat = liste_genres.index(target)

        cols_ = st.columns(4)
        count = 0

        for i, (_, row) in enumerate(df_top.iterrows()):
            if target in row["liste_genre"]:
                with cols_[i % 4]:
                    if pd.notna(row["poster_url"]):
                        st.image(row["poster_url"])
                    else:
                        st.image("logo.png")

                    st.markdown(
                        f"<div class='movies_title'>{row['originalTitle']}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        "<div style='height: 20px;'></div>",
                        unsafe_allow_html=True)
                    if st.button("Voir", key=f"btn-{i}", use_container_width=True):

                        st.session_state.movies_infos = row
                        st.session_state.page2 = "infos"
                        st.rerun()
                count += 1
                if count == 16:
                    break

    # ==============================================================
    #     AVIS DE LE PAGE TOP
    # ==============================================================

    if st.session_state.page2 == "avis":

        infos = st.session_state.movies_infos

        st.markdown(
            f"<div class='avis_title'>Qu'avez vous pensé de {infos['originalTitle']} ?</div>",
            unsafe_allow_html=True,
        )
        st.text_area(label="", max_chars=300)
        st.markdown(
            f"<div class='slider_avis_title'>Comment le sitier vous ?</div>",
            unsafe_allow_html=True,
        )
        st.select_slider(
            "", ["Mauvais", "Sans plus", "Bon", "Tres bon", "Excellent"], value="Bon"
        )

        if st.button("Valider", use_container_width=True):
            st.session_state.avis_status = False
            st.session_state.page2 = "infos"
            st.rerun()

    # ==============================================================
    #    PAGE INFO FILM SI MEMOIRE     DE LE PAGE TOP
    # ==============================================================

    if st.session_state.page2 == "infos":

        infos = st.session_state.movies_infos

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if pd.notna(infos["poster_url"]):
                st.image(infos["poster_url"])
            else:
                st.image("logo.png")

        st.markdown(
            f"<div class='spe_title'>{infos['originalTitle']}</div>",
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            st.markdown(
                f"<div class='date'>Année: {infos['startYear']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='time'>Durée: {hours(infos['runtimeMinutes'])}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='note'>Note: {infos['averageRating']}</div>",
                unsafe_allow_html=True,
            )

            if st.button("Donnez votre avis", use_container_width=True):
                st.session_state.avis_status = True
                st.session_state.page2 = "avis"
                st.rerun()

        with col3:
            st.markdown(
                f"<div class='genres'>{translate(infos['genres'])}</div>",
                unsafe_allow_html=True,
            )

        if pd.notna(infos["overview"]):
            st.markdown(
                f"<div class='description'>{translate(infos['overview'])}</div>",
                unsafe_allow_html=True,
            )
        if pd.isna(infos["overview"]):
            st.markdown(
                f"<div class='spe_title'>Pas de description</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "<div style='height: 30px;'></div>",
            unsafe_allow_html=True)

        if st.button("Retour", use_container_width=True):
            st.session_state.movies_infos = None
            st.session_state.page2 = "top"
            st.rerun()

        # ==============================================================
        #    RECO DU Film INFOS
        # ==============================================================

        df_reco_infos = reco(infos["originalTitle"], df, 5)

        cols = st.columns(5)

        for i, (_, row) in enumerate(df_reco_infos.iterrows()):
            with cols[i]:
                with stylable_container(
                        key=f"carte_{i}",
                        css_styles="""
                        {
                            background-color: #000000;
                            border-radius: 12px;
                            padding: 20px;
                            transition: all 0.2s ease;
                        }"""):
                    if pd.notna(row["poster_url"]):
                        st.image(row["poster_url"])
                    else:
                        st.image("logo.png")

                    if pd.notna(row["originalTitle"]):
                        st.markdown(
                            f"<div class='movies_title'>{row['originalTitle']}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<div class='movies_title'>{row['primaryTitle']}</div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        "<div style='height: 10px;'></div>",
                        unsafe_allow_html=True)

                    if st.button("Voir", key=f"btn_{i}", use_container_width=True):

                        st.session_state.movies_infos = row
                        st.session_state.page = "infos"
                        st.rerun()

# ==============================================================
#    PAGE KPI
# ==============================================================
if menu == "KPI":
    st.markdown(
            "<div class='title-center'>Participation des utilisateurs</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<div style='height: 40px;'></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='title-etude'>Compréhension et exploitation des notes</div>", unsafe_allow_html=True
        )
    st.markdown("""Dans le cadre de l’élaboration de notre système de recommandation, nous cherchons à 
                comprendre la signification et l’impact des notes attribuées par les utilisateurs, dans 
                un but de compréhension et d’amélioration du système.""")

    
    # ==============================================================
    #    Fig 1 
    # ==============================================================
    note_evolution = df.groupby('startYear')['averageRating'].mean()
    fig = px.line(note_evolution)

    fig.update_traces(
        line=dict(
            color="red",
            dash="solid"
        )
    )

    fig.update_layout(
        title={
            "text": "Évolution des notes",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 25}
            },
        xaxis_title="Année",
        yaxis_title="Note Moyenne",
        paper_bgcolor= "#0e1117",
        plot_bgcolor="#121212",
        font=dict(color="white")

    )
    with st.container(border=True):
        st.plotly_chart(fig)
        st.markdown("""Nous pouvons observer que les notes présentent une tendance à la baisse depuis le début de nos données.""")

    st.markdown("""Plusieurs raisons peuvent expliquer cette baisse, comme le montre les deux hypothèses suivantes. 
                Dans un premier temps, on pourrait penser que les films sont de moins bonne qualité, ce qui pourrait 
                se traduire par une baisse des budgets. Cependant, comme nous pouvons le constater, ce n’est pas le 
                cas : les budgets moyens des films sont en hausse.""")
    st.markdown(
        "<div style='height: 30px;'></div>",
        unsafe_allow_html=True
    )
    # ==============================================================
    #    Fig 2 
    # ==============================================================

    kpi = df.groupby(["decade"])["budget"].mean().reset_index()
    fig2 = px.bar(
        kpi,
        x="decade",
        y="budget",
        barmode="group"
    )

    fig2.update_traces(marker_color="red")

    fig2.update_layout(
        title={
            "text": "Évolution du budget",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 25}
            },
        xaxis_title= "Décenie",
        yaxis_title= "Budget",
        paper_bgcolor= "#0e1117",
        plot_bgcolor="#121212",
        font=dict(color="white")
        )

    components.html(f"""
    <style>
    .card {{
        background: #420f0f;
        padding: 10px;
        border-radius: 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }}
    </style>

    <div class="card">
        {fig2.to_html(full_html=False, include_plotlyjs="cdn")}
    </div>
    """, height=550)

    st.markdown("""La seconde hypothèse est que les utilisateurs contribuent moins en laissant des avis sur les films. 
                Cette hypothèse est renforcée par l’observation de niveaux de participation extrêmement faibles. Toutefois, 
                les films connaissent une croissance significative de leur popularité, ce qui permet de supposer que les 
                utilisateurs tendent à moins exprimer leurs retours, y compris lorsqu’ils apprécient les œuvres."""
    )

    st.markdown(
        "<div style='height: 30px;'></div>",
        unsafe_allow_html=True
    )

    # ==============================================================
    #    Fig 3 
    # ==============================================================
    evo_nbr = df.groupby('startYear')['numVotes'].mean()
    fig3 = px.line(evo_nbr)

    fig3.update_traces(
        line=dict(
            color="red",
            dash="solid"
        )
    )

    fig3.update_layout(
        title={
            "text": "Évolution du nombre de vote",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 25}
            },
        xaxis_title="Decennie",
        yaxis_title="Nombre de vote",
        paper_bgcolor= "#0e1117",
        plot_bgcolor="#121212",
        font=dict(color="white")

    )
    with st.container(border=True):
        st.plotly_chart(fig3)
    
    st.markdown(
        "<div style='height: 80px;'></div>",
        unsafe_allow_html=True
    )
    # ==============================================================
    #    Fig 4 
    # ==============================================================
    st.markdown("""Cette hypothèse semble se confirmer, notamment à travers 
                l’augmentation des profits générés par les films.""")

    revenue_profit = df.groupby("startYear")[['revenue', 'profit']].mean()

    fig5 = px.line(revenue_profit["profit"])

    fig5.update_traces(
        line=dict(
            color="red",
            dash="solid")
    )

    fig5.update_layout(
        title={
            "text": "Évolution du profit",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 25}
            },
        xaxis_title="Année",
        yaxis_title="Profit Moyenne",
        paper_bgcolor= "#0e1117",
        plot_bgcolor="#121212",
        font=dict(color="white")
    )

    components.html(f"""
    <style>
    .card {{
        background: #420f0f;
        padding: 10px;
        border-radius: 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }}
    </style>

    <div class="card">
        {fig5.to_html(full_html=False, include_plotlyjs="cdn")}
    </div>
    """, height=550)

    st.markdown(
        "<div style='height: 20px;'></div>",
        unsafe_allow_html=True
    )

    # ==============================================================
    #    Fig 5 
    # ==============================================================
    color_map = {
    "Comedy": "#ff4d4d",        # rouge vif
    "Fantasy": "#ff6666",       # rouge clair
    "Romance": "#ff3366",       # rose rouge intense
    "Documentary": "#ff99aa",   # rose doux

    "War": "#b30000",           # rouge foncé
    "Horror": "#660000",        # rouge très sombre
    "Sci-Fi": "#ff4d4d",        # rouge (réutilisé)

    "Drama": "#ffb3c1",         # rose pâle
    "Mystery": "#ffd6d6",       # rose très clair
    "Thriller": "#cc0000",      # rouge fort
    "Crime": "#990000",         # rouge profond

    "Western": "#fff5f5",       # quasi blanc rosé
    "Biography": "#ffccd5",     # rose doux
    "History": "#ffe6e6",       # rose très clair

    "Action": "#ff1a1a",        # rouge dynamique
    "Adventure": "#ff8080",     # rouge moyen

    "Animation": "#ffc0cb",     # rose classique
    "Music": "#ffe0e0",         # rose clair
    "Musical": "#fff0f0",       # très proche blanc

    "Family": "#ffffff",        # blanc
    "Sport": "#ff4d6d",         # rose rouge sportif
    "News": "#f2f2f2"}          # blanc cassé

    compteur_vote = {}

    df["vote_count"] = df["vote_count"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["vote_count"])

    for _, row in df.iterrows():
        if pd.notna(row['vote_count']):
            for genres in row["genres"].split(","):
                if genres in compteur_vote:
                        compteur_vote[genres] += row['vote_count']
                else:
                        compteur_vote[genres] = row['vote_count']

    compteur_genre = {}

    for _, row in df.iterrows():
        if pd.notna(row['vote_count']):
            for genres in row["genres"].split(","):
                if genres in compteur_genre:
                        compteur_genre[genres] += 1
                else:
                        compteur_genre[genres] = 1

    fig7 = px.pie(names=list(compteur_vote.keys()),
               values=list(compteur_vote.values()),
               color=list(compteur_genre.keys()),
               title="repartion des genres",
               color_discrete_map= color_map)

    fig7.update_layout(
        title={
            "text": "Les genres générant le plus de votes",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 25}
            },
        xaxis_title="Année",
        yaxis_title="Profit Moyenne",
        paper_bgcolor= "#0e1117",
        plot_bgcolor="#121212",
        font=dict(color="white")
    )

    fig8 = px.pie(names=list(compteur_genre.keys()),
               values=list(compteur_genre.values()),
               color=list(compteur_genre.keys()),
               title="repartion des genres",
               color_discrete_map= color_map)

    fig8.update_layout(
        title={
            "text": "Les genres les plus produits",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 25}
            },
        xaxis_title="Année",
        yaxis_title="Profit Moyenne",
        paper_bgcolor= "#0e1117",
        plot_bgcolor="#121212",
        font=dict(color="white")
    )

    st.markdown("""L’un des objectifs est d’identifier les genres de films les plus engageants pour 
                les spectateurs, en fonction du volume de retours générés. Cette analyse est comparée 
                à la répartition des genres les plus produits afin de déterminer les catégories sur 
                lesquelles des efforts pourraient être entrepris pour augmenter le taux de retour.""")

    col1, col2 = st.columns(2)

    
    components.html(f"""
    <style>
    .card {{
        background: #420f0f;
        padding: 10px;
        border-radius: 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }}
    </style>

    <div class="card">
        {fig7.to_html(full_html=False, include_plotlyjs="cdn")}
    </div>
    """, height=550)
    
    components.html(f"""
    <style>
    .card {{
        background: #420f0f;
        padding: 10px;
        border-radius: 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }}
    </style>

    <div class="card">
        {fig8.to_html(full_html=False, include_plotlyjs="cdn")}
    </div>
    """, height=550)

    st.markdown(
        "<div style='height: 70px;'></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='title-etude'>Conclusion de l’étude</div>", unsafe_allow_html=True
        )
    st.markdown("""Cette étude met en évidence une tendance générale à la baisse des notes et des retours utilisateurs, 
                malgré une augmentation de la popularité et des profits des films. Cela suggère que les spectateurs continuent 
                de consommer davantage de contenus, mais participent moins activement à l’évaluation des films.
                L’analyse des genres permet également d’identifier des écarts entre les types de films les plus produits et ceux 
                générant le plus d’engagement. Certains genres semblent ainsi moins inciter les utilisateurs à laisser un avis, 
                ce qui constitue une piste d’amélioration importante.""")
    st.write("_______________________________________________________________________________________________________________")
    st.markdown(
        "<div class='title-etude'>Solutions proposées</div>", unsafe_allow_html=True
        )
    st.markdown("""Afin d’améliorer le nombre de retours utilisateurs, plusieurs actions peuvent être envisagées :
                Optimisation du bouton d’avis déjà en place : le rendre plus visible, plus attractif ou intégré directement après le visionnage d’un film.
                Réduction de la friction : simplifier davantage le processus de notation (ex : notation en 1 clic, réactions rapides type emoji).
                Incitation à l’engagement : proposer des rappels discrets ou des suggestions personnalisées pour laisser un avis après avoir regardé un film.
                Gamification : récompenser les utilisateurs actifs (badges, recommandations améliorées, historique enrichi).
                Analyse ciblée des genres peu commentés : adapter l’interface ou les incitations selon les types de films les moins évalués.""")
    st.write("_______________________________________________________________________________________________________________")