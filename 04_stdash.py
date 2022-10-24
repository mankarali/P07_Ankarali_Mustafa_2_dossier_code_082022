# ------------ Libraries import ---------------------------
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import requests
import pickle
import sklearn
from PIL import Image

import xgboost as xgb


# Importer le jeu de données (normale, normalisée) and modele

path1 = 'https://raw.githubusercontent.com/mankarali/mankarali07dash/master/credit_test_sample_data.csv'
path2 = 'https://raw.githubusercontent.com/mankarali/mankarali07dash/master/credit_test_sample_data_normalise.csv'

# 1200 Semples 1200 pour df_test
df_test = pd.read_csv(path1, encoding='unicode_escape').sample(1200, random_state=42)


df_test = df_test.loc[:, ~df_test.columns.str.match ('Unnamed')]
df_test = df_test.sort_values ('SK_ID_CURR')
# 1200 Semples 1200 pour df_test_normalize
df_test_normalize = pd.read_csv (path2, index_col=0).sample(1200, random_state=42)

features = df_test_normalize.columns[: -1]
ID_de_client = df_test_normalize.index.sort_values()
treshold = 0.50


# importer le modele
model = xgb.XGBClassifier ()
model.load_model("XGBClassifier_credit.json")




## importer les images ##
img1 = Image.open(r'image_entreprise.png')

img2 = Image.open(r'shape_general.png')
img3 = Image.open(r'shape_local.png')





def load_infos_gen(data):
    lst_infos = [data.shape[0],
                 round(data["AMT_INCOME_TOTAL"].mean(), 2),
                 round(data["AMT_CREDIT"].mean(), 2)]

    nb_credits = lst_infos[0]
    rev_moy = lst_infos[1]
    credits_moy = lst_infos[2]

    gender = data.CODE_GENDER.value_counts()

    return nb_credits, rev_moy, credits_moy, gender




# ------------ Définir la configuration de base pour streamlit -------
st.set_page_config(layout="wide")

# ------------ Configuration de la barre latérale (sidebar) ----------------------


### Afficher l'image avec streamlit ###
st.sidebar.image(img1)

### Ajouter une colonne pour input de l'utilisateur ###
st.sidebar.header('Definir ID de client:')
selected_credit = st.sidebar.selectbox('ID de client', ID_de_client)
### Ajouter checkbox pour afficher différentes informations client ###
client_data = st.sidebar.checkbox('Informations générales')
client_pred_score = st.sidebar.checkbox('Analyse de la demande de crédit')

### Ajouter checkbox pour afficher l'analyse des données client ###
client_analysis = st.sidebar.checkbox('Analyse des features de client')

# Chargement des informations générales
st.sidebar.header("**Informations générales**")
nb_credits, rev_moy, credits_moy, gender = load_infos_gen(df_test)

# Revenu moyen
st.sidebar.markdown("<u>Revenu moyen (USD) :</u>", unsafe_allow_html=True)
st.sidebar.text(rev_moy)

# AMT CREDIT
st.sidebar.markdown("<u>Montant moyen du prêt (USD) :</u>", unsafe_allow_html=True)
st.sidebar.text(credits_moy)





# ------------ Affichage principal -----------------
## Titre générique ##
st.write('# Implémentez un modèle de scoring')
st.write("## **Classification d'une demande de crédit pour 'Prêt à dépenser'**")

## Afficher input dataframe avec une sélection multiple de fonctionnalités pour toute la liste de passagers disponible (les données ne sont pas mises à l'échelle standard ici !) ##
st.write('### Informations générales clients ('+ str(df_test.shape[0]) + 'semples):')
st.write('Dimension de données: ' + str(df_test.shape[0]) + ' lignes ' + str(df_test.shape[1]) + ' colonnes')
selections = st.multiselect('Vous pouvez ajouter ou enlever une donnée présente dans cette liste:', df_test.columns.tolist(),
 df_test.columns.tolist()[0:10])
st.dataframe(df_test.loc[:,selections].sort_index())
### ajouter un extenseur pour plus d'explications sur les données  ###
with st.expander('Informations complémentaires'):
    st.write(""" Ici vous trouvez les informations disponibles pour tous les clients.  \n"""
            """ Pour plus d'informations sur les features (variables) disponibles merci de contacter l'équipe support. """)



colors = ['red', 'green']

fig = go.Figure(data=[go.Pie(labels=[ "Default", "Non Default"],
                             values=[8.3 , 91.7])])
fig.update_traces(hoverinfo='label+percent', textinfo='label+percent', textfont_size=20,pull=[0, 0.1],
                  marker=dict(colors=colors, line=dict(color='#000000', width=1)))

st.write("### Proportion de 'TARGET'")
st.plotly_chart(fig)




colors = ['blue', 'pink']
fig = go.Figure(data=[go.Pie(labels=[ "Hommes", "Femmes"],
                             values=[65 , 35])])
fig.update_traces(hoverinfo='label+percent', textinfo='label+percent', textfont_size=20,pull=[0, 0.05],
                  marker=dict(colors=colors, line=dict(color='#000000', width=1)))
st.write("### Proportion de 'GENDRE'")
st.plotly_chart(fig)









## Afficher les données client sélectionnées (condition de checkbox : 'Données client') ##
if client_data:
    st.write(f'### Données du client, demande {selected_credit}')
    

    ### définir les valeurs à afficher pour l'histogramme et les données client (avec un maximum à 5) ###
    selections_client0 = st.multiselect('Vous pouvez afficher 5 données maximum parmi cette liste:', df_test_normalize[features].columns.tolist(),
    df_test_normalize[features].columns.tolist()[1:5])
    ### définir des colonnes pour diviser un visuel en deux  ###
    col1, col2 = st.columns(2)
    ### Afficher les informations client concernant les features sélectionnées ###
    col1.dataframe(df_test_normalize.loc[selected_credit, selections_client0])
    ### définir pyplot pour col2 barchart avec les informations sur les passagers sélectionnés avec condition du nombre de features sélectionnées ###
    
    if len(selections_client0) <= 5:
        



        dict_tmp = {'Features': df_test_normalize[features].loc[selected_credit, selections_client0].index,
                    'Valeur normalisée': df_test_normalize[features].loc[selected_credit, selections_client0].values}
        
        df_temp = pd.DataFrame(dict_tmp)
        fig = px.bar(        
           df_temp,
           x = "Features",
           y = "Valeur normalisée",
           title = f'Diagramme bar données ID: {selected_credit}',hover_data= ['Features', 'Valeur normalisée'] 
          )
        st.plotly_chart(fig)

    else:
        col2.write("Vous avez sélectionné trop de feature!!! Le graphique n'a pas pu être affiché")

    ### ajouter un extenseur pour plus d'explications sur les données client sélectionnées  ###
    with st.expander('Informations complémentaires'):
        st.write(""" Ici vous trouvez les informations client disponibles pour la demande de prêt sélectionnée.  \n"""
            """ La graphique en bâton donne les valeurs de features (variables) normalisées pour pouvoir les afficher sur la même échelle. """)





## Afficher la réponse du prêt concernant le calcul de probabilité du modèle ( via API Flask pour obtenir le résultat / checbox condition : 'Résultat de la demande de prêt') ##
if client_pred_score:
    st.write('### Décision sur la demande de prêt')
    ###  attention l'url de l'API doit être changée pour le déploiement en série !!###
    url_api_model_result = 'https://mankarali07api.herokuapp.com/scores'
    ### Faites attention aux paramètres, avec doit avoir un dict avec la valeur de prêt d'index / ID. C'est ainsi qu'il est implémenté dans notre API ###
    get_request = requests.get(url=url_api_model_result, params={'index': selected_credit})
    ### Nous obtenons les informations de prédiction à partir du format json du modèle d'API  ###
    prediction_value = get_request.json()['Credit_score']
    ### Nous obtenons la réponse concernant l'acceptation du prêt ######
    #answer_value = bool(get_request.json()['Answer'])
    ### Afficher les résultats ###
    st.write(f'Demande de prêt ID: {selected_credit}')
    st.write(f'Probabilité de défauts de remboursement: {prediction_value*100:.2f} %')
    if prediction_value < treshold :
        st.write('Demande de prêt acceptée!')
    else:
        #### ajouter une autre condition en fonction de la valeur de la prédiction ####
        if prediction_value > treshold and prediction_value <= 0.53:
            st.write('Demande de prêt refusée --> à discuter avec le conseiller')
        else:
            st.write('Demande de prêt refusée!')
    ### ajouter une jauge pour la valeur de prédiction avec la bibliothèque plotly ###
    fig_gauge = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = float(f'{prediction_value*100:.1f}'),
    mode = "gauge+number+delta",
    title = {'text': "Score(%)"},
    delta = {'reference': treshold*100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
    gauge = {'axis': {'range': [0, 100]},
             'bar': {'color': 'black'},
             'steps' : [
                 {'range': [0, 30], 'color': "darkgreen"},
                 {'range': [30, (treshold*100)], 'color': "lightgreen"},
                 {'range': [(treshold*100),53], 'color': "orange"},
                 {'range': [53, 100], 'color':"red"}],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': treshold*100}}))
    st.plotly_chart(fig_gauge)
    ### ajouter un extenseur pour plus d'explications sur le résultat de la prédiction ###
    with st.expander('Informations complémentaires'):
        st.write(""" Le retour de l'API de prédiction donne un score entre 0 et 100% qui représente la probabilité de refus de prêt.  \n"""
            """ Trois cas de figure sont alors possibles:  \n """
            """ 1) Le score est en dessous de 50% → la demande de prêt est acceptée.  \n """
            """ 2) Le score est entre 50 et 53% → la demande de prêt est refusée 
            mais peut être discutée avec le conseiller pour éventuellement l'accepter 
            (grâce notamment aux onglets 'interprétations du score' et 'analyse des features clients').  \n"""
            """3) Le score est au dessus de 53% → la demande de prêt est refusée. """)





## Afficher la comparaison avec tous les clients et les clients proches dans le score(en utilisant la fonction créée pour filtrer les clients proches / checkbox: 'Analyse des features clients' ) ##
if client_analysis:
    st.write('### Analyse des features clients')
    
    ### Analyse univariée choisir le type de tracé (boîte à moustaches ou histogramme/barplot)  ###
    st.write('#### *Analyse univariée*')
    #### sélectionnez entre les distributions boxplot ou histogramme/barplot pour l'analyse univariée ####
    selected_anaysis_gh = st.selectbox('Sélectionner un graphique', ['Boxplot'])
    if selected_anaysis_gh == 'Boxplot':
        ##### Ajout de la possibilité d'afficher plusieurs entités sur la même parcelle #####
        selections_analysis = st.multiselect('Vous pouvez ajouter ou enlever une donnée présente dans cette liste:', df_test_normalize[features].columns.tolist(),
        df_test_normalize[features].columns.tolist()[0:5])
        ##### afficher la boîte à moustaches #####
        ###### créer dans chaque df une colonne pour les identifier et utiliser les paramètres de teinte ######
        df_test_normalize['data_origin'] = 'Tous les clients'
        
        ###### concaténer deux df avant de dessiner le boxplot ######
        cdf = pd.concat([df_test_normalize[selections_analysis + ['data_origin']] ])
        
        ###### Créer un DataFrame à partir de la série d'ID de prêt client sélectionnée ######
        df_loan = pd.DataFrame([df_test_normalize.loc[selected_credit, features].tolist()], columns=features)
        ###### en utilisant melt mehtod pour adapter notre dataframe concaténé au format que nous voulons (pour afficher plusieurs features) avec Seaborn ######
        cdf = pd.melt(cdf, id_vars='data_origin', var_name='Features')
        df_loan = pd.melt(df_loan[selections_analysis], var_name='Features')
        df_loan['data_origin'] = 'ID_prêt_client_selectionné'

        ###### affichage la figure ######
        figure_boxplot = plt.figure(figsize=(4,2))
        ax = sns.boxplot(x = 'Features', y = 'value', hue='data_origin', data=cdf , showfliers=False, palette = 'tab10')
        sns.stripplot(x = 'Features', y = 'value', data = df_loan, hue = 'data_origin', palette=['yellow'], s=8, linewidth=1.5, edgecolor='black')
        plt.xticks(fontsize=6, rotation=45)
        plt.yticks(fontsize=6)
        plt.ylabel('Valeur normalisée')
        leg = plt.legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ###### modifier l'objet de légende pour l'ID de prêt client sélectionné afin qu'il corresponde au style de graphique ######
        leg.legendHandles[-1].set_linewidth(1.5)
        leg.legendHandles[-1].set_edgecolor('black')
       
        fig = px.box(cdf, x = 'Features', y = 'value', points = False)
        fig.update_traces(quartilemethod="linear", jitter=0, col=1)
        
        st.plotly_chart(fig)
        st.pyplot(figure_boxplot, clear_figure=True)




        ###### ajouter un extenseur pour plus d'explications sur le nuage de points ######
        with st.expander('Informations complémentaires'):
            st.write(""" Ce boxplot permet d'afficher les distributions des groupes de clients en fonction de la valeur du client sélectionné.  \n"""
            """ Notez que les variables sont normalisées afin d'avoir une image de la situation de notre client par rapport aux autres groupes de clients.""")
   






    st.write('#### *Analyse bivariée*')
    # Relation Âge / Revenu Total Graphique interactif 
    data_sk = df_test.reset_index(drop=False)
    data_sk.DAYS_BIRTH = (abs(data_sk['DAYS_BIRTH']) / 365).round(1)
    fig, ax = plt.subplots(figsize=(10, 10))
    fig = px.scatter(data_sk, x='DAYS_BIRTH', y="AMT_INCOME_TOTAL",
                         size="AMT_INCOME_TOTAL", color='CODE_GENDER',
                         hover_data=['CNT_CHILDREN', 'SK_ID_CURR'])


    fig.update_layout({'plot_bgcolor': '#f0f0f0'},
                          title={'text': "Relation Âge / Revenu Total", 'x': 0.5, 'xanchor': 'center'},
                          title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))

    fig.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
    fig.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Age", title_font=dict(size=18, family='Verdana'))
    fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Income Total", title_font=dict(size=18, family='Verdana'))

    st.plotly_chart(fig)

    st.write("#### *L'effet des 20 features de SHAP*")
    st.write("##### *(A cause de limite de 500Mo de HEROKU, on n'affiche que l'image!!!)*")
    st.image(img2)
    st.write('#### *(Exemple) LOCAL Feature Importance *')
    st.write("##### *(A cause de limite de 500Mo de HEROKU, on n'affiche que l'image!!!)*")
    st.image(img3)





