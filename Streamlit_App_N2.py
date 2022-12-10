import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from PIL import Image

# Set title 

st.title('Data Science App')

image = Image.open('christopher-burns-Kj2SaNHG-hg-unsplash.jpg')
st.image(image)

# Subtitle

st.write('## Streamlit App N2 with Streamlit')

st.write('### Exploring different classifiers')

st.text('---'*30)

def main():
    activities=['EDA', 'Visualization', 'Model', 'About Us']
    options=st.sidebar.selectbox('Activities selection: ', activities)

    data = st.file_uploader('Please, upload your file', type=['csv', 'xlsx', 'json', 'txt'])

    if data is not None:
        st.success('File successfully uploaded')
        df = pd.read_csv(data)
        st.dataframe(df.head(20))

# EDA PART

    if options=='EDA':
        st.subheader('Exploratory data analysis')


        if st.checkbox('Shape'):
                st.write(df.shape)

        if st.checkbox('Data types'):
                st.write(df.dtypes)
              
        if st.checkbox('Null values'):
                st.write(df.isnull().sum())

        if st.checkbox('Correlation'):
                st.write(df.corr())    

        if st.checkbox('Columns'):
                st.write(df.columns)

        if st.checkbox('Select columns'):
                selected_columns = st.multiselect('Columns: ', df.columns) 
                df_selected_columns= df[selected_columns]
                st.dataframe(df_selected_columns)
                 
        if st.checkbox('Summary'):
                st.write(df_selected_columns.describe().T)

# VISUALIZATION PART
    
    elif options=='Visualization':
        st.subheader('Data Visualization')

        if st.checkbox('Select columns to plot'):
                selected_columns = st.multiselect('Columns: ', df.columns) 
                df_selected_columns= df[selected_columns]
                st.dataframe(df_selected_columns)

        if st.checkbox('Heatmap'):
                fig = plt.figure()
                sns.heatmap(df_selected_columns.corr(), vmax=1, square=True, annot=True, cmap='mako')
                st.pyplot(fig)

        if st.checkbox('Pair plot'):                
                fig = sns.pairplot(data=df_selected_columns, diag_kind='kde')
                st.pyplot(fig)        


# MODEL BUILDING

    elif options=='Model':
        st.subheader('Model Building')

        if st.checkbox('Select columns'):
                selected_data = st.multiselect('Columns (target variable need to be the last column to be selected): ', df.columns) 
                df_selected_data = df[selected_data]
                st.dataframe(df_selected_data)

                # Dividing the x's and the y
                X = df_selected_data.iloc[:, 0:-1]
                y = df_selected_data.iloc[:,-1]

        seed = st.sidebar.slider('Seed', 1, 200)
                
        classifier_name = st.sidebar.selectbox('Select the classifier:', ('KNN', 'SVM', 'Logistic', 'Naive Bayes', 'Decision Tree'))

        def add_parameter(name_of_clf):
                params = dict()
                
                if name_of_clf == 'KNN':
                        K = st.sidebar.slider('K', 1, 15)
                        params['K'] = K

                else:
                        name_of_clf == 'SVM'
                        C = st.sidebar.slider('C', 0.01, 15.0)
                        params['C'] = C

                return params

        params = add_parameter(classifier_name)        

        def get_classifier(name_of_clf, params):
                clf = None

                if name_of_clf == "KNN":
                        clf = KNeighborsClassifier(n_neighbors=params['K'])

                elif name_of_clf == "SVM":
                        clf = SVC(C=params['C'])

                elif name_of_clf == "Logistic":
                        clf = LogisticRegression()

                elif name_of_clf == "Naive Bayes":
                        clf = GaussianNB()

                elif name_of_clf == "Decision Tree":
                        clf = DecisionTreeClassifier()

                else:
                        st.warning('Choose one classifier')

                return clf

        # Calling the function
        clf = get_classifier(classifier_name, params) 


        # Running the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        st.write('Classifier name:', classifier_name)
        st.write('Accuracy score:', accuracy)

    elif options=='About Us':
        st.markdown('## This is an interactive app for Data Science')
        
    
if __name__ == '__main__':
    main()    