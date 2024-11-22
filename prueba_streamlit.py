import pandas as pd
import numpy as np
import streamlit as st
import pickle


# Cargar el modelo desde el archivo .pkl
with open('modelo_regresion_lineal.pkl', 'rb') as file:
    modelo = pickle.load(file)

# Título de la aplicación
st.title("Ventas de café")

# Solicitar al usuario que ingrese el número de horas estudiadas
horas = st.number_input("Por favor, ingresa el número de productos:", min_value=0.0, step=0.1)

# Realizar la predicción
if st.button("Predecir cantidad de ventas"):
    prediccion = modelo.predict([[horas]])
    st.write(f"La cantidad predicha es: {prediccion[0]:.2f}")