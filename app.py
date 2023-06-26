import streamlit as st
import pandas as pd
import joblib

mpg = pd.read_csv("auto-mpg.csv")

st.header("MPG Prediction Model")

def main():
    option = st.sidebar.selectbox("Select ",["View Data","Make A Prediction","Visualize"])
    if option=="View Data":
        data()
    elif option=="Make A Prediction":
        prediction()
    else:
        visualize()
    

def data():
    st.dataframe(data=mpg)

def prediction():
    cylinder = st.number_input("Enter No. of cylinder",min_value=0)
    displacement = st.number_input("Enter displacement value")
    weight = st.number_input("Enter Weight",min_value=0)
    acceleration = st.number_input("Enter Acceleration")
    model_year = st.number_input("Enter Model Year",min_value=0)
    predict = st.button("Predict MPG")
    if predict:
        model = joblib.load("mymodel.h5")
        y_pred = model.predict([[cylinder,displacement,weight,acceleration,model_year]])
        st.success(y_pred)
        st.write("Predicted With an accuracy of 68%")
        
def visualize():
    st.line_chart(data=mpg,x="horsepower",y="acceleration")     

if __name__ == '__main__':
    main()
