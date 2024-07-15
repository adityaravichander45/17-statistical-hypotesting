import numpy as np
import pandas as pd
import streamlit as st
import joblib

model=joblib.load('Linear_reg_model.pkl')

df=pd.read_csv('dataset_auto.csv')


col_names=['symboling', 'normalised_losses', 'make', 'fuel_type', 'aspiration','num_of_doors', 'car_style', 'drive_wheels', 'engine_location','wheel_base', 
           'length', 'width', 'height', 'curb_weight', 'engine_type','num_of_cylinders', 'engine_size', 'fuel_system', 'bore', 'stroke', 'compression_ratio', 'horse_power', 'peak_rpm', 'city_mpg','highway_mpg', 'price']


def predict_price(symboling,normalised_losses,make,fuel_type,aspiration,num_of_doors,car_style,drive_wheels,engine_location,wheel_base,length,width,height,curb_weight,engine_type,num_of_cylinders,engine_size,fuel_system,bore,stroke,compression_ratio,horse_power,peak_rpm,city_mpg,highway_mpg,price):
      new=[symboling,normalised_losses,make,fuel_type,aspiration,num_of_doors,car_style,drive_wheels,engine_location,wheel_base,length,width,height,curb_weight,engine_type,num_of_cylinders,engine_size,
           fuel_system,bore,stroke,compression_ratio,horse_power,peak_rpm,city_mpg,highway_mpg,price]   
      test=pd.DataFrame([new])
      test.columns=col_names
      model.predict(test)
      return model.predict(test)


def main():
    
    st.title('Automobile price prediction')
    
    temp="""
    <div style='background-color:blue;'>
    <h2 style='color:black;text-align:centre;'>Predict the price</h2>
    </div>
    """

    st.markdown(temp,unsafe_allow_html=True)
    
    symboling=st.select_slider('Choose the car symbol',options=[-3, -2, -1, 0, 1, 2, 3])
    st.write('You choose',symboling)
    normalised_losses=st.number_input('Write down the loss')
    st.write('You choose',normalised_losses)
    make=st.selectbox('Choose the model',pd.unique(df['make']))
    fuel_type=st.radio('Select the fuel type',pd.unique(df['fuel_type']))
    aspiration=st.radio('Select the aspiration',pd.unique(df['aspiration']))
    num_of_doors=st.radio('Select the number of doors',pd.unique(df['num_of_doors']))
    car_style=st.selectbox('Select the car-style',pd.unique(df['car_style']))
    drive_wheels=st.radio('Select the type of drive-wheels',pd.unique(df['drive_wheels']))
    engine_location=st.selectbox('Select the location of the engine',pd.unique(df['engine_location']))
    wheel_base=st.number_input('Write down the base of the wheel')
    st.write('Ýou selected',wheel_base)
    length=st.number_input('Write down the length')
    st.write('You choose',length)
    width=st.number_input('Write down the width')
    st.write('You choose',width)
    height=st.number_input('Write down the height')
    st.write('You choose',height)
    curb_weight=st.number_input('Write down the curb-weight')
    st.write('You choose',curb_weight)
    engine_type=st.selectbox('choose the engine-type',pd.unique(df['engine_type']))
    num_of_cylinders=st.selectbox('Select the number of cylinders',pd.unique(df['num_of_cylinders']))
    engine_size=st.number_input('Write down the size of the engine')
    st.write('You choose',engine_size)
    fuel_system=st.selectbox('Select the system of the fuel',pd.unique(df['fuel_system']))
    bore=st.number_input('Write down the bore size')
    st.write('You choose',bore)
    stroke=st.number_input('Write down the stroke')
    st.write('You choose',stroke)
    compression_ratio=st.number_input('Write down the compression-ratio')
    st.write('You choose',compression_ratio)
    horse_power=st.number_input('Write down the horsepower')
    st.write('You choose',horse_power)
    peak_rpm=st.number_input('Write down the peak-rpm')
    st.write('You choose',peak_rpm)
    city_mpg=st.number_input('Write down the city-mpg')
    st.write('You choose',city_mpg)
    highway_mpg=st.number_input('Write down the highway-mpg')
    st.write('You choose',highway_mpg)
    price=st.number_input('Write down the price')
    st.write('You choose',price)
    
    if st.button('Price'):
        predicted_price=predict_price(symboling,normalised_losses,make,fuel_type,aspiration,num_of_doors,car_style,drive_wheels,engine_location,wheel_base,length,width,height,curb_weight,engine_type,num_of_cylinders,engine_size,fuel_system,bore,stroke,compression_ratio,horse_power,peak_rpm,city_mpg,highway_mpg,price)
        st.write(predicted_price)



if __name__=='__main__':
    main()
