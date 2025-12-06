from collections import Counter
import joblib

model = joblib.load('ML_Model/disaster_prediction_model.joblib')
#input variables
Year=int(input('input the year : '))
DMS = float(input("estimate magnitude scale : "))
DMV = int(input("estimate magnitude value : "))
Country=int(input("country id : "))
Lon=float(input("input coordinates Longitude : "))
Lat = float(input("input coordinates Longitude : "))

X_new = [[Year,DMS,DMV,Country,Lon,Lat]]
# Make prediction
"""
| Disaster Type Code | Disaster Type             | 
| :----------------- | :------------------------ | 
| 0                  | Drought                   | 
| 1                  | Earthquake                | 
| 2                  | Epidemic                  | 
| 3                  | Extreme temperature       | 
| 4                  | Flood                     | 
| 5                  | Fog                       | 
| 6                  | Impact                    | 
| 7                  | Insect infestation        | 
| 8                  | Mass movement (dry)       | 
| 9                  | Mass movement (wet)       | 
| 10                 | Storm                     | 
| 11                 | Volcanic activity         | 
| 12                 | Wildfire                  |
"""
prediction = model.predict(X_new)
if(prediction==0):
    print("There might be a Drought")
elif(prediction==1):
    print("There might be an Earthquake")
elif(prediction==2):
    print("There might be an Epidemic")
elif(prediction==3):
    print("There might be an Extreme temperature Wave")
elif(prediction==4):
    print("There might be a Flood")
elif(prediction==5):
    print("There might be Fog")
elif(prediction==6):
    print("There might be an Impact")
elif(prediction==7):
    print("There might be an Insect infestation")
elif(prediction==8):
    print("There might be a Mass movement (dry)")
elif(prediction==9):
    print("There might be a Mass movement (wet)")
elif(prediction==10):
    print("There might be a Storm")
elif(prediction==11):
    print("There might be Volcanic activity")
elif(prediction==12):
    print("Threre might be a Wildfire")


