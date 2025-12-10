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
"""
| Country Code | Country Name | 
| 0 Afghanistan | 1 Albania | 2 Algeria 
| 3 American Samoa | 4 Angola 5 Anguilla
| 6 | Antigua and Barbuda | | 7 | Argentina | | 8 | Australia | 
| 9  Austria | | 10 | Azores Islands | | 11 | Bahamas (the) | 
| 12 | Bahrain | | 13 | Bangladesh | | 14 | Barbados | | 15 | Belgium | 
| 16 | Belize | | 17 | Benin | | 18 | Bermuda | | 19 | Bhutan | 
| 20 | Bolivia (Plurinational State of) | | 21 | Botswana | | 22 | Brazil | 
| 23 | Bulgaria | | 24 | Burkina Faso | | 25 | Burundi | | 26 | Cabo Verde | 
| 27 | Cambodia | | 28 | Cameroon | | 29 | Canada | | 30 | Central African Republic | 
| 31 | Chad | | 32 | Chile | | 33 | China | | 34 | Colombia | | 35 | Comoros (the) | | 36 
| Congo (the) | | 37 | Congo (the Democratic Republic of the) | | 38 | Cook Islands (the) | 
| 39 | Costa Rica | | 40 | Côte d’Ivoire | | 41 | Croatia | | 42 | Cuba | | 43 | Cyprus | 
| 44 | Czechoslovakia | | 45 | Denmark | | 46 | Djibouti | | 47 | Dominica | | 48 | Dominican Republic (the) | 
| 49 | Ecuador | | 50 | Egypt | | 51 | El Salvador | | 52 | Ethiopia | | 53 | Fiji | | 54 | Finland | 
| 55 | France | | 56 | French Polynesia | | 57 | Gabon | | 58 | Gambia (the) | | 59 | Germany | 
| 60 | Germany Dem Rep | | 61 | Germany Fed Rep | | 62 | Ghana | | 63 | Greece | | 64 | Grenada | 
| 65 | Guadeloupe | | 66 | Guam | | 67 | Guatemala | | 68 | Guinea | | 69 | Guinea-Bissau | 
| 70 | Guyana | | 71 | Haiti | | 72 | Honduras | | 73 | Hong Kong | | 74 | Hungary | | 75 | Iceland | 
| 76 | India | | 77 | Indonesia | | 78 | Iran (Islamic Republic of) | | 79 | Iraq | | 80 | Ireland | 
| 81 | Israel | | 82 | Italy | | 83 | Jamaica | | 84 | Japan | | 85 | Jordan | | 86 | Kenya | | 87 
| Kiribati | | 88 | Korea (the Democratic People's Republic of) | | 89 | Korea (the Republic of) | 
| 90 | Lao People's Democratic Republic (the) | | 91 | Lebanon | | 92 | Lesotho | | 93 | Liberia | 
| 94 | Libya | | 95 | Luxembourg | | 96 | Madagascar | | 97 | Malawi | | 98 | Malaysia | 
| 99 | Maldives | | 100 | Mali | | 101 | Martinique | | 102 | Mauritania | | 103 | Mauritius | 
| 104 | Mexico | | 105 | Micronesia (Federated States of) | | 106 | Mongolia | | 107 | Montserrat | 
| 108 | Morocco | | 109 | Mozambique | | 110 | Myanmar | | 111 | Namibia | | 112 | Nepal | 
| 113 | Netherlands (the) | | 114 | Netherlands Antilles | | 115 | New Caledonia | | 116 | New Zealand | 
| 117 | Nicaragua | | 118 | Niger (the) | | 119 | Nigeria | | 120 | Niue | | 121 | Norway | 
| 122 | Oman | | 123 | Pakistan | | 124 | Panama | | 125 | Papua New Guinea | | 126 | Paraguay | 
| 127 | Peru | | 128 | Philippines (the) | | 129 | Poland | | 130 | Portugal | | 131 | Puerto Rico | 
| 132 | Qatar | | 133 | Romania | | 134 | Russian Federation | | 135 | Rwanda | | 136 | Réunion | | 137 | Saint Kitts and Nevis | | 138 | Saint Lucia | | 139 | Saint Vincent and the Grenadines | 
| 140 | Samoa | | 141 | Sao Tome and Principe | | 142 | Saudi Arabia | | 143 | Senegal | | 144 | Sierra Leone | | 145 | Singapore | | 146 | Solomon Islands | 
| 147 | Somalia | | 148 | South Africa | | 149 | Soviet Union | | 150 | Spain | | 151 | Sri Lanka | | 152 | Sudan (the) | | 153 | Suriname | | 154 | Swaziland | 
| 155 | Sweden | | 156 | Switzerland | | 157 | Syrian Arab Republic | | 158 | Taiwan (Province of China) | 
| 159 | Tanzania, United Republic of | | 160 | Thailand | | 161 
| Timor-Leste | | 162 | Togo | | 163 | Tokelau | | 164 | Tonga | | 165 | Trinidad and Tobago | | 166 | Tunisia | 
| 167 | Turkey | | 168 | Turks and Caicos Islands (the) | | 169 | Tuvalu | | 170 | Uganda | 
| 171 | United Arab Emirates (the) | | 172 | United Kingdom of Great Britain and Northern Ireland (the) | | 173 | United States of America (the) | | 174 | Uruguay | 
| 175 | Vanuatu | | 176 | Venezuela (Bolivarian Republic of) | | 177 | Viet Nam | | 178 | Virgin Islands (British) | 
| 179 | Virgin Islands (U.S.) | | 180 | Wallis and Futuna | | 181 | Yemen Arab Rep | | 182 | Yemen P Dem Rep | | 183 | Yugoslavia | | 184 | Zambia | | 185 | Zimbabwe |
"""


