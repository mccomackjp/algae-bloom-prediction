"""
The purpose of this entire file is to keep a single file that will hold all of the data dictionaries in order to bucket
the individual features. Keeping them all in one file will help keep maintenance easier.

Before each of the individual dictionaries, please keep the column in teh dictionary they are associated with.

Note: the inner variable does not matter but each of the dictionary keys does not matter but the all cases must be
considered. If using a lambda function you need if and an Else because Lambdas NEED to return something.
"""

'''
dictionary for 'PRCP' column
'''
rainy_dict = {'Rained_bucket': lambda x: 1 if x > 0 else 0}

'''
dictionary for 'Wind Speed'
Values taken from the Beaufort scale
'''

windspeed_dict = {'Calm_bucket': lambda x: "Calm" if x < 0.5 else None,
                  'Light_Air_bucket': lambda x: "Light Air" if 1.5 > x >= 0.5 else None,
                  'Light_breeze_bucket': lambda x: "Light breeze" if 3.3 > x >= 1.5 else None,
                  'Gentle_Breeze_bucket': lambda x: "Gentle Breeze" if 5.5 > x >= 3.3 else None,
                  'Moderate_Breeze_bucket': lambda x: "Moderate Breeze" if 7.9 > x >= 5.5 else None,
                  'Fresh_Breeze_bucket': lambda x: "Fresh Breeze" if 10.7 > x >= 7.9 else None,
                  'Strong_Breeze_bucket': lambda x: "Strong Breeze" if 13.8 > x >= 10.7 else None,
                  'Near_Gale_bucket': lambda x: "Near Gale" if 17.1 > x >= 13.8 else None,
                  'Gale_bucket': lambda x: "Gale" if 20.7 > x >= 17.1 else None,
                  'Strong_Gale_bucket': lambda x: "Strong Gale" if 24.4 > x >= 20.7 else None,
                  'Storm_bucket': lambda x: "Storm" if 28.4 > x >= 24.4 else None,
                  'Violent_Storm_bucket': lambda x: "Violent Storm" if 32.6 > x >= 28.4 else None,
                  'Hurricane_Force_bucket': lambda x: "Hurricane Force" if x >= 32.6 else None
                  }

'''
Values taken from http://snowfence.umn.edu/Components/winddirectionanddegreeswithouttable3.htm
N   348.75 - 11.25
NNE 11.25 - 33.75
NE  33.75 - 56.25
ENE 56.25 - 78.75
E   78.75 - 101.25
ESE 101.25 - 123.75
SE  123.75 - 146.25
SSE 146.25 - 168.75
S   168.75 - 191.25
SSW 191.25 - 213.75
SW  213.75 - 236.25
WSW 236.25 - 258.75
W   258.75 - 281.25
WNW 281.25 - 303.75
NW  303.75 - 326.25
NNW 326.25 - 348.75

dictionary for 'Wind Direction'

'''

winddirection_dict = {'N_bucket_bottom': lambda x: "N" if x < 11 else None,
                      'NNE_bucket': lambda x: "NNE" if 33 > x >= 11 else None,
                      'NE_bucket': lambda x: "NE" if 56 > x >= 33 else None,
                      'ENE_bucket': lambda x: "ENE" if 78 > x >= 56 else None,
                      'E_bucket': lambda x: "E" if 101 > x >= 78 else None,
                      'ESE_bucket': lambda x: "ESE" if 123 > x >= 101 else None,
                      'SE_bucket': lambda x: "SE" if 146 > x >= 123 else None,
                      'SSE_bucket': lambda x: "SSE" if 169 > x >= 146 else None,
                      'S_bucket': lambda x: "S" if 191 > x >= 169 else None,
                      'SSW_bucket': lambda x: "SSW" if 213 > x >= 191 else None,
                      'SW_bucket': lambda x: "SW" if 236 > x >= 213 else None,
                      'WSW_bucket': lambda x: "WSW" if 258 > x >= 236 else None,
                      'W_bucket': lambda x: "W" if 281 > x >= 258 else None,
                      'WNW_bucket': lambda x: "WNW" if 303 > x >= 281 else None,
                      'NW_bucket': lambda x: "NW" if 326 > x >= 303 else None,
                      'NNW_bucket': lambda x: "NNW" if 348 > x >= 326 else None,
                      'N_bucket_top': lambda x: "N" if x >= 348 else None
                      }

'''
dictionary for the amount of precipitation tha that fallen. Categories taken from 
https://en.wikipedia.org/wiki/Rain#Intensity

units in mm
column = 'PRCP'
'''
precipitation_dict = {'None_bucket': lambda x: 'None' if x < 0.001 else None,
                      'Light_bucket': lambda x: 'Light' if 2.5 > x >= 0.001 else None,
                      'Moderate_bucket': lambda x: 'Moderate' if 7.6 > x >= 2.5 else None,
                      'Heavy_bucket': lambda x: 'Heavy' if 50.0 > x >= 7.6 else None,
                      'Violent_bucket': lambda x: 'Violent' if x >= 50.0 else None}

'''
dictionary for the amount of snow that has fallen

coulmn = 'SNOW'
'''
snow_dict = {'None_bucket': lambda x: 'None' if x < 0.001 else None,
             'Light_bucket': lambda x: 'Light' if 2.0 > x >= 0.001 else None,
             'Moderate_bucket': lambda x: 'Moderate' if 5.0 > x >= 2.0 else None,
             'Heavy_bucket': lambda x: 'Heavy' if x >= 5.0 else None}

'''
dictionary for the snow depth

column = 'SNWD'
'''
snow_depth_dict = {'None_bucket': lambda x: 'None' if x < 0.001 else None,
                   'Minimal_bucket': lambda x: 'Minimal' if 5.0 > x >= 0.001 else None,
                   'Moderate_bucket': lambda x: 'Moderate' if 10.0 > x >= 5.0 else None,
                   'Deep_bucket': lambda x: 'Deep' if x >= 10.0 else None}

'''
dictionary for the maximum air temperature

column = 'TMAX' or 'TMIN'
'''
temp_dict = {'Freezing_bucket': lambda x: 'Freezing' if x < 0 else None,
             'Cold_bucket': lambda x: 'Cold' if 4.44 > x >= 0 else None,
             'Cool_bucket': lambda x: 'Cool' if 10.0 > x >= 4.44 else None,
             'Moderate_bucket': lambda x: 'Moderate' if 15.56 > x >= 10.0 else None,
             'Room_bucket': lambda x: 'Room' if 21.11 > x >= 15.56 else None,
             'Warm_bucket': lambda x: 'Warm' if 26.67 > x >= 21.11 else None,
             'Hot_bucket': lambda x: 'Hot' if 32.22 > x >= 26.67 else None,
             'Extreme_bucket': lambda x: 'Extreme' if 37.78 > x >= 32.22 else None,
             'Dangerous_bucket': lambda x: 'Dangerous' if x >= 37.78 else None
             }

'''
all of the dictionaries in one. Key = column name, value = dictionary defining categories
'''
all_dict = {'Wind Speed': windspeed_dict,
            'Wind Angle': winddirection_dict,
            'PRCP': precipitation_dict,
            'SNOW': snow_dict,
            'SNWD': snow_depth_dict,
            'TMAX': temp_dict,
            'TMIN': temp_dict}
