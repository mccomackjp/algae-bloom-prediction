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
rainy_dic = {'var1': lambda x: 1 if x > 0 else 0}


'''
dictionary for 'Wind Speed'
Values taken from the Beaufort scale
'''

windspeed_dic = {'var0':  lambda x: "Calm"              if x < 0.5          else None,
                 'var1':  lambda x: "Light Air"         if 1.5 > x >= 0.5   else None,
                 'var2':  lambda x: "Light breeze"      if 3.3 > x >= 1.5   else None,
                 'var3':  lambda x: "Gentle Breeze"     if 5.5 > x >= 3.3   else None,
                 'var4':  lambda x: "Moderate Breeze"   if 7.9 > x >= 5.5   else None,
                 'var5':  lambda x: "Fresh Breeze"      if 10.7 > x >= 7.9  else None,
                 'var6':  lambda x: "Strong Breeze"     if 13.8 > x >= 10.7 else None,
                 'var7':  lambda x: "Near Gale"         if 17.1 > x >= 13.8 else None,
                 'var8':  lambda x: "Gale"              if 20.7 > x >= 17.1 else None,
                 'var9':  lambda x: "Strong Gale"       if 24.4 > x >= 20.7 else None,
                 'var10': lambda x: "Storm"             if 28.4 > x >= 24.4 else None,
                 'var11': lambda x: "Violent Storm"     if 32.6 > x >= 28.4 else None,
                 'var12': lambda x: "Hurricane Force"   if x >= 32.6        else None
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

winddirection_dic = {'var1':  lambda x: "N"   if x < 11         else None,
                     'var2':  lambda x: "NNE" if 33 > x >= 11   else None,
                     'var3':  lambda x: "NE"  if 56 > x >= 33   else None,
                     'var4':  lambda x: "ENE" if 78 > x >= 56   else None,
                     'var5':  lambda x: "E"   if 101 > x >= 78  else None,
                     'var6':  lambda x: "ESE" if 123 > x >= 101 else None,
                     'var7':  lambda x: "SE"  if 146 > x >= 123 else None,
                     'var8':  lambda x: "SSE" if 169 > x >= 146 else None,
                     'var9':  lambda x: "S"   if 191 > x >= 169 else None,
                     'var10': lambda x: "SSW" if 213 > x >= 191 else None,
                     'var11': lambda x: "SW"  if 236 > x >= 213 else None,
                     'var12': lambda x: "WSW" if 258 > x >= 236 else None,
                     'var13': lambda x: "W"   if 281 > x >= 258 else None,
                     'var14': lambda x: "WNW" if 303 > x >= 281 else None,
                     'var15': lambda x: "NW"  if 326 > x >= 303 else None,
                     'var16': lambda x: "NNW" if 348 > x >= 326 else None,
                     'var17': lambda x: "N"   if x >= 348        else None
                     }

'''
dictionary for the amount of precipitation tha that fallen. Categories taken from 
https://en.wikipedia.org/wiki/Rain#Intensity

units in mm
column = 'PRCP'
'''
precipitation_dic = {'var1': lambda x: 'None'       if x < 0.001        else None,
                     'var2': lambda x: 'Light'      if 2.5 > x >= 0.001 else None,
                     'var3': lambda x: 'Moderate'   if 7.6 > x >= 2.5   else None,
                     'var4': lambda x: 'Heavy'      if 50.0 > x >= 7.6  else None,
                     'var5': lambda x: 'Violent'    if  x >= 50.0       else None}

'''
dictionary for the amount of snow that has fallen

coulmn = 'SNOW'
'''
snow_dic = {'var1': lambda x: 'None'     if x < 0.001        else None,
            'var2': lambda x: 'Light'    if 2.0 > x >= 0.001 else None,
            'var3': lambda x: 'Moderate' if 5.0 > x >= 2.0   else None,
            'var4': lambda x: 'Heavy'    if x >= 5.0         else None}

'''
dictionary for the snow depth

column = 'SNWD'
'''
snow_depth_dic = {'var1': lambda x: 'None'      if x < 0.001         else None,
                  'var2': lambda x: 'Minimal'   if 5.0 > x >=  0.001 else None,
                  'var3': lambda x: 'Moderate'  if 10.0 > x >= 5.0   else None,
                  'var4': lambda x: 'Deep'      if x >= 10.0         else None}


'''
dictionary for the maximum air temperature

column = 'TMAX' or 'TMIN'
'''
temp_dic = {'var1': lambda x: 'Freezing'    if x < 0                else None,
            'var2': lambda x: 'Cold'        if 4.44 > x >= 0        else None,
            'var3': lambda x: 'Cool'        if 10.0 > x >= 4.44     else None,
            'var4': lambda x: 'Moderate'    if 15.56 > x >= 10.0    else None,
            'var5': lambda x: 'Room'        if 21.11 > x >=  15.56  else None,
            'var6': lambda x: 'Warm'        if 26.67 > x >= 21.11   else None,
            'var7': lambda x: 'Hot'         if 32.22 > x >= 26.67   else None,
            'var8': lambda x: 'Extreme'     if 37.78 > x >= 32.22   else None,
            'var9': lambda x: 'Dangerous'   if  x >= 37.78          else None
            }

'''
all of the dictionaries in one. Key = column name, value = dictionary defining categories
'''
all_dic = {'Wind Speed': windspeed_dic,
           'Wind Angle': winddirection_dic,
           'PRCP': precipitation_dic,
           'SNOW': snow_dic,
           'SNWD': snow_depth_dic,
           'TMAX': temp_dic,
           'TMIN': temp_dic}

