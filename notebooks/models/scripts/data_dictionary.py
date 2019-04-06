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

windspeed_dic = {'var0':lambda x: "Calm" if x < 0.3 else None,
                 'var1':lambda x: "Light Air" if x < 1.5 else None,
                 'var2':lambda x: "Light breeze" if x < 3.3 else None,
                 'var3':lambda x: "Gentle Breeze"if x < 5.5 else None,
                 'var4':lambda x: "Moderate Breeze"if x < 7.9 else None,
                 'var5':lambda x: "Fresh Breeze"if x < 10.7 else None,
                 'var6':lambda x: "Strong Breeze" if x < 13.8 else None,
                 'var7':lambda x: "Near Gale" if x < 17.1 else None,
                 'var8':lambda x: "Gale" if x < 20.7 else None,
                 'var9':lambda x: "Strong Gale"if x < 24.4 else None,
                 'var10':lambda x: "Storm"if x < 28.4 else None,
                 'var11':lambda x: "Violent Storm"if x < 32.6 else "Hurricane Force"
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

winddirection_dic = {'var1':lambda x: "N"   if x < 11 else None,
                    'var2': lambda x: "NNE" if x < 33 else None,
                    'var3': lambda x: "NE"  if x < 56 else None,
                    'var4': lambda x: "ENE" if x < 78 else None,
                    'var5': lambda x: "E"   if x < 101 else None,
                    'var6': lambda x: "ESE" if x < 123 else None,
                    'var7': lambda x: "SE"  if x < 146 else None,
                    'var8': lambda x: "SSE" if x < 169 else None,
                    'var9': lambda x: "S"   if x < 191 else None,
                    'var10':lambda x: "SSW" if x < 213 else None,
                    'var11':lambda x: "SW"  if x < 236 else None,
                    'var12':lambda x: "WSW" if x < 258 else None,
                    'var13':lambda x: "W"   if x < 281 else None,
                    'var14':lambda x: "WNW" if x < 303 else None,
                    'var15':lambda x: "NW"  if x < 326 else None,
                    'var16':lambda x: "NNW" if x < 348 else "N",
                 }

