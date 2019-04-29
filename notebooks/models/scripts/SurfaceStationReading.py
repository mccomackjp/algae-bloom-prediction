import datetime


class SurfaceStationReading:
    """
    All of these values are taken from the the NOAA document describing the ISD (integrated surface Data) format.
    The acronyms are as follows:
        GRT - Geophysical Report Type
        GPO - Geophysical point observation
        FWS - Fixed weather station
        MPO - Meteorological point observation
        WO  - Wind observation
        SCO - Sky condition observation
        VO  - Visibility observation
        ATO - Air Temperature observation (degrees celsius)
        APO - Atmospheric pressure observation
    """
    def __init__(self, reading):
        """
        The initializing of the reading for the station
        :param reading: the reading in ISD  format described by NOAA
        """
        self.total_chars                = 105 + int(reading[0:4])
        self.FWS_usaf_ident             = reading[4:9]
        self.FWS_ncei_ident             = reading[10:15]
        self.GPO_timestamp              = self._timestamp(reading[15:23], reading[23:27])
        self.GPO_source_flag            = reading[27]
        self.GPO_lat                    = reading[28:34]
        self.GPO_lon                    = reading[34:41]
        self.GRT_code                   = reading[41:46]
        self.GPO_ele                    = reading[46:51]
        self.FWS_call_letter_ident      = reading[51:56]
        self.MPO_process_name           = reading[56:60]
        self.WO_wind_angle              = reading[60:63]
        self.WO_wind_angle_qual         = reading[63]
        self.WO_wind_angle_type         = reading[64]
        self.WO_wind_speed              = reading[65:69]
        self.WO_wind_speed_qual         = reading[69]
        self.SCO_ceiling_height         = reading[70:75]
        self.SCO_ceiling_qual           = reading[75]
        self.SCO_ceiling_determination = reading[76]
        self.SCO_covak_code             = reading[77]
        self.VO_distance                = reading[78:84]
        self.VO_distance_qual           = reading[84]
        self.VO_variability_code        = reading[85]
        self.VO_quality_var_code        = reading[86]
        self.ATO_air_temp               = reading[87:92]
        self.ATO_air_qual_code          = reading[92]
        self.ATO_dew_pt                 = reading[93:98]
        self.ATO_dew_pt_qual_code       = reading[98]
        self.APO_sea_lvl_pres           = reading[99:104]
        self.APO_sea_lvl_pres_qual_code = reading[104]

    def _timestamp(self, date, time):
        """
        Crates a time stamp with based on the date and time string passed in
        :param date: the string of the date to be used in the format YYYYMMDD
        :param time: the string of the time to be used in the format HHSS
        :return: a date time object for the date and time passed in
        """
        return datetime.datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), int(time[0:2]), int(time[2:4]))