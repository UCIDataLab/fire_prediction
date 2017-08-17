## Open Questions About Data and Timing (v1.0)
Casey A Graff
August 10th, 2017


1. #### MODIS (Moderate Resolution Imaging Spectroradiometer)
	1. Timing of measurements
		1. What times of day (in local time) do the Aqua & Terra satellites fly over Alaska
		2. Difference between day/night detections
		3. **Plot histogram of measurements binned by time (and color-coded by satellite)**
			1. Possibly compare northern vs. southern Alaska
	2. Confidence
		1. How much does confidence vary and is it useful
		2. **Plot histogram of confidences**
			1. Include equations for confidence from Giglio et al. 2003
		3. Does confidence have patterns spatially (e.g. lower confidence close to the shore)
			1. **Plot confidences on map**
		4. Does confidence have a correlation with the sample #
			1. **Plot confidence vs sample #**
	3. FRP Fire Radiated Power
		1. Can this be used as a feature (or as the target variable)
	4. Fire Type
		1. Is the type being used to filter out non-vegetation fires (volcano, offshore)
			1. **Currently the type is not being used**
	5. Is leap year being accounted for in the MODIS data?
		1. **Look for a detection on the 29th day of February**
	6. Is there a correlation between FRP & confidence
		1. **Calculate correlation between the two measures**
		2. There should be a correlation because the equation for confidence is a function of the temperature of the fire
	7. Possible double counting
		1. When Aqua  and Terra overlap in the mid-day do they double count the same fire pixel
		2. **Measure spatial distance between fire detections and nearest neighbor (within a day)**
2.  #### GFS (Global Forecast System)
	1. Missing Data
		1. **Explore missing data between years, months; inside and outside fire season**
	2. Spatial correlation of data
		1. Is there a strong correlation between the grid cells (temperature, humidity, wind, rain)
		2. Should weather measurements be taken an interpolation of nearby cells
		3. **Spatially plot the correlation between cells on the map**
			1. Create 8 map plots; one for the correlation between the center grid and each adjacent cell
	3. At what times are instantaneous vs. integrated variables available
		1. **Explore the grb2 files used for GFS**
	4. Is the rain measurement done at the correct time/day
		1. Is the T-2 offset for rain due to the physical properties of the system or a consequence of an error in the code
		2. **Explore code for possible timing issues**
		3. **Plot rain (and other weather variables) alongside fire detections for individual fire events**
	5. Are there other weather forecast datasets that are better
3. #### Station LCD (Local Climatological Data)
	1. Measuring performance of GFS
		1. Can we use station data to investigate any lag/lead in GFS data
		2. How can we use station data to investigate inaccuracy of GFS data
		3. **Plot station measurements alongside the nearest GFS pixel's measurements**
	2. Integration with GFS data
		1. Can station data be integrated with GFS data for more accurate weather measurement/forecasting
4. #### Timing (between multiple datasets)
	1. When to cut-off fire predictions for day T vs T+1
		1. Will the prediction be done in the morning of day T+1 or the evening of day T 
		2. **Plot satellite measurements and weather measurements in local time**
	2. When should the weather measurements be taken
		1. Could be taken in the middle of the day (local time) or before the first satellite pass
		2. **Run prediction using several possible times**