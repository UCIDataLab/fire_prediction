## Progress Report (9-1-2017) v1.0
Casey A Graff
September 1, 2017



1. #### Last Time
	1. Covered "Open Questions About Data and Timing v1.0"
	2. MODIS	
		1. Possibly remove nighttime detections (outside 10 a.m. to 3 p.m. local)
		2. Likely filter out low confidence detections
		3. Consider predicting (mean FRP * # detections)
	3. GFS
		1. Contact NOAA regarding missing data
		2. Consider GFS-ANL 1.0 degree
		3. Plot "global" fire detections against weather (Wiggins et al. 2016)
2. #### Completed Work
	1. Completed all stages of data pipeline (except prediction)
>		1. Data
>			a. GFS: FTP Fetch (8.5 hours/year) --> Extract (.5 hours/year) -->  Aggregate (5 min.) -->
>			b. MODIS: FTP Fetch (15 min.) --> Extract/Aggregate (10 min.) -->
>		2. Processing
>			a. GFS: --> Processing (1 min.) --> 
>			b. MODIS: --> Clustering (40 min.) --> 
>		3, Integration 
>			a. GFS/MODIS: --> Integration (3 min.) -->
>		3. Prediction
>			a. GFS/MODIS: --> Prediction
	
	2. Missing GFS-ANL .5 data cannot be recovered from any online source
		1. Not available on any of the NCDC sources (FTP, HTTPS, TDS, HAS)
		2. Not available on NCAR archives
		3. **Could go back to NOAA GFS team**
		
	3. Differences with original implementation
		1. Using "true" local time for seasonal and daily cutoff
			a. hour offset = (longitude * (12 / pi)) / 60
		2. Using spherical distance model for clustering
		3. Excluding non-vegetation fires
		4. (Possibly) Time of day for weather measurements
	
	4. Continued data exploration
		1. Spatial weather correlation
		2. GFS +0, +3, +6 Offset data
		3. Detections vs. weather comparison (w/ re-extracted data)
			1. Per cluster and per year
		4. MODIS Confidence spatial plotting
		5. MODIS Aqua/Terra Daily Overlap
		6. MODIS Day vs Night %

3. #### Questions
	1. Is the lat/lon for GFS pixels the center of the .5 degree pixel or one of the corners?
	2. Are there other GFS variables (outside of rain, wind, humidity and temperature) that may be useful for prediction?
		1. Water equivalent of accumulated snow dept
		2. Max/Min temperature
		3. Water runoff
	3. Is all of the ERA-Interim data available?
	4. Grid-based clustering or spherical model-based?
	5. If we throw out nighttime detections do we still use them for clustering?
	6. Time of day to collect weather measurements?
		1. If we base it on local time it may be between two measurement intervals
>		If our target time is 12:00 p.m. local.
>			@ -8:45 timezone shift: closest measurement is 9:15 a.m. local (18:00 UTC)
>			@ -9:00 timezone shift: closest measurement is 9:00 a.m or 3:00 p.m. local (18:00 or 00:00 UTC)
>			@ -9:15 timezone shift: closest measurement is 2:45 p.m. local (00:00 UTC)
	
		2. Should be interpolated between two closest times?

4. #### Next Steps
	1. Finish prediction pipeline	
	2. Comparison of Replacement Weather Data (against LCD data)
		1. GFS-ANL 1.0 degree
		2. ERA-Interim
	
	3. Finish data exploration
		1. Weather vs. fire detections
			1. Inter-day weather vs. fire detections time plot
			2. Correlation for time-shifted weather vs. fire detections
		2. Compare correlation of nearby stations
			1. Check if the correlation between adjacent GFS pixels is 'overly' smooth due to the modeling
			2. Fairbanks Intl. Airport vs. Fairbanks Wainwright
		3. MODIS Overlap Comparison
			1. Mean nearest neighbor distance between detections within cluster

		
		