fire_prediction
===============
Using MODIS active fire data and ERA Interim (or GFS) data to predict the spread of forest fires, particularly in Alaska.

After fetching the ERA data, use run_pred.py to fetch the MODIS data, process the MODIS and ERA data, and train the model.

The project relies of Luigi (a project from Spotify) for creating and running pipelines. This helps simplify getting
started by combining data processing with model training (when the processed data isn't found on the system). For more
information on Luigi and how it works see https://github.com/spotify/luigi.

Detailed Project Organization
-----------------------------
* run_pred.py           <- used to train models; includes parameter configuration (for data/model specification) and starts the training pipeline
* data
    - raw               <- Original, immutable data
    - interim           <- Intermediate data that has gone through some transformation
    - processed         <- Terminal form of the data; ready for modeling
    - exploratory       <- Transformed data that is not intended for modeling
* notebooks             <- Jupyter notebooks. Format is "<description/title>-<versioning #>-<author identifier>"
    - exploratory       <- Notebooks for exploring data and results
    - reports           <- More polished notebooks that can be exported to reports
    - tutorial          <- Notebooks that are instructive for learning about the project
* reports               <- Generated analysis (e.g. HTML, PDF, LaTeX) 
    - generated         <- Reports that were generated from src contained in this project
    - figures           <- Graphics and figures for reports (of particular note is reports/figures/final_figures for the paper figures)
* requirements.txt
* scripts               <- Standalone scripts (importantly 13_fetch_era.py is used to fetch ERA data)
* src                   <- Source code
    - data              <- Downloading, generating, extracting and aggregating data (final output from this stage should be source agnostic)
    - features          <- Transforming and processing data
    - models            <- Training and testing models
    - visualization     <- Creating exploratory and report visualizations/figures

Hardware Requirements
---------------------
The code can be run on most computers, anything with 8 or 16GB of RAM should suffice. Training only on active fires
takes less than five minutes and training and training on all data (Fire and Non-Fire days) should take a few hours.

Some of the analysis code is written for fast exploration, but not memory effecient. Some parts of the analysis-code
may require >40GB of RAM and running the full figure generation notebook may take >150GB. Running it can be broken into
smaller stages to accomodate machines with less RAM. A future update may add a more optimized version.

Data processing may take more than 8 or 16GB, but only needs to be run once. I don't recall the peak RAM utilization or
full duration, but for reference the processing was run on a machine with 256GB of RAM and 16 cores. However, I don't
believe the peak utilization was anywhere near this ammount of RAM. Some parts of the data pipeline will benefit from
multiple cores if Luigi is configured correctly.

Installation
------------
All software was managed by Conda. The packages for training and running models is fairly standard (numpy, scipy,
statsmodels, etc.). However, for extracting some of the data we needed packages only found on conda-forge.

Running the Analysis
--------------------
To run the analysis, three things must be done:

1. Fetch the MODIS and ERA Interim data
2. Train the appropriate model(s) to be analyzed
3. Run the Jupyter notebook with the anaylsis you are interested in

Data Fetching
-------------
The MODIS data fetching is built into the full pipeline (which will be automatically run if you try to train a model),
but the ERA Interim data must be manually fetched by running scripts/13_fetch_era.py. The ERA data should be written to 
data/interim/gfs/aggregated/era/.

You must setup an account and get an access key to download the ERA data (some addtional information is available in
13_fetch_era.py 

While the MODIS data will not require an account you should set the variable "MODIS_SERVER_PASSWORD" to your email
address (at the request of the maintainers of the data ftp site).

Note that for "historical" reasons, the code sometimes uses gfs to refer to either GFS or ERA Interim data 
(e.g. src/pipeline/gfs_pipeline.py is used for GFS and ERA processing).

GFS data can be used with the model, but the data fetching for the GFS data may take up to a week (but it is automated
via the pipeline).

Model Training
--------------
Models can be trained by running the src/pipeline/train_pipeline.py code. An example can be found in run_pred.py which
includes how to specificy the model/data parameters for running the training.

If all data fetching and processing has been completed, the model will train relatively quickly. A saved model has two
important outputs: its output to stdout and a saved "experiment" pickle file. The information put to stdout (I save 
these by redirecting to a log file) includes the parameters used, evaluation results, and the experiment ID. This ID
will identify the pickle file that includes most of the information from stdout (params and results) and includes the
trained models (so they can be loaded and run) and if the save_residuals flag is set, it will include all model predictions.

Analysis Notebooks
------------------
The most important analysis notebooks are exploratory/spatial_and_temporal_dependence.ipynb and 
figures/final_figures.ipynb. These include all of the code to load the saved experiments, run the analysis, and 
generate figures.

Currently they load saved experiments using their experiment ID (which is a randomly generated number). These will need
to be substituted with the correct ID for any re-run experiments. 

In the future we will update the code to support named experiments so these IDs will not need to be changed.
