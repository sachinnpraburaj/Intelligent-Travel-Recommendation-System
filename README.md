# Intelligent Travel Recommendation System <br />
<br />
Video demo: https://youtu.be/V635gdcw1h0 <br />
Project Report: report.pdf <br />
Poster: poster.pdf

# Objective:
To provide a tailor made plan consisting of possible places to stay, attractions to visit and restaurants to eat at for the entire duration of travel. We recommend restaurants separately for each meal of the day (breakfast, lunch and dinner) and provide two recommendations per meal per day. We also recommend five possible stay options (hotels) for your travel alongside possible attractions to view. Attractions are recommended based on timing, (i.e) which ones to view during the day and which ones are better off at night. Again we provide two attraction recommendations per timing per day for the entire duration of travel.

# Project Overview:
We have used three different recommender systems (one each for attractions, hotels and restaurants).
1. RBM, a Deep learning technique for Attractions.
2. Matrix Factorization with ALS, a highly scalable and distributed Collaborative Filtering technique for hotels.
3. Hybrid- A combination of K-Means algorithm for Content Based Filtering and K-Nearest Neighbors for Memory based Collaborative Filtering for restaurants.

Few visualizations for the project were done using python libraries and others have been done using Tableau software. They can be accessed from [EDA](/EDA) folder.

# Steps for execution:
For restaurants- Dataset for the project should be downloaded from [Yelp dataset challenge](https://www.yelp.ca/dataset/download) and stored in yelp_dataset folder.

For hotels- We scraped TripAdvisor to obtain the dataset. Dataset can be read from [tripadvisor_hotel_output](/tripadvisor_hotel_output) folder.

For attractions- We scraped TripAdvisor to obtain the dataset. Dataset can be read from [outputs](/outputs) folder.

# Files: <br />
###### [attractions_crawler.ipynb](/attractions_crawler.ipynb)
  --  To collect urls of attractions from tripadvisor.

###### [attractions_details_crawler.ipynb](/attractions_details_crawler.ipynb)
  --  To extract attraction details and reviews on each attraction from collected urls in batches

###### [combine_batches.ipynb](/combine_batches.ipynb)
  --  To combine the data collected in batches

###### [attraction_etl.ipynb](/attraction_etl.ipynb)
  --  To perform ETL on attraction details and attractions reviews datasets.

###### [attractions_recc.py](/attractions_recc.py)
  -- The core code to provide attraction recommendation using the trained RBM model.

###### [final_hotel_recc.ipynb](/final_hotel_recc.ipynb)
  -- The final code that integrates ETL on hotels dataset and MF-ALS model output to display hotel recommendations.

###### [get_att_recc.ipynb](/get_att_recc.ipynb)
  -- The final code that integrates ETL on attractions dataset and RBM model output to display attraction recommendations.

###### [hotel_etl.ipynb](/hotel_etl.ipynb)
  -- To perform 'Extract Transform Load (ETL)' on hotels dataset that has been scraped from TripAdvisor.

###### [hotel_recc.py](/hotel_recc.py)
  -- The core code that models MF-ALS and outputs recommendations.

###### [rbm_training.ipynb](/rbm_training.ipynb)
  -- The code to perform training and tuning of the RBM, deep learning model.

###### [rbm.py](/rbm.py)
  -- The code that loads the best model and outputs recommendations for users.

###### [requirements.txt](/requirements.txt)
  -- File to handle dependencies for thus project.

###### [Restaurants (Yelp) Dataset-EDA.ipynb]
  -- The notebook that has the code and shows EDA visualizations for Yelp (restaurants) dataset.

###### [TripAdvisor_Crawler_Parser.ipynb](/TripAdvisor_Crawler_Parser.ipynb)
  -- The notebook performs collection, extraction, cleaning, parsing and obtaining hotel urls, hotel related information, user reviews, user ratings and user related information.

###### [utils.py](/utils.py)
  -- Consists of helper functions for the RBM model.

###### [Hybrid_Recommder.ipynb](/Hybrid_Recommder.ipynb)
  -- The core code for ETL on yelp dataset and hybrid recommender model.


# Folders:

###### [etl](/etl)
  -- Saved model parameters and model outputs from MF-ALS.

###### [input-output](/input-output)
  -- contains screenshots of input and output images of ITRS application on the whole.

###### [outputs](/outputs)
  -- Contains dataset for attractions.

###### [tripadvisor_hotel_output](/tripadvisor_hotel_output)
  -- Contains dataset for hotels.

###### [downloads](/downloads)
  -- Contains attraction images downloaded using google_download_images API.

###### [mf_models](/mf_models)
  -- Contains the saved best obtained Matrix Factorization- ALS (MF-ALS) model.

###### [rbm_models](/rbm_models)
  -- Contains saved RBM models, tried out for different parameters.

###### [EDA](/analysis)
  -- all results (visualizations) of Exploratory Data Analysis (EDA) are be stored here.
