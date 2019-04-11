import pandas as pd
import ipywidgets as w
from IPython.display import display, IFrame
import math, re, numpy as np, pyspark, glob
from urllib.parse import quote
from urllib.request import Request, urlopen
from google_images_download import google_images_download
from pyspark.sql import SQLContext, functions, types
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.sql import Row
from geopy.geocoders import Nominatim

def get_rating(x):
    val = x / 5
    if x >= 0 and x <= val:
        return 1
    elif x > val and x <= 2*val:
        return 2
    elif x > 2*val and x <= 3*val:
        return 3
    elif x > 3*val and x <= 4*val:
        return 4
    else:
        return 5
    
## Finding number of amenities present in hotels that user likes
def amenities_rating(spark, amenities_pref, newh_df):
    pa_df = pd.DataFrame(amenities_pref,columns=["amenities_pref"])

    a_df = spark.createDataFrame(pa_df)
    a_df.createOrReplaceTempView('a_df')
    
    newh_df.createOrReplaceTempView('del_dup')
    newa_df  = spark.sql("SELECT * FROM newh_df INNER JOIN a_df WHERE newh_df.amenities=a_df.amenities_pref")

    ameni_comb = newa_df.groupBy(functions.col("id")).agg(functions.collect_list( functions.col("amenities")).alias("amenities"))
    
    amenities_len=ameni_comb.withColumn("ameni_len",functions.size(ameni_comb["amenities"])).orderBy(functions.col("ameni_len"), ascending=False)
    amenities_len.createOrReplaceTempView("amenities_len")

    ameni_df = spark.sql("SELECT a.id,h.amenities,a.ameni_len FROM del_dup h INNER JOIN amenities_len a WHERE h.id=a.id ORDER BY a.ameni_len DESC")
    
    find_rating = functions.udf(lambda a: get_rating(a), types.IntegerType())
    usr_rating = ameni_df.withColumn("rating",find_rating("ameni_len"))
    return usr_rating

def model_train(spark, usr_rating):
    ## Adding new user info to original dataset for training
    u_id_df = spark.read.json('project/u_id_df')
    u_id_df.createOrReplaceTempView('u_id_df')
    
    uid_count = u_id_df.select("user_id").distinct().count()

    usrid_df = usr_rating.withColumn("usr_id", functions.lit(uid_count)).join(u_id_df.select(["id","att_id"]), "id")

    usrid_final_df = usrid_df.select(usrid_df["usr_id"].alias("user_id"),usrid_df["att_id"].alias("att_id"),usrid_df["rating"].alias("user_rating"))

    org_df = u_id_df.select("user_id","att_id","user_rating")

    (usrid_s1, usrid_s2) = usrid_final_df.randomSplit([0.1,0.9])

    comb_df = org_df.union(usrid_s1)
    
    ## Model training and evaluation
    (training,validation) = comb_df.randomSplit([0.8,0.2])

    ranks=[4,8,12]
    error = 20000
    errors = []
    for i in ranks:
        als = ALS(maxIter= 5,regParam= 0.01,rank=i,userCol="user_id",itemCol="att_id", ratingCol="user_rating", coldStartStrategy="drop")
        model = als.fit(training)
        predictions = model.transform(validation)
        evaluator = RegressionEvaluator(metricName="rmse",labelCol="user_rating",predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    errors.append(rmse)
    if rmse < error:
        model.write().overwrite().save("mf_models/model_file")
        rank = i
        error = rmse
    
    return rank, error, errors, usrid_s2

def get_hotel_recc(spark, usrid_s2):
    als_model = ALSModel.load("mf_models/model_file")

    user = usrid_s2.select("user_id").distinct()
    recomm = als_model.recommendForUserSubset(user,50)
    recomm.createOrReplaceTempView('recomm')

    recomm_df  = spark.sql("SELECT user_id,explode(recommendations) as recommendations FROM recomm")

    get_attid = recomm_df.withColumn("att_id",functions.col("recommendations.att_id")).withColumn("rating",functions.col("recommendations.rating"))
    get_attid.createOrReplaceTempView("get_attid")
    
    u_id_df = spark.read.json('project/u_id_df')
    u_id_df.createOrReplaceTempView('u_id_df')
    u_tempdf = spark.sql("SELECT u_id_df.id FROM u_id_df INNER JOIN get_attid on u_id_df.att_id=get_attid.att_id")
    
    return u_tempdf

def get_image(name):
    name = re.sub(' ','_',name)
    response = google_images_download.googleimagesdownload()
    args_list = ["keywords", "keywords_from_file", "prefix_keywords", "suffix_keywords",
             "limit", "format", "color", "color_type", "usage_rights", "size",
             "exact_size", "aspect_ratio", "type", "time", "time_range", "delay", "url", "single_image",
             "output_directory", "image_directory", "no_directory", "proxy", "similar_images", "specific_site",
             "print_urls", "print_size", "print_paths", "metadata", "extract_metadata", "socket_timeout",
             "thumbnail", "language", "prefix", "chromedriver", "related_images", "safe_search", "no_numbering",
             "offset", "no_download"]
    args = {}
    for i in args_list:
        args[i]= None
    args["keywords"] = name
    args['limit'] = 1
    params = response.build_url_parameters(args)
    url = 'https://www.google.com/search?q=' + quote(name) + '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch' + params + '&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'
    try:
        response.download(args)
        for filename in glob.glob("downloads/{name}/*jpg".format(name=name))+glob.glob("downloads/{name}/*png".format(name=name)):
            return filename
    except:
        for filename in glob.glob("downloads/*jpg"):
            return filename
        
def get_hotel_output(days, final):
    fields = ['NAME', 'PRICE', 'RATING', 'EXPERIENCE','LOCATION', 'ADDRESS', "AMENITIES"]
    recommendations = ['Recommendation']

    box_layout = w.Layout(justify_content='space-between',
                        display='flex',
                        flex_flow='row', 
                        align_items='stretch',
                       )
    column_layout = w.Layout(justify_content='space-between',
                        width='75%',
                        display='flex',
                        flex_flow='column', 
                       )
    tab = []
    for i in range(len(final['name'])):
        image = open(final['image'][i], "rb").read()
        name = final['name'][i]
        price= final['price'][i]
        rating= final['rating'][i]
        experience= final['experience'][i]
        loc=final['location'][i]
        address=final['address'][i]
        amenities=final['amenities'][i]
        tab.append(w.VBox(children=
                        [
                         w.Image(value=image, format='jpg', width=300, height=400),
                         w.HTML(description=fields[0], value=f"<b><font color='black'>{name}</b>", disabled=True),
                         w.HTML(description=fields[1], value=f"<b><font color='black'>{price}</b>", disabled=True),
                         w.HTML(description=fields[2], value=f"<b><font color='black'>{rating}</b>", disabled=True), 
                         w.HTML(description=fields[3], value=f"<b><font color='black'>{experience}</b>", disabled=True), 
                         w.HTML(description=fields[4], value=f"<b><font color='black'>{loc}</b>", disabled=True),
                         w.HTML(description=fields[5], value=f"<b><font color='black'>{address}</b>", disabled=True)
                        ], layout=column_layout))

    tab_recc = w.Tab(children=tab)
    for i in range(len(tab_recc.children)):
        tab_recc.set_title(i, str('Hotel '+ str(i+1)))
    return tab_recc
