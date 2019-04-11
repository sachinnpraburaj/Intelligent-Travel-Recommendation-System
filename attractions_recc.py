import pandas as pd
import numpy as np
import ipywidgets as w
from ipywidgets import HBox, VBox
from ipywidgets import Layout, widgets
from IPython.display import display, IFrame, HTML
from utils import Util
from rbm import RBM
import math, re, datetime as dt, glob
from urllib.parse import quote
from urllib.request import Request, urlopen
from google_images_download import google_images_download
from PIL import Image
from nltk.corpus import wordnet

def f(row):
    avg_cat_rat = dict()
    for i in range(len(row['category'])):
        if row['category'][i] not in avg_cat_rat:
            avg_cat_rat[row['category'][i]] = [row['rating'][i]]
        else:
            avg_cat_rat[row['category'][i]].append(row['rating'][i])
    for key,value in avg_cat_rat.items():
        avg_cat_rat[key] = sum(value)/len(value)
    return avg_cat_rat

def sim_score(row):
    score = 0.0
    match = 0
    col1 = row['cat_rat']
    col2 = row['user_data']
    for key, value in col2.items():
        if key in col1:
            match+=1
            score += (value-col1[key])**2
    if match != 0:
        return ((math.sqrt(score)/match) + (len(col2) - match))
    else:
        return 100

def get_recc(att_df, cat_rating):
    util = Util()
    epochs = 50
    rows = 40000
    alpha = 0.01
    H = 128
    batch_size = 16
    dir= 'etl/'
    ratings, attractions = util.read_data(dir)
    ratings = util.clean_subset(ratings, rows)
    rbm_att, train = util.preprocess(ratings)
    num_vis =  len(ratings)
    rbm = RBM(alpha, H, num_vis)
    
    joined = ratings.set_index('attraction_id').join(attractions[["attraction_id", "category"]].set_index("attraction_id")).reset_index('attraction_id')
    grouped = joined.groupby('user_id')
    category_df = grouped['category'].apply(list).reset_index()
    rating_df = grouped['rating'].apply(list).reset_index()
    cat_rat_df = category_df.set_index('user_id').join(rating_df.set_index('user_id'))
    cat_rat_df['cat_rat'] = cat_rat_df.apply(f,axis=1)
    cat_rat_df = cat_rat_df.reset_index()[['user_id','cat_rat']]
    
    cat_rat_df['user_data'] = [cat_rating for i in range(len(cat_rat_df))]
    cat_rat_df['sim_score'] = cat_rat_df.apply(sim_score, axis=1)
    user = cat_rat_df.sort_values(['sim_score']).values[0][0]
    
    print("Similar User: {u}".format(u=user))
    filename = "e"+str(epochs)+"_r"+str(rows)+"_lr"+str(alpha)+"_hu"+str(H)+"_bs"+str(batch_size)
    reco, weights, vb, hb = rbm.load_predict(filename,train,user)
    unseen, seen = rbm.calculate_scores(ratings, attractions, reco, user)
    rbm.export(unseen, seen, 'rbm_models/'+filename, str(user))
    return filename, user, rbm_att

def filter_df(filename, user, low, high, province, att_df):
    recc_df = pd.read_csv('rbm_models/'+filename+'/user{u}_unseen.csv'.format(u=user), index_col=0)
    recc_df.columns = ['attraction_id', 'att_name', 'att_cat', 'att_price', 'score']
    recommendation = att_df[['attraction_id','name','category','city','latitude','longitude','price','province', 'rating']].set_index('attraction_id').join(recc_df[['attraction_id','score']].set_index('attraction_id'), how="inner").reset_index().sort_values("score",ascending=False)
    
    filtered = recommendation[(recommendation.province == province) & (recommendation.price >= low) & (recommendation.price >= low)]
    url = pd.read_json('outputs/attractions_cat.json',orient='records')
    url['id'] = url.index
    with_url = filtered.set_index('attraction_id').join(url[['id','attraction']].set_index('id'), how="inner")
    return with_url

def get_image(name):
    name = name.split(",")[0]
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
        
        for filename in glob.glob("downloads/{name}/*jpg".format(name=name)) + glob.glob("downloads/{name}/*png".format(name=name)):
            return filename
    except:
        for filename in glob.glob("downloads/*jpg"):
            return filename

def top_recc(with_url, final):
    i=0
    while(1):
        first_recc = with_url.iloc[[i]]
        if(first_recc['name'].values.T[0] not in final['name']):
            final['name'].append(first_recc['name'].values.T[0])
            final['location'].append(first_recc[['latitude','longitude']].values.tolist()[0])
            final['price'].append(first_recc['price'].values.T[0])
            final['rating'].append(first_recc['rating'].values.T[0])
            final['image'].append(get_image(first_recc['name'].values.T[0]))
            final['category'].append(first_recc['category'].values.T[0])
            return final
        else:
            i+=1

def find_closest(with_url, loc, tod, final):
    syns1 = wordnet.synsets("evening")
    syns2 = wordnet.synsets("night")
    evening = [word.lemmas()[0].name() for word in syns1] + [word.lemmas()[0].name() for word in syns2]
    distance = list()
    for i in with_url[['latitude','longitude']].values.tolist():
        distance.append(math.sqrt((loc[0]-i[0])**2 + (loc[1]-i[1])**2))
    with_dist = with_url
    with_dist["distance"] = distance
    sorted_d = with_dist.sort_values(['distance','price'], ascending=['True','False'])
    if tod == "Evening":
        mask = sorted_d.name.apply(lambda x: any(j in x for j in evening))
    else:
        mask = sorted_d.name.apply(lambda x: all(j not in x for j in evening))
    final = top_recc(sorted_d[mask], final)
    return final

def final_output(days, final):
    time = ['MORNING', 'EVENING']
    fields = ['NAME', 'CATEGORY', 'LOCATION', 'PRICE', 'RATING']
    recommendations = ['Recommendation 1:', 'Recommendation 2:']

    box_layout = Layout(justify_content='space-between',
                        display='flex',
                        flex_flow='row', 
                        align_items='stretch',
                       )
    column_layout = Layout(justify_content='space-between',
                        width='75%',
                        display='flex',
                        flex_flow='column', 
                       )
    tab = []
    for i in range(days):
        images = final['image'][i*4:(i+1)*4]
        images = [open(i, "rb").read() for i in images]
        name = [re.sub('_',' ',i).capitalize() for i in final['name'][i*4:(i+1)*4]]
        category = [re.sub('_',' ',i).capitalize() for i in final['category'][i*4:(i+1)*4]]
        location = ["("+str(i[0])+","+str(i[1])+")" for i in final['location'][i*4:(i+1)*4]]
        price = [str(i) for i in final['price'][i*4:(i+1)*4]]
        rating = [str(i) for i in final['rating'][i*4:(i+1)*4]]
        tab.append(VBox(children=
                        [HBox(children=
                              [VBox(children=
                                    [widgets.HTML(value = f"<b><font color='orange'>{time[0]}</b>"),
                                     widgets.HTML(value = f"<b><font color='purple'>{recommendations[0]}</b>"),
                                     widgets.Image(value=images[0], format='jpg', width=300, height=400), 
                                     widgets.HTML(description=fields[0], value=f"<b><font color='black'>{name[0]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[1], value=f"<b><font color='black'>{category[0]}</b>", disabled=True),
                                     widgets.HTML(description=fields[2], value=f"<b><font color='black'>{location[0]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[3], value=f"<b><font color='black'>{price[0]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[4], value=f"<b><font color='black'>{rating[0]}</b>", disabled=True)
                                    ], layout=column_layout), 
                                VBox(children=
                                    [widgets.HTML(value = f"<b><font color='orange'>{time[1]}</b>"), 
                                     widgets.HTML(value = f"<b><font color='purple'>{recommendations[0]}</b>"),
                                     widgets.Image(value=images[2], format='jpg', width=300, height=400), 
                                     widgets.HTML(description=fields[0], value=f"<b><font color='black'>{name[2]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[1], value=f"<b><font color='black'>{category[2]}</b>", disabled=True),
                                     widgets.HTML(description=fields[2], value=f"<b><font color='black'>{location[2]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[3], value=f"<b><font color='black'>{price[2]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[4], value=f"<b><font color='black'>{rating[2]}</b>", disabled=True)
                                    ], layout=column_layout)
                              ], layout=box_layout),

                         HBox(children=
                              [VBox(children=
                                    [widgets.HTML(value = f"<b><font color='purple'>{recommendations[1]}</b>"),
                                     widgets.Image(value=images[1], format='jpg', width=300, height=400), 
                                     widgets.HTML(description=fields[0], value=f"<b><font color='black'>{name[1]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[1], value=f"<b><font color='black'>{category[1]}</b>", disabled=True),
                                     widgets.HTML(description=fields[2], value=f"<b><font color='black'>{location[1]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[3], value=f"<b><font color='black'>{price[1]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[4], value=f"<b><font color='black'>{rating[1]}</b>", disabled=True)
                                    ], layout=column_layout), 
                                VBox(children=
                                    [widgets.HTML(value = f"<b><font color='purple'>{recommendations[1]}</b>"),
                                     widgets.Image(value=images[3], format='jpg', width=300, height=400), 
                                     widgets.HTML(description=fields[0], value=f"<b><font color='black'>{name[3]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[1], value=f"<b><font color='black'>{category[3]}</b>", disabled=True),
                                     widgets.HTML(description=fields[2], value=f"<b><font color='black'>{location[3]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[3], value=f"<b><font color='black'>{price[3]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[4], value=f"<b><font color='black'>{rating[3]}</b>", disabled=True)
                                    ], layout=column_layout),
                              ], layout=box_layout)

                        ]))

    tab_recc = widgets.Tab(children=tab)
    for i in range(len(tab_recc.children)):
        tab_recc.set_title(i, str('Day '+ str(i+1)))
    return tab_recc
