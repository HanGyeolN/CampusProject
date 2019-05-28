# -*- coding:utf-8 -*-
# 제품 리뷰 스크래핑
# "https://www.productreview.com.au/listings/lg-gb-450?page=3#reviews"

import requests as rq
import json
import pandas as pd
from pandas import DataFrame as df
from bs4 import BeautifulSoup as bs

reviews = list()
productName = "hyundai-i30"
endPage = 7

########################################
# 반복

for pageNum in range(1,endPage+1):
    #######################################
    # block 가져오기
    url = "https://www.productreview.com.au/listings/"+productName+"?page="+str(pageNum)+"#reviews"
    res = rq.get(url)
    soup = bs(res.content, 'lxml')

    selector = "#content > div.relative_2e- > div > div > div.flex-column_36B.d-flex_b9D > div.flex-grow-1_3A8.relative_2e- > div.mt-3_32-.relative_2e-.container_Fok > div.row_TNM.gutterX-margin-3_20n > div.col-lg-8_2Dc.gutterX-padding-3_1qr.main_2l0 > div > div:nth-child(1) > div.mb-3_2I3.card_134.card-full_3wf.card-full-md_2gh > div > div > div"
    soup2= soup.select(selector) 
    soup2 = soup2[0]

    ########################################
    # string 뽑기 
    soup3 = soup2.find_all("p")
    #print(len(soup3))    

    for review in soup3:
        reviews.append(review.text.strip())
    
    print(str(pageNum),"page complete")

########################################
# 저장 
print("Number of Reviews :", len(reviews))

path = "C:/Users/user/desktop/CampusProject/Sentiment Analysis/ReviewData/"+productName+".txt"
fw = open(path, 'w', -1, "utf-8") # UnicodeEncodeError: 'cp949' codec can't encode character '\u2764' in position 65597
fw.write(str(reviews))
fw.close()

print(productName,"Complete")

