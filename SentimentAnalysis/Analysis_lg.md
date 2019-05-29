Product Review Sentiment Analysis
================

``` r
library(KoNLP)
```

    ## Checking user defined dictionary!

``` r
library(rJava)
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(readr)
library(ggplot2)
```

    ## Registered S3 methods overwritten by 'ggplot2':
    ##   method         from 
    ##   [.quosures     rlang
    ##   c.quosures     rlang
    ##   print.quosures rlang

``` r
library(tidyr)
library(tidytext)
library(stringr)
library(RWeka)
library(tm)
```

    ## Loading required package: NLP

    ## 
    ## Attaching package: 'NLP'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     annotate

``` r
library(SnowballC)
```

## 목차

0.  프로젝트 목적
1.  프로젝트 개요
2.  내용
3.  결과
4.  분석
5.  개선사항

# 0\. 프로젝트 목적

  - 수백개의 제품 리뷰 중에서
  - 긍정적인 리뷰와 부정적인 리뷰를 구분하고
  - 각각에서 제품의 강점과 약점을 찾아낸다.

# 1\. 프로젝트 개요

  - 제품 리뷰는 크롤링을 이용해서 csv파일로 수집
  - 감성 분석은 감성분석 사전을 이용해서 점수화
  - 점수가 높은 집단과 낮은 집단을 추출
  - 강점과 약점에 대한 키워드 추출은 tfidf를 이용

\-\> group\_by 감성 -\> 긍정/부정적인 리뷰에서 자주 등장한 단어 추출 -\> 강점 / 약점
분석

# 2-1. 감성분석

``` r
my.text.location = "C:/Users/user/Desktop/CampusProject/SentimentAnalysis/ReviewData/hyundai-i30.csv"

### csv 불러오기
reviews<-read_csv(my.text.location)
```

    ## Parsed with column specification:
    ## cols(
    ##   `0` = col_double(),
    ##   `Bought new from Hyundai dealer, had trouble with the transmission from day one, zero km on the clock.  Took it back 7 times in the five months I owned it.` = col_character()
    ## )

``` r
### 열 이름 주기
colnames(reviews)<-c("review.id", "content")

### 단어 토큰으로 쪼개기 
reviews %>% 
  unnest_tokens(word, content) ->
  reviews.word

### 5. 감성사전으로 점수 매기기
inner_join(reviews.word, get_sentiments("bing"), by="word")
```

    ## # A tibble: 1,457 x 3
    ##    review.id word         sentiment
    ##        <dbl> <chr>        <chr>    
    ##  1         2 wrong        negative 
    ##  2         3 loss         negative 
    ##  3         5 recommend    positive 
    ##  4         5 successfully positive 
    ##  5         6 troublesome  negative 
    ##  6         6 loyalty      positive 
    ##  7         9 love         positive 
    ##  8         9 great        positive 
    ##  9         9 work         positive 
    ## 10        11 amazing      positive 
    ## # ... with 1,447 more rows

``` r
# 6.score column 추가 
reviews.word %>% 
  inner_join(get_sentiments("bing"), by="word") %>% 
  count(word, review.id, sentiment) %>% 
  spread(sentiment, n, fill=0) %>% 
  arrange(review.id) ->
  reviews.result


# 7.계산
reviews.result %>% 
  group_by(review.id) %>% 
  summarise(pos.sum = sum(positive),
            neg.sum = sum(negative),
            score = pos.sum-neg.sum) ->
  reviews.result2


# reviews<-reviews.origin
# View(reviews)
# print(reviews)
# class(reviews)
# View(reviews.result2)
# print(reviews.result)
# class(reviews.result)

#View(left_join(reviews, reviews.result2, by="review.id", all=T)
reviews.result3<-left_join(reviews, reviews.result2, by="review.id", all=T)
length(reviews.result3$score)
```

    ## [1] 250

``` r
target<-(reviews.result3$score<=-5)
target<-replace_na(target, FALSE)
which(target)
```

    ## [1]  14  24  71 137 144

``` r
View(reviews.result3$content[which(target)])
```

## 2-2. 키워드 추출을 위한 전처리

1.  공백처리
2.  특수문자 처리
3.  숫자 처리
4.  불용어 처리
5.  n-gram

<!-- end list -->

``` r
mycorpus<-VCorpus(VectorSource(reviews$content))
DocumentTermMatrix(mycorpus)
```

    ## <<DocumentTermMatrix (documents: 250, terms: 3634)>>
    ## Non-/sparse entries: 11108/897392
    ## Sparsity           : 99%
    ## Maximal term length: 24
    ## Weighting          : term frequency (tf)

``` r
temp2<-mycorpus

for(i in 1:length(mycorpus)){
  
  # 1. 공백처리
  temp2[[i]]$content<-str_replace_all(temp2[[i]]$content,"[[:space:]]{1,}", " ")
  temp2[[i]]$content<-str_replace_all(temp2[[i]]$content,"[[:digit:]]{1,}","")
  temp2[[i]]$content<-str_replace_all(temp2[[i]]$content,"[[:punct:]]{1,}","")
  
}

# 2. 불용어 사전 이용해서 불용어 제거
tm_map(temp2, FUN = removeWords, words = stopwords("en"))->temp2

# 3. wordstem으로 어근 변환 # library(SnowballC)
for(i in 1:length(mycorpus)){
  temp2[[i]]$content<-wordStem(temp2[[i]]$content)
}

# 4. 단어 추출 
for(i in 1:length(mycorpus)){
  temp2[[i]]$content <- paste(unlist(str_extract_all(temp2[[i]]$content, boundary("word"))), collapse = " ")
}


length(mycorpus)
```

    ## [1] 250

``` r
temp2[[12]]$content
```

    ## [1] "Only fall long legs back seat definitely squeez"

``` r
length(temp2[[1]]$content)
```

    ## [1] 1

## 2-3. tf-idf이용 키워드 추출(시각화)

``` r
# 5. TDM
TermDocumentMatrix(temp2)->tdm.a
tdm.a
```

    ## <<TermDocumentMatrix (terms: 2586, documents: 250)>>
    ## Non-/sparse entries: 8195/638305
    ## Sparsity           : 99%
    ## Maximal term length: 22
    ## Weighting          : term frequency (tf)

``` r
#View(inspect(tdm.a[1:10,1:20]))

# tfidf
weightTfIdf(tdm.a)->tp1
tp1*tdm.a->tp2

tp2<-tp1


#View(tdm.a)
#View(tp1)
#View(tp2)


n=71
#View(tp2[,n])

# 단어 중요도 순서대로 출력
p<-tp2[,71]$dimnames$Terms[tp2[,71]$i[order(tp2[,71]$v, decreasing = T)]]
print(p)
```

    ##  [1] "clutch"     "needed"     "told"       "around"     "aware"     
    ##  [6] "steering"   "replaced"   "fault"      "again"      "clunk"     
    ## [11] "disclosure" "fixes"      "gotten"     "habit"      "ignore"    
    ## [16] "impact"     "knock"      "properly"   "rode"       "taught"    
    ## [21] "fixed"      "since"      "appear"     "needing"    "twice"     
    ## [26] "cars"       "couple"     "mechanics"  "often"      "column"    
    ## [31] "wear"       "bad"        "fully"      "maybe"      "mechanic"  
    ## [36] "started"    "corners"    "fix"        "mark"       "put"       
    ## [41] "window"     "know"       "quickly"    "along"      "drivers"   
    ## [46] "make"       "sure"       "yet"        "manual"     "wheel"     
    ## [51] "havent"     "full"       "never"      "way"        "issue"     
    ## [56] "better"     "the"        "issues"     "year"       "can"       
    ## [61] "new"        "drive"      "car"

``` r
extractNoun(paste(p,collapse = " "))
```

    ##  [1] "clutch"     "needed"     "told"       "around"     "aware"     
    ##  [6] "steering"   "replaced"   "fault"      "again"      "clunk"     
    ## [11] "disclosure" "fixes"      "gotten"     "habit"      "ignore"    
    ## [16] "impact"     "knock"      "properly"   "rode"       "taught"    
    ## [21] "fixed"      "since"      "appear"     "needing"    "twice"     
    ## [26] "cars"       "couple"     "mechanics"  "often"      "column"    
    ## [31] "wear"       "bad"        "fully"      "maybe"      "mechanic"  
    ## [36] "started"    "corners"    "fix"        "mark"       "put"       
    ## [41] "window"     "know"       "quickly"    "along"      "drivers"   
    ## [46] "make"       "sure"       "yet"        "manual"     "wheel"     
    ## [51] "havent"     "full"       "never"      "way"        "issue"     
    ## [56] "better"     "the"        "issues"     "year"       "can"       
    ## [61] "new"        "drive"      "ca"         "r"

``` r
class(tp2[,71]$v)
```

    ## [1] "numeric"

``` r
print(tp2[,n]$dimnames$Terms[tp2[,n]$i[tp2[,n]$v>=0.1]])
```

    ## [1] "around"   "aware"    "clutch"   "fault"    "needed"   "replaced"
    ## [7] "steering" "told"

``` r
reviews.result3$content[[n]]
```

    ## [1] "A couple of fixes needed along the way:- It needed a new clutch at around the 6 year mark (maybe around 100,000). The mechanic told me it was a know fault for the clutch to wear quickly in these cars, but I am fully aware that I was never taught to drive a manual 'properly' and often rode the clutch around corners. A bad habit I have since fixed, so not sure if it can be put down to the car itself. No issues since.- The mechanics in the drivers window needed to be replaced twice. Again, I was told this is a fault in the cars.- At around 180,000 the steering wheel started to 'knock' and clunk. I have been told this is the steering column needing to be replaced and (again) this is an issue in the make of the car. Full disclosure, I haven't gotten it fixed yet, it does not appear to impact the steering, but I am aware that it is better to fix it than ignore it."

``` r
tp2[,24]$v[which(tp2[,24]$i == 1576)]
```

    ## named numeric(0)

``` r
temp2[[24]]$content
```

    ## [1] "My Hyundai I done km service I noticed paint defect see photo column behind drivers seat The Hyundai dealer sent photos Hyundai I communicated Customer Care well Two panel beaters said defect due exteranl influences rubbing I driven years never issue The thing I can guess possible cause seat belt flicking back hitting column released car newHyundai declined assist way The repair will cost Their attitude disappointing I feel I caused defect recklessness lack car"

## 2-4.

``` r
# (tp1$v)
```

## 3\. 결과(그래프)

\#\#4. 개선사항 \* 키워드 추출에서 의미없는 단어 \* 리뷰에서 주요 문장.

  - source: <https://www.productreview.com.au/>
