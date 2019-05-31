Product Review Sentiment Analysis
================


  - 여러개의 제품 리뷰 중에서
  - 가장 긍정적인 리뷰와 부정적인 리뷰를 추출하고
  - 각각에서 키워드 찾기

# 1\. 프로젝트 개요

  - 제품 리뷰는 크롤링을 이용해서 csv파일로 수집
  - 감성 분석은 감성분석 사전을 이용해서 점수화
  - 점수가 높은 집단과 낮은 집단을 추출
  - 강점과 약점에 대한 키워드 추출은 tfidf를 이용

\-\> group\_by 감성 -\> 긍정/부정적인 리뷰에서 자주 등장한 단어 추출 -\> 강점 / 약점 분석

# 2-1. 감성분석

``` r
# 기준치 설정 
score.neg = -4
score.pos = 5


my.text.location = "C:/Users/user/Desktop/CampusProject/SentimentAnalysis/ReviewData/apple-airpods.csv"

### csv 불러오기
reviews<-read_csv(my.text.location)
```

    ## Parsed with column specification:
    ## cols(
    ##   `0` = col_double(),
    ##   `This is ok, and its sold out as usual, if u see it in store grab it or you be missing out on it. The quality is good for being wireless, Look no further since its a apple product. Buy it as soon u see it in store today. buy it, cause they are rare to find and so popular why not join the crowd.` = col_character()
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

    ## # A tibble: 964 x 3
    ##    review.id word        sentiment
    ##        <dbl> <chr>       <chr>    
    ##  1         1 convenience positive 
    ##  2         1 amazing     positive 
    ##  3         1 well        positive 
    ##  4         1 great       positive 
    ##  5         1 great       positive 
    ##  6         1 super       positive 
    ##  7         1 easy        positive 
    ##  8         1 solid       positive 
    ##  9         2 tired       negative 
    ## 10         2 like        positive 
    ## # ... with 954 more rows

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

    ## [1] 191

``` r
#부정적인 리뷰 
target<-(reviews.result3$score<=score.neg)
target<-replace_na(target, FALSE)
target.neg<-which(target)


#긍정적인 리뷰
target<-(reviews.result3$score>=score.pos)
target<-replace_na(target, FALSE)
target.pos<-which(target)

print(target.pos)
```

    ##  [1]   1   5  27  41  44  46  49  52  54  55  60  61  66  67  68  73  74
    ## [18]  78  79  83  86  87  92 103 110 115 118 120 123 127 130 136 147 150
    ## [35] 165 168 172 175 176 186 187 189 190 191

``` r
View(reviews.result3)
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

    ## <<DocumentTermMatrix (documents: 191, terms: 2089)>>
    ## Non-/sparse entries: 7013/391986
    ## Sparsity           : 98%
    ## Maximal term length: 19
    ## Weighting          : term frequency (tf)

``` r
temp2<-mycorpus

for(i in 1:length(mycorpus)){
  # 1. 공백, 숫자, 기호 전처리
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

# length(mycorpus)
# temp2[[12]]$content
# length(temp2[[1]]$content)
```

## 2-3. tf-idf이용 키워드 추출(시각화)

``` r
# 5. TDM 생성
TermDocumentMatrix(temp2)->tdm.a
tdm.a
```

    ## <<TermDocumentMatrix (terms: 1547, documents: 191)>>
    ## Non-/sparse entries: 5000/290477
    ## Sparsity           : 98%
    ## Maximal term length: 18
    ## Weighting          : term frequency (tf)

``` r
#View(inspect(tdm.a[1:10,1:20]))
# tfidf 생성
weightTfIdf(tdm.a)->tp2

#View(tp2[,n])

print(paste("target.pos :", paste(target.pos, collapse = " "), collapse = " "))
```

    ## [1] "target.pos : 1 5 27 41 44 46 49 52 54 55 60 61 66 67 68 73 74 78 79 83 86 87 92 103 110 115 118 120 123 127 130 136 147 150 165 168 172 175 176 186 187 189 190 191"

``` r
print(paste("target.neg :", paste(target.neg, collapse = " "), collapse = " "))
```

    ## [1] "target.neg : 48 89 106 154"

``` r
# 부정 키워드 상위 n개 출력

n.keyword=5
print("---------------------negative---------------")
```

    ## [1] "---------------------negative---------------"

``` r
for(neg in target.neg){
  neg.top5<-tp2[,neg]$i[order(tp2[,neg]$v, decreasing = T)]
  neg.top5<-head(neg.top5,n.keyword)
  print(tp2[,neg]$dimnames$Terms[neg.top5])
  #print(reviews.result3$content[neg])
}
```

    ## [1] "bend"          "nothing"       "pick"          "maybe"        
    ## [5] "uncomfortable"
    ## [1] "sad"    "single" "thats"  "theyre" "cause" 
    ## [1] "arm"     "cable"   "clipped" "crazy"   "haha"   
    ## [1] "bench"      "both"       "constantly" "contain"    "course"

``` r
# 긍정 키워드 상위 n개 출력
print("------------------positive--------------------")
```

    ## [1] "------------------positive--------------------"

``` r
for(pos in target.pos){
  pos.top5<-tp2[,pos]$i[order(tp2[,pos]$v, decreasing = T)]
  pos.top5<-head(pos.top5,n.keyword)
  print(tp2[,pos]$dimnames$Terms[pos.top5])
  #print(reviews.result3$content[pos])
}
```

    ## [1] "personally" "purchas"    "fan"        "huge"       "inear"     
    ## [1] "pair"       "although"   "compared"   "fine"       "headphones"
    ## [1] "amaz"   "budget" "empty"  "wallet" "cost"  
    ## [1] "nice"    "ehhh"    "market"  "custom"  "thought"
    ## [1] "bose"   "guys"   "nerdy"  "rarely" "unlike"
    ## [1] "comments" "future"   "other"    "provides" "flawless"
    ## [1] "cycle"  "got"    "dryer"  "son"    "washer"
    ## [1] "worst"   "looked"  "pods"    "bump"    "certain"
    ## [1] "control" "didnt"   "volume"  "dropped" "water"  
    ## [1] "awesome" "pretty"  "has"     "husband" "losing" 
    ## [1] "working"    "area"       "compliment" "conditions" "efficient" 
    ## [1] "auto"     "facility" "flexible" "amount"   "covers"  
    ## [1] "somehow"    "little"     "aftersales" "dental"     "floss"     
    ## [1] "autostop"   "cablefree"  "earli"      "everydayif" "replay"    
    ## [1] "access"      "barrierfree" "link"        "links"       "meters"     
    ## [1] "carry"     "listener"  "free"      "always"    "perfectly"
    ## [1] "hold"         "parties"      "wife"         "they"        
    ## [5] "notification"
    ## [1] "everyon" "hiding"  "receive" "freedom" "hair"   
    ## [1] "headset"    "accustomed" "art"        "impeccable" "line"      
    ## [1] "functional" "everyon"    "stylish"    "glad"       "box"       
    ## [1] "call"          "work"          "efficiency"    "ended"        
    ## [5] "entertainment"
    ## [1] "opened" "pops"   "prompt" "due"    "stable"
    ## [1] "video"      "tried"      "earbuds"    "headphones" "break"     
    ## [1] "awai"      "ios"       "macos"     "reviewers" "seamless" 
    ## [1] "quite"    "gestures" "major"    "outstand" "round"   
    ## [1] "bonus"    "misplace" "typing"   "also"     "picks"   
    ## [1] "money"      "everywhere" "internet"   "matter"     "they"      
    ## [1] "accident" "arms"     "making"   "pulling"  "received"
    ## [1] "allowing"     "audiobooks"   "fractionally" "heavier"     
    ## [5] "notch"       
    ## [1] "clarity"   "excellent" "take"      "ever"      "feedback" 
    ## [1] "device"    "books"     "explained" "games"     "manual"   
    ## [1] "price"     "earphones" "con"       "family"    "have"     
    ## [1] "discret"        "fashionable"    "groundbreaking" "amazing"       
    ## [5] "accessory"     
    ## [1] "alreadi"     "synced"      "unboxed"     "sounded"     "unobtrusive"
    ## [1] "ive"       "audio"     "theyre"    "accidents" "blue"     
    ## [1] "pretty"    "entirely"  "generally" "rave"      "good"     
    ## [1] "recipient" "clear"     "discreet"  "exchange"  "stays"    
    ## [1] "fantastic"    "black"        "decisionthey" "obstructive" 
    ## [5] "tablet"      
    ## [1] "nice"        "convienient" "effects"     "self"        "light"      
    ## [1] "favourite"  "investment" "whoosh"     "time"       "freedom"   
    ## [1] "flat"     "hiding"   "honestly" "person"   "hair"    
    ## [1] "iphone"    "accessori" "feels"     "magic"     "process"  
    ## [1] "answering" "enough"    "genius"    "hit"       "jackpot"  
    ## [1] "despite"   "fit"       "differing" "experi"    "well"

## 2-4.

``` r
# (tp1$v)
```

## 3\. 결과(그래프)

\#\#4. 개선사항 \* 키워드 추출에서 의미없는 단어 \* 리뷰에서 주요 문장.

  - source: <https://www.productreview.com.au/>
