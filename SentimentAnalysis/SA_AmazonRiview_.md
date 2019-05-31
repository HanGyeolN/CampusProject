Amazon Reviews
================
## 리뷰 크롤링

``` r
productName="Winix-Purifier"
# 위닉스 공기청정기:https://www.amazon.com/Winix-5500-2-Purifier-PlasmaWave-Reducing/product-reviews/B01D8DAYII/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=
# 다이슨 코드제로:https://www.amazon.com/Dyson-Motorhead-Cordless-Cleaner-227591-01/product-reviews/B01MSZ036Q/ref=cm_cr_arp_d_paging_btm_next_2?&reviewerType=all_reviews&pageNumber=
# 삼성 m.2 ssd:"https://www.amazon.com/Samsung-970-PRO-512GB-MZ-V7P512BW/product-reviews/B07C8Y31G2/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber="

url.base<-"https://www.amazon.com/Samsung-970-PRO-512GB-MZ-V7P512BW/product-reviews/B07C8Y31G2/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber="
page = 1

url<-paste0(url.base, page, collapse="")

#################################
# 차단 우회용 헤더 추가
header = httr::user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36")

request_result <- httr::GET(url,header)
```

``` r
#########################################
# 원래 꺼 -> 막힘
# htxt<-read_html(url,headers = header)


#########################################
# 수정 후 -> user-agent 헤더 추가
htxt<-read_html(request_result$content)

reviews.count <- htxt %>%
  html_nodes("[data-hook=total-review-count]") %>%
  html_text

# 숫자 1000자리 구분 컴마 제거
reviews.count<-str_replace_all(reviews.count,"[[:punct:]]","")
page.end = ceiling(as.integer(reviews.count)/10)

# 저장할 변수 초기화
reviews.all <- c()
stars.all <- c()
dates.all <- c()

# for (page in 1:page.end){
#   url<-paste0(url.base,page,collapse="")
#   request_result <- httr::GET(url,header)
#   htxt<-read_html(request_result$content)
#   
#   #################################
#   # 리뷰 크롤링
#   reviews.temp <- htxt %>%
#   html_nodes("[data-hook=review-body]") %>%
#   html_text %>% 
#   str_replace_all("[[:space:]]{1,}", " ")
#   
#   #################################
#   # 평점 크롤링
#   stars.temp <- htxt %>% 
#     html_nodes("[id=cm_cr-review_list]") %>% 
#     html_nodes("[class=a-icon-alt]") %>% 
#     html_text %>% 
#     str_extract_all("\\b[[:digit:]]\\.[[:digit:]]") %>% 
#     unlist()
# 
#   #################################
#   # 날짜 크롤링
#   
#   dates.temp <- htxt %>% 
#     html_nodes("[id=cm_cr-review_list]") %>% 
#     html_nodes("[data-hook=review-date]") %>% 
#     html_text
#     
#   dates.all<-c(dates.all, dates.temp)
#   stars.all<-c(stars.all, stars.temp)
#   reviews.all<-c(reviews.all, reviews.temp)
#   
#   if(is.na(reviews.temp[1])){
#     print("Blocked")
#     break
#   }
#   
#   print(paste("page", page, "complete"))
# }
# 
# print("길이체크")
# print(length(reviews.all)==length(stars.all)&length(stars.all)==length(dates.all))
# 
# Data<-tibble("id" = c(1:length(reviews.all)), "content"=reviews.all, "date"=dates.all, "star"=stars.all)
# 
fileName = paste0(productName,".csv", collapse="")
# 
# write.table(Data,fileName)
# 
# print(paste("전체 페이지 갯수:", page.end))
# print(paste("전체 리뷰 갯수:", length(reviews.all)))

Data<-read.table(fileName, stringsAsFactors = F)
```

## 리뷰 감성분석

``` r
reviews <- Data
result <- Data
# View(reviews.all)
reviews.word <- Data %>% 
  unnest_tokens(word, content)

# positive 갯수 추출을 위한 객체 
pos <- reviews.word %>% 
  inner_join(get_sentiments("bing"), by="word") %>% 
  count(id,sentiment) %>% 
  filter(sentiment=="positive") %>% 
  select(id,n)

# 감성 결과 분석
result.word <- reviews.word %>% 
  inner_join(get_sentiments("bing"), by="word") %>% 
  mutate(wordScore=ifelse(sentiment=="positive", 1, -1)) %>% 
  group_by(id) %>% 
  summarise(score=sum(wordScore),total=sum(abs(wordScore)))

result <- left_join(result, result.word, all.x=T, by="id") #감성 score 합치기
result <- left_join(result, pos,by="id",all.x=T) #positive 갯수 합치기
result <- rename(result,"positive"=n) #이름 수정 

# 변환후 전체 갯수 확인
# table(result$positive, useNA = "always")
#result[673,]

result$positive[which(is.na(result$positive))]<-0
result$negative <- result$total - result$positive

# 긍정률 부정률 
result$pos.rate <- result$positive/result$total
result$pos.rate[which(is.na(result$pos.rate))] <- 0
result$neg.rate <- result$negative/result$total
result$neg.rate[which(is.na(result$neg.rate))] <- 0
#View(result)

# x테스트
# result
# View(result)
# length(result$total)
# length(result$positive)
# length(reviews.all)
# length(result$id)
```

## 긍/부정리뷰 갯수 세기

``` r
result.table<-table(result$score)
result.table<-as.data.frame(result.table,stringsAsFactors = F)
result.table$Var1<-as.integer(result.table$Var1)
# result.table

table.len<-length(result.table$Var1)

number.of.positive.review = 0
number.of.negative.review = 0
number.of.neutral.review = 0

# result.table$Var1[1]
for (i in 1:table.len){
  ifelse(result.table$Var1[i] > 0, number.of.positive.review <- number.of.positive.review + result.table$Freq[i], 
         ifelse(result.table$Var1[i] < 0, number.of.negative.review <- number.of.negative.review + result.table$Freq[i], 
                number.of.neutral.review <- number.of.neutral.review + result.table$Freq[i]))
}

print(paste("+ :", number.of.positive.review)) #998
```

    ## [1] "+ : 406"

``` r
print(paste("- :", number.of.negative.review)) #137
```

    ## [1] "- : 39"

``` r
print(paste("0 :", number.of.neutral.review))
```

    ## [1] "0 : 24"

## 전체 평점 그래프

``` r
ggplot(result.table, aes(x=Var1,y=Freq))+
  geom_col(fill = 'royalblue')+
  xlab('Sentiment Score')+
  ylab('Reviews')+
  theme_bw()+
  geom_vline(aes(xintercept=0))+
  ggtitle('Sentiment-Score for Review')
```

![](SA_AmazonRiview_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

## 검증그래프 (score / stars)

``` r
result.plot<-result
  
ggplot(result.plot,aes(x=score, y=star))+
  geom_jitter()+
  geom_vline(aes(xintercept=0))+
  geom_hline(aes(yintercept=3)) 
```

    ## Warning: Removed 44 rows containing missing values (geom_point).

![](SA_AmazonRiview_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

## 개요

``` r
result %>% 
  arrange(desc(score)) %>%
  head(10) 
```

    ##     id
    ## 1    3
    ## 2  466
    ## 3  475
    ## 4    5
    ## 5   23
    ## 6   51
    ## 7  473
    ## 8  482
    ## 9  502
    ## 10 472
    ##                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            content
    ## 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              The Samsung 970 Pro 1TB NVMe SSD is the best SSD on the market today. With the revised Samsung pricing been 130 dollars less than the original MSRP this SSD is now a very very good value. When you look at the performance and advantages of the 970 Pro over the 970 EVO and other SSDs on the market it becomes very clear this SSD is the best. First of all, the type of NAND the 970 Pro uses is different than almost any other SSD on the market. While almost every SSD on the market is made of 3-bit 3D NAND, the 970 Pro uses the much better 2-bit MLC V-NAND. This NAND has several advantages over 3-bit 3D NAND, much better performance, much better endurance, more reliable and the most important one, doesn't slow down like 3-bit 3D NAND does. Samsung has 5 years warranty on this drive, has the best support in the business and when you look at the overall picture, the Samsung 970 Pro is well worth the extra money over the 970 EVO in my opinion. The experience by using this SSD is awesome. Your PC feels so fast and responsive. I highly recommend that if you can spend the extra 100 for the 970 Pro just do it. The advantages are very clear. This is a wonderful wonderful SSD. 10/10. 
    ## 2  Title: Writes as Fast as It Reads!Current system is as followsGigabyte z370 Aourus Mainboard16 gigabytes DDR4 RAMSamsung 950 pro SSD Boot Drive2 TB Mechanical storage DriveNvidia Gtx1070Intel 240 Gig ssd for Music productionIntel core i5 8th Gen 8670K unlocked and running at 4.2 ghzPackaging:Very well packaged and well protected in a harder shell plastic shellInstallation:Easy install. NVMe slots means the drive can only be insertedone way with a single screw to hold the drive in place.Cloning from previous drive:Installing the Samsung data migration software downloaded from their website went smoothly. Then came rebooting and launching the software. It selected the current boot disk as source, selected the 970 pro as destination. Then it was just click OK to begin the cloning process. Cloning drives to SSD's usually takes anywhere from 15 minutes to a 2 hours depending on the amount of data to be copied and the speed of the source disk. Also there is typically a little bit of testing to be done afterwards because (especially with Quickbooks) something doesn't copy correctly and a reinstallation is needed to get things working again.Not this time. The write speed of the new drive was impressive.Typical SSD's have a very fast read speed and a not so fast write speed. So cloning can take a while as the write speed is significantly slower than read. Cloning from a Samsung 950 pro SSD, the read speed out of the 950 would normally be buffered down to the write speed of the new 970. This was not the case. The drive with 266 gigabytes of data was cloned in just under 10 minutes. The Write speed of the new 970 is just as fast as the read speed of the 950.The transfer rate of 311 MB/s is ridiculously fast. Imagine installing photoshop in 3 seconds. If it was just a raw datatransfer, that would be possible. Basically a transfer of a gigabyte of data in just under 4 seconds.Once cloning was completed, booted into system BIOS and changed the boot drive to the new 970 and reset.Initial boot took about 12 seconds, still really fast but not as fast as the 950 just replaced. Once windows fully booted it discovered the drive and installed the appropriate NVMe driver and asked for a reboot.The next boot was significantly faster, up and running and into the desktop in about 4 seconds.Testing:Quickbooks has never worked after cloning a drive without some sort of “not installed properly” error. The only way to resolve it was to uninstall it, then delete the Intuit folder out of the hidden AppData folder at the root of C:, and reinstall. Unprecedentedly, it installed with no error!This must be due to the Samsung cloning software doing something different than anyone else. ( usually EZ GIG IV)Everything else just plain worked. Typical launch times increased quite a bit, Word 1 second, Excel 1 second, Quickbooks 3 seconds, browser windows almost instantly, Photoshop and Lightroom 5 seconds. So nice.SSD's have been great for many years and Samsung has always made the best. This one is no exception. It is extremely fast, very reliable, and combined with the Samsung migration software is a winning combo.Absolutely recommended!!(thanks to a very smart nephew-in-law) 
    ## 3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          Installed this in my new build to use for media creation projects. Installed easy on my Maximus X Hero motherboard, just had to create a volume on the drive and it was ready to go. Blazing fast speed, much faster than I'm used to on my old (HDD based computer) and faster than using my 1TB EVO (which holds my operating system). Taking full advantage of my 8700k processor is important and I love how easy this is to work with even though I'm a non-professional relative novice (this was my first complete build).The included software provides very helpful information about the drive, and the cloning software works amazingly easy, so that you can spend time working not time fiddling about.Couldn't be happier and now to get back to work... 
    ## 4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      I recently acquired the new MBP 15.4" and wanted to utilize its Thunderbolt 3 technology to the fullest and started looking around for a great external SSD. I settled on the TEKQ Rapide Thunderbolt 3 SSD and decided I want it maxed out in terms of performance so I took the plunge on this Samsung 970 PRO 1TB PCIe NVMe - M.2 to replace the TEKQ's internal SSD.Well, I'm truly blown away by the performance of this SSD. Samsung is definitely on the cutting edge with SSD. This thing smokes. I'm not a video/photo pro myself but do handle large media files for the company's marketing work which involves editing and shuttling of studio photos of products and demo videos. I also oversee operations so I don't want such work to slow me down in anyway.I can happily say that the TEKQ Rapide Thunderbolt 3 with this Samsung 970 PRO SSD is, by far, the best external SSD I've ever had the pleasure of using. I suppose it'll only get faster and better in the future but, for what I do, this is about as good as it gets. It feels and works as fast as my iMac 27" 5K and MBP's internal SSDs. An HD movie file transfers in a split second.I'm not a geek so I don't care to measure the speed using an app like Blackmagic. I can easily sense it while doing real life work and it makes me more productive. I have the HP P800 as well in the work office and it's great but the TEKQ with this Samsung SSD is a step above, even if we're talking a fragment of a second. Awesome stuff. I don't see myself outgrowing this SSD for a long time to come. 
    ## 5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           I’m new to these NVME drives but definitely knew I wanted to try one for a new build. I installed this on a Asus x470-I mini itx motherboard with R7 2700x and 16GB of Corsair LPX 3000MHz RAM. The board recognized it and installed windows on it very quickly from a USB 3.0 thumb drive.The speeds are incredible and I’ve used SSD on my builds for the past 9 years and am blown away by how much faster these drives perform compared. The speed, physical size, and capacity are impressive and the price while higher than 2.5 inch SSD’s are pretty reasonable for what you get in performance.This is the first build I’ve used only solid state drives for all of my in build storage and this drive definitely helps to keep things snappy. If you’re on a budget then there isn’t a need to use this over a SSD. The real world performance you’ll see is improved but the jump from HDD to SSD is already huge and enough for most. You’ll be better off spending less and getting a TB in storage if you’re on a budget. If you you’re willing to spend a bit more then it is easy to recommend this drive. 
    ## 6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           I love it … easy to install for someone like me with self-build computer skills. Actually easier than a HDD or SSD, and is it FAST … WOW. Had to remove the graphics card to gain access to the M.2 socket which was partial blocked beneath … but it slipped in quickly. Took me about 30 minutes, including doing a quick vacuuming of the tower insides. Booted computer … entered Disk Management … formatted SSD … and voila, DONE. One of the best upgrades I've done. I think the price is well worth it. And did I say FAST!(Added) BTW … the Amazon website indicated that this "may not" work with my ASDUS Maximus VIII Formula motherboard … it definitely DOES. Check your motherboard specifications, as all you need is an M.2 socket with PCle/SATA support.. 
    ## 7                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       -Note: I didn't buy this on amazon-I did buy the laptop on Amazon if that counts..The 970pro is an awesome drive. I upgraded my 2.5 drive first to a 860evo, which was a great step up from the 5400rpm drive, especially with rapid. Then yesterday, I decided to upgrade my m.2 from the factory SanDisk 250g sata3 to the 970pro, I did have to get a 2.5 to m.2 case to clone the drive but all really went well.The 970pro really kicked my computer up a notch. Everything feels really snappy and quick. Almost too quick. But that's a good thing.. I am using this laptop as my personal workstation for work.. I have a program that I run for work and created a bench mark test that taxes all components... Cpu, ram, and storage(not gpu intensive at all). But what I have found is; stock form, my computer did one task in 3min 45seconds. After the 860, that went down to 3:15-3:30, and now I'm doing it in 2:45. A solid minute over stock, which is ~ 30% system improvement over stock.Boot time is down to almost half, 30seconds to 17seconds to log in screen.Laptop used is a Asus gl702zc with a ryzen 7 1700 and 32gb ram scoring around 1550 on cinebench. This computer is a power house now! Really happy with the performance of this drive. Wouldn't hesitate buying one again and I might if I decid to pull the trigger on a new nuc hades canyon..Ultimately all pcie m.2s are going to be, or should be, insanely fast but the 970pro is a no compromise ssd that you can trust every day. Don't hesitate if you're ready and able. 
    ## 8                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         Installs really easily - check that your motherboard supports it. This is a TINY drive that requires an M.2 slot on your motherboard. M.2 is the form factor. If your motherboard supports M.2 then you should be good, although you may not get fastest speeds unless it also supports NVMe.If you're just adding to an existing rig, I recommend the Disk Management tool that's built into windows, to add a partition and format.I'm still using it and comparing it to my older SSD. So far the general usage times are noticeably faster than my Samsung 850 EVO 250GB, which uses the older SATA interface and form factor.500GB is good for your primary volume and a most apps, but probably not for a full Steam library or a lot of media. At around $200 it's probably the sweet spot if you don't mind some active management of your drives.Overall, I'm pretty happy with this as a workhorse main drive. If you're building a tiny PC (I'm not, this is a full size desktop) then you get extra value from the tiny size.Note - it doesn't come with an anchoring screw, so dig out the screws from your motherboard, or be ready to go hunting!s exactly what we were looking for and we had no problems with its use. 
    ## 9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          I upgraded my tower recently (music production and video editing) with a new i9 processor, motherboard and DDR4 ram. In the midst of that this 970 Pro showed up for review and I knew it was worth the tax hit to take it (Vine stuff isn't really free, lol). I made it my C drive in the dedicated PCIe 4x M.2 slot on the motherboard. I used userbenchmark dot com to get results. The results were beyond amazing:Rank 4 - 335%Read 2,352Write 1,966Mixed 1,294SusWrite 1,860419% 1,868 MB/sCopying a 5GB video file from this drive to another SSD I nearly 2gbB/sec, just crazy. Speeds like that are really needed in multitrack audio and MIDI orchestration where we are reading 1TB libraries of orchestral samples from other SDs on the system, and then the system needs to play everything back in real time. That's where the Samsung 970 Pro is worth its weight in gold 10 times over, there's no more hard drive bottle neck which MIDI composers even got from SATA SSDs. Being able to run dozens of tracks in Studio One with a low bit rate and have no crackle and 4ms latency is pretty crazy (that will mean something to audio engineers and MIDI composers). Video editing now, with my Assets on another 970 non Pro, is equally amazing. Full HD video renders in way less time that real time in Premiere Pro, even with multiple plugins. So I would have to say that gamers would benefit hugely as well, because the OS layer has always been choked by the C drive, and that's not a limitation any longer. The technology isn't cheap but it's already cheaper than 18 months ago. But even using this as a data drive when you have to stream in assets will be phenomenally faster (scenery for games, video for editing, samples for audio, etc). For those who do MIDI composing, I have the Pro for C drive and the non Pro 970 for libraries, and I can load a 70 seat orchestral patch from the full NI Symphony Series in under 3 seconds. When you need speed, SATA SSDs are just standard now, not exciting, the NVMe SSDs are worth every penny. Mechanical drives are for backups are archives, they're not even usable for running a PC in 2018. ;-D 
    ## 10                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          NVMe drives like this are the way mainstream storage is headed. It is nice to just pop a tiny drive onto your motherboard and not need power or data cables to make it work. If you pick up a good drive like this from a company that backs their product it is hard to go wrong. All of these drives are fast enough that unless you have a specialized use case and you know you’ll benefit from a particular drive you generally won’t notice any difference in the during the course of normal use. I ran this drive in my Ryzen system and compared it to other NVMe drives. The drives I used for comparison were the 1TB and 512GB Western Digital Black, 1TB Samsung 960 Evo, 1TB Samsung 970 Evo, and the 1TB HP EX920. This drive did come in first when tested using PCMARK 8 but the numbers were so close that I doubt it’d matter in normal usage. I didn’t notice a difference. This drive is supposed to have a better endurance rating so like I said earlier about specialized use cases this might benefit you there. So, in the end, I’d recommend this drive and say to choose whatever drive that is priced the best at the time you are buying as long as it has the backing and warranty that you want. 
    ##                 date star score total positive negative  pos.rate
    ## 1      July 12, 2018    5    23    25       24        1 0.9600000
    ## 2      June 27, 2018    5    21    29       25        4 0.8620690
    ## 3      July 13, 2018    5    17    17       17        0 1.0000000
    ## 4    August 18, 2018    5    16    20       18        2 0.9000000
    ## 5  November 20, 2018    5    13    13       13        0 1.0000000
    ## 6    January 8, 2019    5    13    13       13        0 1.0000000
    ## 7       July 1, 2018    5    13    15       14        1 0.9333333
    ## 8       July 4, 2018    4    11    13       12        1 0.9230769
    ## 9  September 4, 2018    5    11    23       17        6 0.7391304
    ## 10     June 19, 2018    5    10    16       13        3 0.8125000
    ##      neg.rate
    ## 1  0.04000000
    ## 2  0.13793103
    ## 3  0.00000000
    ## 4  0.10000000
    ## 5  0.00000000
    ## 6  0.00000000
    ## 7  0.06666667
    ## 8  0.07692308
    ## 9  0.26086957
    ## 10 0.18750000

``` r
result %>% 
  arrange(score) %>% 
  head(10)
```

    ##     id
    ## 1    4
    ## 2  312
    ## 3   35
    ## 4   42
    ## 5   50
    ## 6   59
    ## 7   11
    ## 8   25
    ## 9   53
    ## 10  61
    ##                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           content
    ## 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             Have this in my ASUS ROG Strix X470-I SFF (small form factor) PC. 31-second boot time from a completely powerless pc, with 18-22 programs that start on start-up. 28-second complete restart cycle, again with 18-22 active programs on start-up. Because this is the highest rated PCIe NVMe - M.2 for consumers, of course, it makes everything I have ever owned before pale in comparison. I'll lastly say, I have zero regrets with the purchase. Highly recommend, especially to those who are interested in SFF computers.Ryzen 7 2700xIC Graphite Thermal Pad 40x40ASUS ROG Strix X470-ISamsung 970 Pro 1TB PCIe NVME M.2CORSAIR VENGEANCE RBG 32GB (2x16GB) DDR4 3000 MhzNH L12 - Noctua CPU CoolerSF Corsair 600w PSU1080 TI FECase: Louqe Ghost S1 MkII 
    ## 2                                                                                                       I bought the 2950x, Corsair 760T chassis, Asus Zenith Extreme, Corsair HX1000I PSU, Thermaltake Floe Riing 360, Corsair LPX 32GB (2x16GB) 3200MHz C16 DDR4 DRAM Memory Kit, Black (CMK32GX4M2B3200C16), SamSung EVO 970 512G NVMe, WD 1TB Blue NVMe, Toshiba HD 6TB x3 from Amazon. The CPU arrived but the chip was loose inside the box. I could not install Windows 10 but was able to install Ubuntu 18.04 LTS. When I ran Chrome or Firefox under , the CPU kept hanging up. I changed to Corsair Vengeance LPX 32GB (2x16GB) DDR4 DRAM 2133MHz (PC4 17000) C13 Memory Kit - Black, thinking may be memmory compabibil;ity from what I read on Youtube, web. But the problem still occured, can not install windows, kept on hanging when runing Chrome, FireFox under Ubuntu. Changed to G.SKILL Flare X Series 16GB (2 x 8GB) 288-Pin DDR4 SDRAM DDR4 3200 (PC4 25600) AMD X370 Memory Model F4-3200C14D-16GFX. Same problem occrued. I changed the mem clock to from 1600-3200 for 3200 MHZ and 1600-2133 MHZ for 2133 MHZ and 1600-3200 for G.Skill. Same problem. Can not install Windows 10. hung when ran Chrome under Ubuntu. Hung when ran Geekbench when it run under multi core, single core is ok. I ran mem test and it was ok. Thinking the CPU was damage during ship, I ask for replacement and got one. When i installed in the system ansd ran with G.Skill, It was able to boot windows 10, Ubuntu and no problem.I had to upgrade the BIOS to 1402 for the Asus Zenith to boot with 2950x.Now it works very smoothly. It run fast!!!!,Very happy with the unit. 
    ## 3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   It was a bit hard to install, but that was just due to my case and oversized graphics card, it probably wouldn't be as difficult in other cases. It wasn't a big deal whatsoever, though. The speed is really amazing, although I wish the one terabyte NVMe was cheaper, as I find myself keeping most of my games and such on my hard drive to save space, missing potentially huge improvements in loading times. One wonderful issue that has been resolved as a result of using this NVMe drive is USB 3.0 transfer speeds. Now I can finally transfer files at USB 3.0 speeds, without frequent and sudden drops or general low speed. I've noticed that even transferring files from my hard drive to a USB is faster now, which I can only assume is due to the fact that my Windows system isn't taking up any hard drive bandwidth by running on the hard drive as I read from it. 
    ## 4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           I have 2 of these in Raid 0 configuration (two drives linked together to split the load while working as one unified drive). They are installed in my msi GT83 laptop. While I did not noticed much of a real world difference over 2 Samsung Evo's in Raid 0, the test did show an increase. I include a photo of tests. The tests were run 5x for each type of hard drive and the test showing the average was put into the photo. 
    ## 5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                This SSD lives up to the hype. I dropped it in an ASUS Maximus X Formula Motherboard to replace a Samsung 850 EVO. I made sure and confirmed the ePCI lanes were set to 4 on the M.2 slot I used, rebooted, installed the Samsung driver for Windows 10... loaded up the Data Transfer / clone software from Samsung and cloned the 850 EVO at around 325 megs a second as expected. After rebooting on the new 970 Pro, I did a quick performance test and it was spot on, 3500 megs / 3.5 gigs a second. You won't be disappointed with this latest offering from Samsung. It's 'crazy' fast! 
    ## 6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      Bought this before realizing my system didn’t even have a port for it. Ended up buying a whole new computer with all the new parts. I don’t regret it. This beast is ludicrously quick. 
    ## 7                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   I was really looking forward to this fast NVMe drive, but a HUGE letdown for me was its inability to support hardware-based full-disk encryption with the "GIGABYTE X299 AORUS GAMING 7 PRO" motherboard. It took me a week of experimentation and re-installation of Windows 10 and back and forth email communication with the Samsung tech support, only to give up in the end.How is it, Samsung, that I can set up hardware encryption with your older SSD drives but not with this new and expensive drive?Also, how about your train your tech support people? I had a very poor experience. At times I thought that I knew more about the subject than they did. They were giving me conflicting answers every time I received a reply. Eventually they just stopped answering my questions.... sad! 
    ## 8  So I've been on the prowl for a new SSD to upgrade my early 2015 MBP from it's meager 256gb. I pulled the trigger and scooped this one up and the difference is night and day. I never full grasp how slow my older SSD was until I installed this joint.First off you'll need to get an adapter and this is the one that helped: https://www.amazon.com/gp/product/B01CWWAENG/ref=oh_aui_detailpage_o04_s00?ie=UTF8&psc=1Of the multitude of examples I looked for make the process seamless, honest the way i did it was fool proof and straight a ahead. First off, make sure you get a decent clone app (i.e. carbon clone) and a spare ssd that can handle the space you'll need to do the swap to the newer SSD. MAKE THE CLONE YOUR NEW BOOT DESTINATION. It's infinately easier than getting another usb, putting High Sierra on it, and dealing with the speed - CAUSE IT'S SLOW. Once I made the clone ssd the boot drive there was no lag - it was if the clone was the installed drive. Secondly, once you install this SSD into the MBP next (a priority for anyone who knows) format the newer SSD. Once that's done, you'll need to restart the computer and go through the process of "restoring" the drive by holding option during the restart process. Use your clone SSD as the source to restore your SSD and use the migrate data function. Next, wait. It took about an hour or so for the migration to take place. More than likely, it's due to how much your moving over.I never had the issue of my computer with going to sleep and I occasionally have the issue when I restart the computer, where it take a while or doesn't fully restart but so far so good. 
    ## 9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    I used Samsung Magician to compare the factory 256Gb M.2 PCIe drive in my $1200 Eluktronics P60HP6 to the Samsung 512gb 970 Pro. It averaged around 8X faster.Sequential speed (MB/s)Read: 450 vs 3,458 Write: 272 vs 2.331Random (IOPS)Read: 30,517 vs 243,896 Write: 26,367 vs 208,251I can tell the difference even on just surfing the internet and booting. I didn't expect the drive Eluktronics included in a high end laptop to be so slow compared to the 970 Pro. 
    ## 10                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 This thing is FAST. Samsung's benchmark showed 3500+ MB reads. Had a moment of confusion trying to set it up in bios - it didn't show up as bootable device at first. Apparently you need to install OS on it first, then it magically pops up in the list.P.S. Wish the 1 TB one wasn't so darn expensive... 
    ##                 date star score total positive negative  pos.rate
    ## 1      June 29, 2018    5    -3     5        1        4 0.2000000
    ## 2   October 18, 2018    5    -3    15        6        9 0.4000000
    ## 3   October 30, 2018    5    -2    12        5        7 0.4166667
    ## 4   February 9, 2019    5    -2     2        0        2 0.0000000
    ## 5  December 14, 2018    5    -2     4        1        3 0.2500000
    ## 6      July 20, 2018    5    -2     2        0        2 0.0000000
    ## 7    October 8, 2018    3    -1     9        4        5 0.4444444
    ## 8  September 8, 2018    5    -1    13        6        7 0.4615385
    ## 9   January 13, 2019    5    -1     1        0        1 0.0000000
    ## 10  January 30, 2019    5    -1     3        1        2 0.3333333
    ##     neg.rate
    ## 1  0.8000000
    ## 2  0.6000000
    ## 3  0.5833333
    ## 4  1.0000000
    ## 5  0.7500000
    ## 6  1.0000000
    ## 7  0.5555556
    ## 8  0.5384615
    ## 9  1.0000000
    ## 10 0.6666667

``` r
target.postop <- which.max(result$score) 

reviews$content[target.postop]
```

    ## [1] "The Samsung 970 Pro 1TB NVMe SSD is the best SSD on the market today. With the revised Samsung pricing been 130 dollars less than the original MSRP this SSD is now a very very good value. When you look at the performance and advantages of the 970 Pro over the 970 EVO and other SSDs on the market it becomes very clear this SSD is the best. First of all, the type of NAND the 970 Pro uses is different than almost any other SSD on the market. While almost every SSD on the market is made of 3-bit 3D NAND, the 970 Pro uses the much better 2-bit MLC V-NAND. This NAND has several advantages over 3-bit 3D NAND, much better performance, much better endurance, more reliable and the most important one, doesn't slow down like 3-bit 3D NAND does. Samsung has 5 years warranty on this drive, has the best support in the business and when you look at the overall picture, the Samsung 970 Pro is well worth the extra money over the 970 EVO in my opinion. The experience by using this SSD is awesome. Your PC feels so fast and responsive. I highly recommend that if you can spend the extra 100 for the 970 Pro just do it. The advantages are very clear. This is a wonderful wonderful SSD. 10/10. "

## 가장 부정적인리뷰 보여주기

``` r
target.negtop <- which.min(result$score) 

reviews$content[target.negtop]
```

    ## [1] "Have this in my ASUS ROG Strix X470-I SFF (small form factor) PC. 31-second boot time from a completely powerless pc, with 18-22 programs that start on start-up. 28-second complete restart cycle, again with 18-22 active programs on start-up. Because this is the highest rated PCIe NVMe - M.2 for consumers, of course, it makes everything I have ever owned before pale in comparison. I'll lastly say, I have zero regrets with the purchase. Highly recommend, especially to those who are interested in SFF computers.Ryzen 7 2700xIC Graphite Thermal Pad 40x40ASUS ROG Strix X470-ISamsung 970 Pro 1TB PCIe NVME M.2CORSAIR VENGEANCE RBG 32GB (2x16GB) DDR4 3000 MhzNH L12 - Noctua CPU CoolerSF Corsair 600w PSU1080 TI FECase: Louqe Ghost S1 MkII "

## TDM

``` r
mycorpus<-VCorpus(VectorSource(result$content))
temp2<-mycorpus
for(i in 1:length(mycorpus)){
  # 1. 공백, 숫자, 기호 전처리
  temp2[[i]]$content<-str_replace_all(temp2[[i]]$content,"[[:space:]]{1,}", " ")
  temp2[[i]]$content<-str_replace_all(temp2[[i]]$content,"[[:digit:]]{1,}","")
  temp2[[i]]$content<-str_replace_all(temp2[[i]]$content,"[[:punct:]]{1,}","")
}

# 2. 불용어 사전 이용해서 불용어 제거
tm_map(temp2, FUN = removeWords, words = stopwords("en"))->temp2
#tm_map(temp2, FUN = removeWords, words = stopwords("SMART"))->temp2
temp2
```

    ## <<VCorpus>>
    ## Metadata:  corpus specific: 0, document level (indexed): 0
    ## Content:  documents: 513

``` r
# 3. wordstem으로 어근 변환 # 
for(i in 1:length(mycorpus)){
  temp2[[i]]$content<-wordStem(temp2[[i]]$content)
}

myTDM<-TermDocumentMatrix(temp2)
ti<-weightTfIdf(myTDM)
```

    ## Warning in weightTfIdf(myTDM): empty document(s): 438 439

## tfidf기반 Keyword 추출

``` r
tempV <-list()

# class(ti[,1]$dimnames$Terms[head(order(ti[,1]$v, decreasing = T),10)])
                      
# result$keyword[1]
# ti[,1]$i # 1번 문서에 등장한 단어 index
# ti[,1]$v # 1번 문서에 등장한 단어의 가중치
# 
# (order(ti[,1]$v, decreasing = T)) # 1번 문서에서 가중치 순위의 index # 가중치가 높은게 1이 되도록
# sort(ti[,1]$i,)

# max(ti[,1]$v)
ti[,1]$v[66] # -> v의 max가 66번 인덱스다. order의 첫번째와 일치
```

    ##          1 
    ## 0.05080721

``` r
test<-head(order(ti[,1]$v, decreasing = T), 10) # 그중에서 상위 10개 인덱스

# 1번 리뷰에서 가중치가 높은 단어들 
top10 <- ti[,1]$dimnames$Terms[test] # 가중치가 가장 높은 10개 단어 추출
```
