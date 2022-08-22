"""

시나리오를 만든 이유
각 인텐트마다 필요한 엔티티가 다르게 정해져 있기 때문.
ex) 신체불편호소 : 신체부위(BODY), 증상(SYMPTOM)

@intent 종류
1) 궁금함
-weather
-dust
-restaurant
-travel

2) 불편함
-physicalDiscomfort
-sleepProblem
-moveHelp
-changePosture
-higieneAct
-otherAct
-environmentalDiscomfort
-expressDesire
-foodDiscomfort

3) 감정 불편 호소
sentimentDiscomfort

@entity 종류
-BODY
-SYMPTOM
-PLACE
-FOOD
-DAILYSUPPLIES
-ACT(IVITY)
-HOMEAPP
-ASSISTANTDEV
-RELIGIOUSACT
-TIME
-WEATHER
-MONEY
"""

from answerer.curious_answerer.dust import DustCrawler
from answerer.curious_answerer.weather import WeatherCrawler
from answerer.curious_answerer.time_answerer import TimeAnswerer
from scenarios.scenario import Scenario
from answerer.discomfort_answerer import DiscomfortAnswerer
#from answerer.emotion_answerer import EmotionAnswerer


#answerer = EmotionAnswerer()


weather = Scenario(
    intent='weather',
    api=WeatherCrawler().request,
    scenario={
        'LOCATION': [],
        'DATE': ['오늘']
    }
)

dust = Scenario(
    intent='dust',
    api=DustCrawler().request,
    scenario={
        'LOCATION': [],
        'DATE': ['오늘']
    }
)

time = Scenario(
    intent='time',
    api=TimeAnswerer().now_time_form,
    scenario={
        'DATE': ['오늘']
    }
)

date = Scenario(
    intent='date',
    api=TimeAnswerer().now_date_form,
    scenario={
        'DATE': ['']
    }
)

weekday = Scenario(
    intent='weekday',
    api=TimeAnswerer().now_weekday_form,
    scenario={
        'DATE': ['']
    }
)

### 추가
# 불편함

requestAct = Scenario(
    intent='활동요구',
    api=DiscomfortAnswerer().requestAct_check_form,
    scenario={
    }
)

environmentalDiscomfort = Scenario(
    intent='환경불편호소',
    api=DiscomfortAnswerer().environmentalDiscomfort_check_form,
    scenario={
    }
)

'''
environmentalDiscomfort = Scenario(
    intent='환경불편호소',
    api=DiscomfortAnswerer().environmentalDiscomfort_check_form,
    emotion_answerer=answerer,
    scenario={
        'PLACE': []
    }
)

sleepProblem = Scenario(
    intent='수면문제호소',
    api=DiscomfortAnswerer().sleepProblem_check_form,
    emotion_answerer=answerer,
    scenario={
    }
)

foodDiscomfort = Scenario(
    intent='음식불편호소',
    api=DiscomfortAnswerer().foodDiscomfort_check_form,
    emotion_answerer=answerer,
    scenario={
        'FOOD': []
    }
)
'''

physicalDiscomfort = Scenario(
    intent='신체불편호소',
    api=DiscomfortAnswerer().physicalDiscomfort_check_form,
    scenario={
        'BODY': [],
        'SYMPTOM': ['']
    }
)


'''
moveHelp = Scenario(
    intent='이동도움요구',
    api=DiscomfortAnswerer().moveHelp_check_form,
    scenario={
        'PLACE': []
    }
)

changePosture = Scenario(
    intent='자세변경요구',
    api=DiscomfortAnswerer().changePosture_check_form,
    emotion_answerer=answerer,
    scenario={
    }
)

higieneAct = Scenario(
    intent='위생활동요구',
    api=DiscomfortAnswerer().higieneAct_check_form,
    emotion_answerer=answerer,
    scenario={
    }
)

otherAct = Scenario(
    intent='기타활동요구',
    api=DiscomfortAnswerer().otherAct_check_form,
    emotion_answerer=answerer,
    scenario={
    }
)
'''


sentimentDiscomfort = Scenario(
    intent='마음상태호소',
    api=DiscomfortAnswerer().requestAct_check_form,
    scenario={
    }
)


# 감정대화는 엔티티 대신 emotion, topic을 요구함