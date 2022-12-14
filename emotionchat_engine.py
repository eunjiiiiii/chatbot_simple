from scenarios.scenario_manager import ScenarioManager
import emotionchat_config as config
from model.emotion.predict import IAI_EMOTION
from model.topic.predict import IAI_TOPIC
from model.intent_entity.intent_entity import JointIntEnt
from scenarios.default_scenario import dust, weather, time, date, weekday, requestAct, physicalDiscomfort, environmentalDiscomfort, sentimentDiscomfort
from model.proc import DistanceClassifier, GensimEmbedder, EntityRecognizer
from data.dataset import Dataset
from model import curious_intent, embed, curious_entity
from model.loss import CenterLoss, CRFLoss


class EmotionChat:

    def __init__(self):

        """
        emotionChat 답변생성 클래스 입니다.

        :param embed_processor: 궁금함용 임베딩 프로세서 객체 or (, 학습여부)
        :param intent_classifier: 궁금함용 인텐트 분류기 객체 or (, 학습여부)
        :param entity_recognizer: 궁금함용 엔티티 인식기 객체 or (, 학습여부)
        """

        self.dataset = Dataset(ood=True)

        self.intent_entity_classifier = JointIntEnt("./model/intent_entity/jointbert_demo_model", no_cuda=True)
        self.emotion_recognizer = IAI_EMOTION
        self.topic_recognizer = IAI_TOPIC("./model/topic/model", no_cuda=True)

        dataset = Dataset(ood=True)
        emb = GensimEmbedder(model=embed.FastText())

        clf = DistanceClassifier(
            model=curious_intent.CNN(dataset.intent_dict),
            loss=CenterLoss(dataset.intent_dict),
        )

        rcn = EntityRecognizer(
            model=curious_entity.LSTM(dataset.entity_dict),
            loss=CRFLoss(dataset.entity_dict)
        )

        self.scenario_manager = ScenarioManager(embed_processor=(emb, False),
                                                  intent_classifier=(clf, False),
                                                  entity_recognizer=(rcn, False))

        self.scenarios = [weather, dust, time, date, weekday,
                          physicalDiscomfort,
                          requestAct,
                          environmentalDiscomfort,
                          sentimentDiscomfort]

        for scenario in self.scenarios:
            self.scenario_manager.add_scenario(scenario)

        self.emotion_to_pce = {key: idx+1
                               for idx, key in enumerate(['분노','슬픔','불안','놀람','기쁨','평온함(신뢰)'])}
        self.topic_to_pce = {key: idx + 1 for idx, key in enumerate(['건강','사회환경','정서인지','경제'])}

    def run(self, text: str, wav_file, pre_result_dict: dict, turn_cnts: dict) -> dict:
        """
        인텐트 인식 후 단계 확인 후 시나리오에 적용해주는 함수
        모든 사용자 발화 text는 이 함수를 먼저 거쳐야 함.
        :param text: 사용자 input text
        :param wav_file: 사용자 음성 파일(.wav)
        :param pre_result_dict: 이전 단계 result_dict
        :param turn_cnts: 이전까지 turn_cnt, intent_turn_cnt 를 value로 가지는 딕셔너리
        :return: result_dict 딕셔너리
        """

        # 0. 이전 result_dict value들을 클래스 인스턴스 변수에 저장
        pre_tokens = pre_result_dict['input']  # 이전 input들의 누적된 token 리스트
        pre_phase = pre_result_dict['current_phase']   # 문자열 형태의 이전 단계
        pre_pred_phases = pre_result_dict['next_phase']    # 이전 단계에서 예상한 다음 단계 리스트
        pre_intent = pre_result_dict['intent'] # 이전 단계 인텐트
        pre_emotion_prob = pre_result_dict['emotion_prob'] # 이전 단계까지의 누적 감정 확률 리스트
        pre_topic_prob = pre_result_dict['topic_prob'] # 이전 단계까지의 누적 주제 확률 리스트
        pre_emotions = pre_result_dict['emotions'] # 이전 단계까지의 누적 감정 리스트
        pre_topics = pre_result_dict['topics'] # 이전 단계까지의 누적 주제 리스트
        intent_turn_cnt = pre_result_dict['intent_turn_cnt']   # 이전 단계까지 이전 단계와 같은 인텐트 턴 횟수
        turn_cnt = turn_cnts['turn_cnt']



        # 1. 불편함/궁금함 인식 ,감정인식/주제인식 일 경우 intent 인식하지 않음
        c_ucs = True    # 이전 단계에서 불,궁,감 대화에 들어왔는가?
        if pre_phase != '' and pre_intent not in (
                config.SORT_INTENT['PHISICALDISCOMFORTnQURIOUS'] + config.SORT_INTENT['SENTIMENTDISCOMFORT']):

            # 이전 단계가 불편함, 마음상태호소, 궁금함 X -> 인텐트 인식
            intent, entity_ = self.intent_entity_classifier(text)
            c_ucs = False
        elif '/check_uc' in pre_pred_phases:
            # 이전 단계의 예상 단계에 /check_uc (재질의) 가 있을 경우 = 현재 예상 단계가 재질의일 경우
            intent, entity_ = self.intent_entity_classifier(text)
            # c_ucs = False  # 이미 인식했기 때문. 재질의 필요 x => c_ucs == False
            c_ucs = False
        elif pre_phase == '':
            # 인사
            intent = '인사'
            c_ucs = False
        else:
            # 이전 단계가 불편함, 마음상태호소, 궁금함 O -> 인텐트 인식 X
            _, entity_ = self.intent_entity_classifier(text)
            intent = pre_intent
            # 인텐트 인식 안해서 정확한 오류 확인할 수 없는 단점은 엔티티가 안채워진걸로 판단하면 됨
            # 감정은 감정확률이나 emotion의 유무


        # 2. intent_turn_cnt 기록

        # 이전 대화의 intent와 현재 대화의 intent가 같으면 intent_turn_cnt 기록
        if pre_intent == intent:
            intent_turn_cnt += 1

        elif pre_intent != intent:
            # 이전 대화의 intent와 현재 대화의 intent가 다르면 intent_turn_cnt 초기화
            intent_turn_cnt = 0


        # 3. inputs -> tokens
        tokens = list(text.split(' '))


        # 4. 인사 처리

        if (("안녕" in text) or (intent == '인사')) and pre_phase == '':
            # 첫번째 turn이고, 인사일 경우
            return {
                'input': tokens + pre_tokens,
                'intent': '인사',
                'entity': [],
                'state': 'SUCCESS',
                'emotion': '',
                'emotions': pre_emotions,
                'emotion_prob': pre_emotion_prob,
                'emotion_probs': [1., 1., 1., 1., 1., 1.],
                #'topic': '',
                'topics': pre_topics,
                'topic_prob': pre_topic_prob,

                'answer': config.ANSWER['welcomemsg_chat'],
                'previous_phase': '',
                'current_phase': '/welcomemsg_chat',
                'next_phase': ['/other_user', '/recognize_uc_chat', '/recognize_emotion_chat', '/recognize_uc',
                               '/recognize_emotion', '/recognize_topic', '/induct_emotion', '/generate_emotion_chat', '/check_uc',
                               '/fill_slot', '/end_phase'],
                'intent_turn_cnt': intent_turn_cnt
            }


        '''
        시나리오에 적용
        '''

        # 5. 엔티티 앞에 I-, B- 제외

        entity = self._edit_entity(entity_)

        # 6. 감정, 주제 라벨 & 확률값 받기
        if intent not in ['인사','작별인사']:
            emotion_label, emotion_probs_array, max_emotion_prob_array = self.emotion_recognizer().predict(text,
                                                                                                           wav_file)
            max_emotion_prob = float(max_emotion_prob_array)
            topic_label, topic_probs_array, max_topic_prob_array = self.topic_recognizer.predict(text)
            max_topic_prob = float(max_topic_prob_array)
            emotion, topic = self.__rename_emotion_topic(emotion_label, topic_label)


        ## scenario_manager() > apply_scenario()로 보내기 위해 result_dict로 변수 묶어놓음
        if intent == '인사':
            result_dict = {
                'input': tokens,
                'intent': intent,
                'entity': entity,
                'state': 'EMOTIONCHAT_ENGINE',
                'emotion': '',
                'emotions': [],
                'emotion_prob': [],
                'emotion_probs': [1., 1., 1., 1., 1., 1.],
                'topics': [],
                'topic_prob': [],
                'answer': '',
                'previous_phase': pre_phase,
                'current_phase': '',
                'next_phase': [],
                'intent_turn_cnt': intent_turn_cnt
            }
        else:
            result_dict = {
                        'input': tokens,
                        'intent': intent,
                        'entity': entity,
                        'state': 'EMOTIONCHAT_ENGINE',
                        'emotion': '',
                        'emotions': [emotion],
                        'emotion_prob': [max_emotion_prob],
                        'emotion_probs': emotion_probs_array[0],
                        'topics': [topic],
                        'topic_prob': [max_topic_prob],
                        'answer': '',
                        'previous_phase': pre_phase,
                        'current_phase': '',
                        'next_phase': [],
                        'intent_turn_cnt': intent_turn_cnt
                    }

        # 7. scenario_manager의 apply_scenario() 실행
        result_dict = self.scenario_manager.apply_scenario(pre_result_dict, result_dict,
                                                               text, c_ucs, turn_cnt)

        # 8. 단계 check & 단계 오류 해결

        if self.__check_phase(pre_pred_phases, result_dict['current_phase']):
            # return self.scenario_manager.apply_scenario(intent, entity, tokens, emotion, topic, intent_turn_cnt)
            return result_dict


        else:
            # 단계 오류 (예상한 단계가 아닐 경우)
            return self.__handle_phase_error(turn_cnt, pre_result_dict, result_dict, text)


    def _edit_entity(self, entity: list) -> list:
        """
        한글로 된 엔티티를 영어로 바꿔주는 함수
        default_scenario의 entity key를 영어로 써놨기 때문에 변경 필수
        :param entity: 엔티티(한글)
        :return: 엔티티(영어)
        """
        result_entity = []

        for e in entity:
            if '-' in e:
                result = e.split('-')[1]
                if result == '신체부위':
                    e = 'body'
                elif result == '증상':
                    e = 'symptom'
                elif result == '장소':
                    e = 'place'
                elif result == '음식':
                    e = 'food'
                elif result == '시간':
                    e = 'date'
            result_entity.append(e)

        return result_entity

    def __rename_emotion_topic(self, emotion_label: int, topic_label: int) -> str:
        """
        감정, 주제 라벨 값(int)을 문자열(한글) 바꿔주는 함수
        :param emotion_label: 감정 라벨 값(int)
        :param topic_label: 주제 라벨 값(int)
        :return: 감정, 주제 라벨 값(str)
        """

        emotion_dict = {0: '놀람', 1: '기쁨', 2: '분노', 3: '불안', 4: '슬픔', 5: '평온함(신뢰)'}
        topic_dict = {0: '건강', 1: '사회환경', 2: '정서인지', 3: '경제'}

        return emotion_dict[emotion_label], topic_dict[topic_label]

    def __check_phase(self, pre_pred_phases: list, current_phase: str):
        """
        단계 체크 함수
        이전 단계에서 예상한 단계에 현재 단계가 포함되는지
        :param pre_pred_phases: 이전 단계에서 예상한 단계(list)
        :param current_phase: 현재 단계(str)
        :return: bool(True, False)
        """
        if current_phase in pre_pred_phases:
            return True
        return False

        # 다음 단계가 종료면 서버측에서 종료

    def __handle_phase_error(self, turn_cnt: int, pre_result_dict: dict, result_dict: dict, text: str) -> dict:
        """
        단계 오류 처리하는 함수
        :return: 다 채운 dictionary
        """

        intent_turn_cnt = result_dict['intent_turn_cnt'] + 1
        if len(result_dict['input']) != intent_turn_cnt:
            result_dict['input'] = result_dict['input'] + pre_result_dict['input']
        result_dict['emotions'] = result_dict['emotions'] + pre_result_dict['emotions']
        result_dict['emotion_prob'] = result_dict['emotion_prob'] + pre_result_dict['emotion_prob']
        result_dict['topics'] = result_dict['topics'] + pre_result_dict['topics']
        result_dict['topic_prob'] = result_dict['topic_prob'] + pre_result_dict['topic_prob']

        if turn_cnt > 5:
            # 전체 turn 횟수가 6회가 넘으면 종료

            result_dict['emotion'] = pre_result_dict['emotion']
            result_dict['state'] = 'OVER_TURN_5'
            result_dict['answer'] = config.ANSWER['goodbyemsg_chat']
            result_dict['previous_phase'] = pre_result_dict['current_phase']
            result_dict['current_phase'] = '/end_phase'
            result_dict['next_phase'] = []


        else:
            # 전체 turn 횟수가 6회가 넘지 않았을 경우

            if pre_result_dict['intent'] == '인사' and result_dict['intent_turn_cnt'] <= 1:
                # 인사 하고 딴소리 하는 경우 -> 1번까지만 봐줌

                #print("인사 이후 대화 오류 들어옴")

                return result_dict


            elif result_dict['intent'] == '작별인사':
                # 원래 작별인사 타이밍이 아닌데 작별인사 하는 경우

                result_dict['emotion'] = pre_result_dict['emotion']
                result_dict['state'] = 'SUDDEN_GOODBYE'
                result_dict['answer'] = config.ANSWER['goodbyemsg_chat']
                result_dict['previous_phase'] = pre_result_dict['current_phase']
                result_dict['current_phase'] = '/end_phase'
                result_dict['next_phase'] = ['/end_phase']


            else:
                # 그 외에 에러 처리(인텐트 인식 잘못 했을 경우)

                result_dict['emotion'] = pre_result_dict['emotion']
                result_dict['state'] = 'ERROR_INTENT'
                result_dict['answer'] = config.ANSWER['default_error_ucs']
                result_dict['previous_phase'] = pre_result_dict['current_phase']
                result_dict['current_phase'] = pre_result_dict['current_phase']
                result_dict['next_phase'] = pre_result_dict['next_phase']

        return result_dict

    # def check_turn_cnt(self, turn_cnt):

    def _input2token(self, inputs: list) -> list:
        """
        inputs(CLS, SEP 토큰 포함)을 tokens로 바꾸기
        :param inputs: JointEnt 모델의 _tokenize() return 값
        :return: inputs에서 CLS, SEP 토큰 제외하고 언더바('_') 제외한 list
        """
        tokens = []
        for e in inputs:
            if e in ['[CLS]', '[SEP]']:
                continue
            tokens.append(e.strip('▁'))

        return tokens

    def __fit_intent(self):
        """
        Intent Classifier를 학습합니다.
        """

        self.intent_classifier.fit(self.dataset.load_intent(self.embed_processor))

    def __fit_entity(self):
        """
        Entity Recognizr를 학습합니다.
        """

        self.entity_recognizer.fit(self.dataset.load_entity(self.embed_processor))

    def __fit_embed(self):
        """
        Embedding을 학습합니다.
        """

        self.embed_processor.fit(self.dataset.load_embed())

    def get_pce_id(self, emotion_name: str, topic_name: str, emotion_prob: float):
        emotion_pce = self.emotion_to_pce[emotion_name]
        topic_pce = self.topic_to_pce[topic_name]
        if emotion_prob >= 0.9:
            strength_pce = 1
        elif emotion_prob <= 0.75:
            strength_pce = 2
        else:
            strength_pce = 3
        pce_code = [emotion_pce, topic_pce, strength_pce]
        pce_code = map(str, pce_code)
        pce_code = "PCE" + ''.join(pce_code)

        return pce_code


def final_emotion(dict_: dict) -> dict:
    """
    컨텐츠 추천을 위한 최종 감정-주제와, 그 확률들을 리턴하는 함수
    :param dict_: 마지막 turn의 return dictionary
    :return: 감정과 주제가 key, 그에 대응하는 확률이 value인 딕셔너리
    """
    emotions = dict_['emotions']
    emotion_prob = dict_['emotion_prob']
    topics = dict_['topics']
    topic_prob = dict_['topic_prob']

    if len(emotions) == 1:
        return {emotions[0]: emotion_prob[0], topics[0]: topic_prob[0]}

    elif len(emotions) == 0:
        return {'NONE_EMOTION':0.0, 'NONE_TOPIC':0.0}

    else:
        # emotion
        if emotions[0] == emotions[1]:
            max_emotion = emotions[0]
            max_emotion_prob = max(emotion_prob)
        else:
            if emotion_prob[0] > emotion_prob[1]:
                max_emotion = emotions[0]
                max_emotion_prob = emotion_prob[0]
            else:
                max_emotion = emotions[1]
                max_emotion_prob = emotion_prob[1]

        # topic
        if topics[0] == topics[1]:
            max_topic = topics[0]
            max_topic_prob = max(topic_prob)
        else:
            if topic_prob[0] > topic_prob[1]:
                max_topic = topics[0]
                max_topic_prob = topic_prob[0]
            else:
                max_topic = topics[0]
                max_topic_prob = topic_prob[1]

        return {max_emotion: max_emotion_prob, max_topic: max_topic_prob}

