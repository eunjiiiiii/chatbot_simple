from emotionchat_engine import EmotionChat, final_emotion
import torch
import random
import os
import numpy as np

import sys
print('Python %s on %s' % (sys.version, sys.platform))

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



emotionchat = EmotionChat()

if __name__ == '__main__':
    # initialization
    seed_everything(1234)
    conversation_history = []
    turn_cnt, intent_turn_cnt = 0, 0
    turn_cnts = {'turn_cnt': turn_cnt,
                 'intent_turn_cnt': intent_turn_cnt
                 }
    result_dict = {
        'input': [],
        'intent': '',
        'entity': [],
        'state': '',
        'emotion': '',
        'emotions': [],
        'emotion_prob': [],
        'emotion_probs': [],
        'topics': [],
        'topic_prob': [],
        'answer': '',
        'previous_phase': '',
        'current_phase': '',
        'next_phase': [],
        'intent_turn_cnt': intent_turn_cnt
    }

    #previous_phase = ['/welcomemsg_chat', '/end_chat']
    #previous_phase = None

    while 1:
        sent = input('User: ')
        # wav_file = './exdata/' + sent + '.wav' # 수정
        wav_file = './exdata/test1.wav'
        result_dict = emotionchat.run(sent, wav_file, result_dict, turn_cnts)


        conversation_history.append(result_dict)

        # turn_cnt 1 증가
        turn_cnt += 1
        intent_turn_cnt = result_dict['intent_turn_cnt']

        print("Bot : ")
        for i in range(len(result_dict['answer'])):
            print(result_dict['answer'][i])

        print("-" * 100 + "\n\n")


        # turn_cnt 종류 turn_cnts에 저장
        turn_cnts = {'turn_cnt': turn_cnt,
                     'intent_turn_cnt': intent_turn_cnt
                     }

        # 현재 단계명이 '/end_phase'이면 강제종료
        if result_dict['current_phase'] == '/end_phase':
            break


    # 최대 감정, 주제와 그에 대응하는 확률 출력
    final_emo_topic = final_emotion(result_dict)

    itemlist = final_emo_topic.items()

    print('\n' + '-'*100 + '-' * 100)
    print('<최종 딕셔너리에서 최대 감정, 주제와 그에 대응하는 확률 출력>')
    print(itemlist)
    print('-' * 100 + "-" * 100 + "\n\n")