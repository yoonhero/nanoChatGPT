# R&E #2 2023 - Make ChatGPT

## TODOS

- [ ] Crawling Good Web Source and Construct Dataset
- [ ] Pretrain GPT with Korean Dataset (NAVER, NAMU, WIKIPEDIA...)
- [ ] Model Improvement (RoFormer, ALBERT, SiLU, GELU, etc)
- [ ] Prepare Prompt for prompt engineering
- [ ] Prompt Translation for Korean Data.
- [ ] Fine-tuning GPT with custom prompt.
- [ ] Evaluation with Various Performance Dataset
- [ ] Simple RLHF.
- [ ] Custom RLHF website.
- [ ] Customize GPT for adversarial training.
- [ ] Evaluation.
- [ ] Make things work!
- [ ] Deploy Custom ChatGPT.

--- 

GPT-3의 성과는 바로 거대한 언어 말뭉치를 신경망이 학습하면 인간이 생성하는 것처럼 그럴듯한 말을 생성할 수 있다는 것이다. 언어란 어떤 개념을 설명하고 인간의 생각과 아이디어를 표현하는 매우 중요한 수단이기 때문에 인공지능이 이러한 언어를 완벽하게 습득하게 된다면 인간과 인공지능의 매끄러운 의사소통이 가능해질 것이다. 


AGI의 시대는 과연 도래할 수 있을까? 단지 단어를 인간과 유사하게 그럴듯하게 생성한다고 그것이 인공지능이 지능을 가지고 생각하여 생성했다고 보기는 어렵다. 하지만 우리가 언어를 배우는 과정을 보면 생각이 살짝 달라진다. 태어나서부터 부모님이 하는 말을 듣고 따라하면서 언어에 대한 학습을 거듭하여 언어를 자유롭게 구사하는 우리처럼 GPT의 방법이 AGI로 가는 방법일 수도 있다. GPT모델이 점점 거대해져서 4, 5, 6까지 진화하게 된다면 결국 이 질문에 대한 해답을 얻을 수 있을 것이다.

GPT-3은 엄청나게 큰 파라미터수로 압도적인 성능을 보였다. 하지만 옳지 않거나 편향적인 글을 아무렇지 않게 생성한다는 것이 가장 큰 문제로 대두하게 되었다. 이를 해결하기 위해서 InstructGPT가 등장하게 되었다. 

## Instruct GPT

우선 사전 학습된 모델을 준비한다. 대량의 언어 코퍼스로 학습된 모델을 고정한 채 <인풋 프롬프트>에 추론하고자 하는 자연어쿼리를 입력한다. 

예) <"너는 굉장하구나"라는 문장의 감정은?> => <긍정>

이 방법은 기존의 fine-tuning과는 다른 방식으로 하나의 태스크를 수행하기 위해 데이터셋을 수집하고 이를 학습시키는 것이 아니라 **모델에게 질문하는 방식**으로 언어 모델의 패러다임이 넘어가고 있다고 한다. 

여기서는 모델에게 질문하는 방법, 즉 "프롬프트를 구성하는 방법"에 따라서 모델의 퀄리티가 좌우되기에 프롬프트 엔지니어링의 중요성이 굉장히 대두되고 있다. 거대한 코퍼스로 사전 학습된 언어 모델을 특정태스크를 잘 수행할 수 있도록 프롬프트 엔지니어링을 수행하는 것이라고 볼 수 있는 것이다.

기존의 GPT 모델은 간접적으로 지시한 프롬프트에 대해서 그럴듯한 글을 써내려갔다. 하지만 InstructGPT는 **직접적**으로 모델에게 지시할 수 있다. 

### RLHF (reinforcement learning with human feedback)

- Step 1: 예제 데이터 수집 후 supervised policy를 학습

    : GPT 모델이 주어진 지시문대로 행동하도록 가르치기 위해서 해당 데이터셋을 만들어서 fine tuning을 실시

    지시 프롬프트와 그에 대한 결과물로 이루어진 데이터셋을 정의. 결과물은 라벨러에 의해서 만들어진다. 

- Step 2: 결과물에 대한 사람의 선호도 데이터를 학습 => Reward Model 확보 

    : Comparison dataset을 사용해 Reward Model 학습 

    Comparison dataset은 프롬프트와 그에 따른 결과물들 (4-9개), 그리고 그 결과에 대한 선호도 순위로 구성된다. 

    Reward Model은 결과물들에 대한 사람의 선호도를 예측하는 방식을호 학습을 진행하는 것이다. 

- Step 3. 강화 학습을 사용해 Reward Model에 대해 policy를 최적화 => InstructGPT

    : Proximal policy optimization algorithm(PPO)를 사용
        - Reward Model을 보상함수로 사용하여 정책을 최적화시키는 방향으로 학습을 진행한다.

        - 1) InstructGPT는 프롬프트를 보고, 그에 대한 결과를 추론한다.
        - 2) 이 결과물을 Reward Model이 평가하여 reward를 계산한다.
        - 3) 보상 값이 InstructGPT에게 주어지고, 모델은 정책을 업데이트하여 사람이 원하는 아웃풋에 더욱 가까운 결과를 내게 된다. 
