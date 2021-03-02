# 졸업작품

**주제: 심층강화학습(deep reinforcement learning) 기반 인공지능 개체의 학습 방법 연구**

AlphaGo로 국내에 유명해진 심층강화학습(deep reinforcement learning)은 인간을 포함한 동물이 학습을 하는 과정을 수학적으로 모델링 하여 미래 인공지능기술의 핵심으로 평가되고 있다. 
환경과 에이전트사이의 상호작용 속에서 trial and error를 반복하며 최적의 행동방식(정책)을 학습하는 것을 목표로 한다. 자세한 내용은 다음과 같다.

- OpenAI에서 개발한 Gym에서 주어진 environment를 사용한다. MuJoCo, Classic Control등의 널리 사용되는 single agent 환경에서부터 CarRacing등의 흥미로운 주제까지 다양하게 지원하고 있다.
(단, 영상정보를 state로 받는 Atari와 같은 환경은 졸업논문 주제로 고려하지 않는다.)
https://gym.openai.com/envs/#classic_control

- 또한, multi-agent 환경에 적합한 강화학습 알고리즘 연구를 위한 숨바꼭질 (hide and seek),
Neural MMO 등의 환경도 존재한다. single agent 환경에서의 학습을 성공한 경우, multi-agent
환경으로 확장 하고자 한다. https://openai.com/blog/emergent-tool-use/
https://openai.com/blog/neural-mmo/

- 위의 환경에 적용할 수 있는 (심층)강화학습 알고리즘은 매우 다양하게 존재한다 (예시:
Q-learning, DQN, DDQN, PPO, DDPG, TD3 등). 이러한 알고리즘을 적용 및 변형을 목적으로 한다. 

- Markov Decision Process (MDP)에 기반한 심층강화학습에 대한 전반적인 학습은 교수의 강의가
아닌 “자율적인 그룹 스터디” 방식으로 진행한다. 국내에 다양한 서적과 인터넷자료를
참고하고, 이론적 이해를 위해 다음의 해외 강의 수강을 의무화 한다.

- David silver: https://www.davidsilver.uk/teaching/ 

- Berkeley CS285:
https://www.youtube.com/playlist?list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A

- 최종 졸업작품은 본인이 사용한 심층강화학습 알고리즘의 공학적 응용 방향에 대한 연구를
포함 한다.

- 모든 구현은 Python과 Pytorch를 기반으로 한다. 

**참고정보:**
- 심층강화학습의 대표알고리즘 학습을 위해선 영어 강의 이해능력과 영어논문 독해능력이
반드시 필요하다. 국내 자료가 많지 않기에 언어적 어려움을 극복할 수 있는 학생이 지원하길
권한다.

- 심층강화학습은 다른 머신러닝기법들에 비해 초기학습량이 많고 진입장벽이 높다. 또한
에이전트의 트레이닝 시간도 several day“s”의 수준으로 오래 걸린다. 따라서, 탄탄한 수학적, 그리고 프로그래밍적 이해력과 많은 인내심과 끈기가 요구된다.

- 심층강화학습은 대체로 GPU가 속도향상을 가져오지 못하기에 개인이 소유한 일반적인
desktop/laptop을 활용하여 CPU기반의 학습을 권장한다. 필요시, Google colab을 활용한다.

- 교수와 사전 협의한 학생에 한하여 본 주제는 변형/확대 할 수 있다. 