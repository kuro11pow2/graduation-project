from enum import Enum

class Env(Enum):
    CARTPOLE = 'CartPole-v1'
    MOUNTAINCAR = 'MountainCar-v0'
    PENDULUM = 'Pendulum-v0'
    ACROBOT = 'Acrobot-v1'
    LUNARLANDER = 'LunarLander-v2'
    CARRACING = 'CarRacing-v0'
    BIPEDALWALKER = 'BipedalWalker-v3'

"""
CARTPOLE

Description:
    카트에 고정되지 않은 막대가 달려있음.
    카트는 마찰 없는 트랙을 움직임.
    진자는 수직으로 시작함.
    목표는 카트의 속도를 조절하여 진자를 땅에 닿지 않게 것

Observation:
    Type: Box(4)
    Num     Observation               Min                     Max
    0       Cart Position             -4.8                    4.8
    1       Cart Velocity             -Inf                    Inf
    2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
    3       Pole Angular Velocity     -Inf                    Inf

Actions:
    Type: Discrete(2)
    Num   Action
    0     Push cart to the left
    1     Push cart to the right
    속도의 증감량은 막대의 위치에 의존한다. 
    왜냐하면 막대의 무게중심이 카트를 움직이는 데 필요한 에너지 양을 증가시키기 때문. 

Reward:
    종료 스텝을 포함하여 매 스텝마다 +1

Starting State:
    시작 시점의 모든 관측값은 [-0.05.05]에서 uniform random value 할당

Episode Termination:
    막대의 각이 12도 이상
    카트의 위치가 2.4 이상 (출력되는 영역의 모서리에 도달)
    에피소드 길이가 200 이상
    
Solved Requirements:
    100회 연속 시행에서 평균 195.0 이상이면 해결로 간주
"""



"""
MOUNTAIN CAR

Description:
    에이전트는 계곡 바닥에서 시작한다.
    주어진 상태에 대해 에이전트는 왼쪽 또는 오른쪽으로 가속하거나, 감속할 수 있다.

Observation:
    Type: Box(2)
    Num    Observation               Min            Max
    0      Car Position              -1.2           0.6
    1      Car Velocity              -0.07          0.07

Actions:
    Type: Discrete(3)
    Num    Action
    0      Accelerate to the Left
    1      Don't accelerate
    2      Accelerate to the Right
    Note: 중력에 의한 가속과 별개로 작용한다.

Reward:
    산 정상의 깃발에 닿으면 +0 (위치 = 0.5)
    위치 < 0.5 이면 -1

Starting State:
    시작 시점의 차의 위치는 [-0.6 , -0.4]에서 uniform random value 할당
    시작 시점의 차의 속력은 항상 0

Episode Termination:
    차 위치가 0.5 이상
    에피소드 길이가 200 이상
"""




"""
LUNAR LANDER

Description:
    달 착륙선을 효율적으로 빠르기 착륙시키기
    * 연료 무한
    * 착륙 지점 (0, 0)

Observation:
    s[0] 수평 좌표
    s[1] 수직 좌표
    s[2] 수평 속도
    s[3] 수직 속도
    s[4] 각도
    s[5] 각속도
    s[6] 1번 다리 충돌 여부
    s[7] 2번 다리 충돌 여부

Actions:
    좌측 엔진 점화
    우측 엔진 점화
    주 엔진 점화
    아무것도 안 하기

Reward:
    화면 상단에서 착륙 지점까지 움직이고 속력을 0으로 하면 +100~140
    착륙 지점에서 멀어질 수록 마이너스
    착륙선이 부서지며 종료하면 -100
    착륙선이 안전하게 종료하면 +100
    다리가 땅에 착지하면 각각 +10
    엔진 점화 시 프레임 당 점수
        주 엔진: -0.3
        측면 엔진: -0.03

    200 점 이상이면 해결
"""
